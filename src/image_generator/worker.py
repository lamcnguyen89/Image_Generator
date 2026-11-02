"""One durable local queue worker; model runtimes are imported only for a job."""

import argparse
from copy import deepcopy
import fcntl
import logging
import math
import os
from pathlib import Path
import signal
from threading import Event, Thread
from time import monotonic
from uuid import uuid4

from .generation import save_batch
from .job_requests import decode_request, verify_fingerprints
from .jobs import JobStore
from .workflows.registry import WORKFLOWS, create_workflow


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WorkerAlreadyRunning(RuntimeError):
    pass


class WorkerCleanupError(RuntimeError):
    pass


class Worker:
    """Own one workflow at a time, with an exclusive OS lock for the whole lifetime.

    A heartbeat is observational: it never authorizes another worker to steal a
    live GPU process. The OS releases the lock on process death; only a new lock
    holder can recover running jobs. Factory injection is for tests, not job data.
    """

    def __init__(self, store: JobStore, output_dir: str | Path, *, video_output_dir: str | Path = PROJECT_ROOT / "Videos",
                 workflow_factory=None,
                 runtime_options=None, poll_interval=0.5, heartbeat_interval=2.0):
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in (poll_interval, heartbeat_interval)):
            raise ValueError("Worker poll and heartbeat intervals must be positive")
        self.store = store
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.video_output_dir = Path(video_output_dir).expanduser().resolve()
        self.workflow_factory = workflow_factory or create_workflow
        self.runtime_options = deepcopy(runtime_options or {})
        self.poll_interval, self.heartbeat_interval = poll_interval, heartbeat_interval
        self.id = uuid4().hex
        self.current_job_id = None
        self._workflow = self._workflow_name = None
        self._lock = self._heartbeat_thread = None
        self._stop = Event()
        self._heartbeat_stop = Event()
        self._heartbeat_failure = None
        self._started = self._closed = False
        self._last_progress = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def start(self):
        if self._closed:
            raise RuntimeError("Create a new Worker after closing an instance")
        if self._started:
            return self
        stream = self.store.database.with_name(self.store.database.name + ".worker.lock").open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise WorkerAlreadyRunning(f"A worker already owns queue {self.store.database}") from None
        except BaseException:
            stream.close()
            raise
        self._lock = stream
        try:
            self.store._start_worker(self.id, os.getpid())
            self._heartbeat_thread = Thread(target=self._heartbeat_loop, name="job-worker-heartbeat", daemon=True)
            self._heartbeat_thread.start()
            self._started = True
        except BaseException:
            try:
                self.store._stop_worker(self.id)
            finally:
                self._release_lock()
            raise
        LOGGER.info("Worker %s started", self.id)
        return self

    def request_stop(self):
        """Finish the current job, then stop claiming new work."""
        self._stop.set()

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(self.heartbeat_interval):
            try:
                self.store._heartbeat(self.id)
            except Exception as error:
                self._heartbeat_failure = error
                self.request_stop()
                return

    def _check_health(self):
        if self._heartbeat_failure is not None:
            raise RuntimeError("Worker heartbeat failed; stopping this worker") from self._heartbeat_failure

    def _close_workflow(self):
        workflow, self._workflow = self._workflow, None
        self._workflow_name = None
        if workflow is not None:
            try:
                workflow.set_progress_callback(None)
                workflow.close()
            except BaseException as error:
                self.request_stop()
                raise WorkerCleanupError("Workflow cleanup failed; a new model will not be loaded") from error

    def _ensure_workflow(self, name):
        if self._workflow is None or self._workflow_name != name:
            self._close_workflow()
            self._workflow = self.workflow_factory(name, **self.runtime_options.get(name, {}))
            self._workflow_name = name
        return self._workflow

    def _process_model_unloads(self):
        identifiers = self.store._pending_model_unloads(self.id)
        if not identifiers:
            return
        try:
            self._close_workflow()
        except WorkerCleanupError:
            self.store._finish_model_unloads(self.id, identifiers, "Model cleanup failed. Restart the generation worker.")
            raise
        self.store._finish_model_unloads(self.id, identifiers)
        LOGGER.info("Worker %s cleared loaded models", self.id)

    def _report_progress(self, job_id, event):
        self._check_health()
        now = monotonic()
        # Persist stage transitions and final steps immediately. Bound SQLite
        # writes during fast denoising to five updates per second.
        if (self._last_progress is not None and self._last_progress[0] == event.stage
                and now - self._last_progress[1] < 0.2
                and not (event.total is not None and event.current == event.total)):
            return
        self.store._progress(job_id, self.id, event.stage, event.current, event.total)
        self._last_progress = (event.stage, now)

    def run_once(self):
        """Claim and finish at most one job. Use within the Worker context."""
        if not self._started:
            raise RuntimeError("Start the worker or enter its context before processing jobs")
        self._check_health()
        if self._stop.is_set():
            return None
        self._process_model_unloads()
        job = self.store._claim_next(self.id)
        if job is None:
            return None
        self.current_job_id = job.id
        LOGGER.info("Job %s running (%s)", job.id, job.workflow)
        try:
            request = decode_request(job.request)
            if request.model.defaults["workflow"] != job.workflow:
                raise ValueError("Queued workflow does not match its request snapshot")
            verify_fingerprints(job.fingerprints)
            self._check_health()
            self.store._progress(job.id, self.id, "preparing")
            workflow = self._ensure_workflow(job.workflow)
            self._last_progress = None
            workflow.set_progress_callback(lambda event: self._report_progress(job.id, event))
            workflow.validate(request)
            self.store._progress(job.id, self.id, "generating")
            iterator = iter(workflow.generate_batches(request))
            count = 0
            try:
                for batch in iterator:
                    self._check_health()
                    save_stage = "encoding_video" if batch.videos else "saving"
                    self.store._progress(job.id, self.id, save_stage, count)
                    root = self.video_output_dir if batch.videos else self.output_dir
                    directory = root / job.id
                    if directory.is_symlink():
                        raise ValueError("Job output directory must not be a symlink")
                    saved = save_batch(batch, directory)
                    self.store._record_outputs(job.id, self.id, saved)
                    count += len(saved)
                    self.store._progress(job.id, self.id, "saving", count)
                    del batch
                    self.store._progress(job.id, self.id, "generating", count)
            finally:
                close_iterator = getattr(iterator, "close", None)
                if close_iterator is not None:
                    close_iterator()
            self._check_health()
            result = self.store._finish(job.id, self.id, "completed")
        except Exception as error:
            cleanup_error = None
            try:
                self._close_workflow()
            except WorkerCleanupError as failure:
                cleanup_error = failure
            result = self.store._finish(job.id, self.id, "failed", f"{type(error).__name__}: {error}")
            LOGGER.warning("Job %s failed (%s)", job.id, type(error).__name__)
            if cleanup_error is not None:
                raise cleanup_error
            if isinstance(error, WorkerCleanupError):
                raise
        except BaseException as error:
            try:
                self._close_workflow()
            finally:
                self.store._finish(job.id, self.id, "interrupted", f"{type(error).__name__}: worker execution interrupted")
            raise
        finally:
            if self._workflow is not None:
                self._workflow.set_progress_callback(None)
            self.current_job_id = None
        self._process_model_unloads()
        LOGGER.info("Job %s %s", job.id, result.status)
        return result

    def run(self, *, once=False, until_idle=False):
        """Run continuously, once, or until the pending queue is empty."""
        with self:
            while not self._stop.is_set():
                job = self.run_once()
                if once or (until_idle and job is None):
                    break
                if job is None:
                    self._stop.wait(self.poll_interval)
            self._check_health()

    def _release_lock(self):
        if self._lock is not None:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None

    def close(self):
        if self._closed:
            return
        self.request_stop()
        self._heartbeat_stop.set()
        try:
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join()
            self._close_workflow()
        finally:
            try:
                if self._started:
                    self.store._stop_worker(self.id)
            finally:
                self._started = False
                self._closed = True
                self._release_lock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data/jobs.sqlite3")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Images")
    parser.add_argument("--video-output-dir", type=Path, help="Video output root (default: Videos)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="process at most one job, then exit")
    mode.add_argument("--drain", action="store_true", help="exit when no queued jobs remain")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--heartbeat-interval", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--comfyui-dir", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        try:
            from dotenv import load_dotenv
        except ImportError:
            pass
        else:
            load_dotenv(PROJECT_ROOT / ".env")
        cache_root = Path(os.getenv("CACHE_LOCATION") or PROJECT_ROOT / "Models").expanduser()
        if not cache_root.is_absolute():
            cache_root = PROJECT_ROOT / cache_root
        options = {"device": args.device, "dtype": args.dtype,
                   "cache_dir": args.cache_dir if args.cache_dir is not None else cache_root / "hub",
                   "local_files_only": args.local_files_only,
                   "token": os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or None}
        runtime_options = {name: dict(options) for name in WORKFLOWS}
        runtime_options["minimax"]["comfyui_dir"] = args.comfyui_dir
        video_output_dir = args.video_output_dir or os.getenv("IMAGE_GENERATOR_JOB_VIDEO_OUTPUT_ROOT") or PROJECT_ROOT / "Videos"
        worker = Worker(JobStore(args.database), args.output_dir, video_output_dir=video_output_dir,
                        runtime_options=runtime_options,
                        poll_interval=args.poll_interval, heartbeat_interval=args.heartbeat_interval)
        previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            for number in previous:
                signal.signal(number, lambda signum, frame: worker.request_stop())
            worker.run(once=args.once, until_idle=args.drain)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        LOGGER.error("Worker stopped: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

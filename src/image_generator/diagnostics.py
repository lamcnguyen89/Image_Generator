"""Bounded process-safe logs and optional, lazy runtime instrumentation.

No GPU libraries are imported here. Resource probes must never prevent generation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import fcntl
from functools import wraps
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import sys
from threading import Event, Thread
from time import perf_counter
from uuid import uuid4


LOG_ROOT = Path(__file__).resolve().parents[2] / "data/logs"
FILE_BYTES = 5_000_000
FILE_COUNT = 15  # Active file + 14 backups = at most 75 MB (decimal).
LOGGER = logging.getLogger(__name__)
_CONTEXT = ContextVar("diagnostic_context", default={})


def _redact(value):
    for name in ("HUGGINGFACE_TOKEN", "HF_TOKEN", "IMAGE_GENERATOR_ACCESS_PASSWORD", "IMAGE_GENERATOR_SECRET_KEY"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"\bhf_[A-Za-z0-9]+", "[REDACTED]", value)
    return re.sub(r"(?i)([?&](?:token|access_token|api_key|password)=)[^&\s\"']+", r"\1[REDACTED]", value)


def _safe_payload(value):
    if isinstance(value, dict):
        return {key: _safe_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_payload(item) for item in value]
    if isinstance(value, str):
        return _redact(value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname, "logger": record.name,
            "pid": record.process, "thread": record.threadName,
            **_CONTEXT.get(), **getattr(record, "diagnostic_fields", {}),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(_safe_payload(payload), ensure_ascii=False, default=str)


class BoundedLogHandler(logging.Handler):
    """Serialize append/rotation across web and worker processes using flock.

    Open the active file only under the lock so no process writes to a renamed
    backup. This application already requires Linux for its queue worker.
    """

    def __init__(self, directory, *, file_bytes=FILE_BYTES, file_count=FILE_COUNT):
        super().__init__()
        if file_bytes < 512 or file_count < 1:
            raise ValueError("Log files require at least 512 bytes and one file")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file_bytes, self.file_count = file_bytes, file_count
        self.path = self.directory / "application.log"
        self.setFormatter(JsonFormatter())
        with self._locked():
            # Clean up owned backups from older retention configurations.
            valid = {"", *(f".{index}" for index in range(1, file_count))}
            for path in self.directory.glob("application.log*"):
                suffix = path.name.removeprefix("application.log")
                if suffix and not re.fullmatch(r"\.[0-9]+", suffix):
                    continue
                if suffix not in valid or path.stat().st_size > file_bytes:
                    path.unlink()

    @contextmanager
    def _locked(self):
        with (self.directory / ".rotation.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def emit(self, record):
        try:
            formatted = self.format(record)
            data = (formatted + "\n").encode("utf-8")
            if len(data) > self.file_bytes:
                # Preserve valid JSON even for enormous exceptions or Unicode.
                data = (json.dumps({"event": "log_record_truncated", "original_bytes": len(data),
                                    "message": formatted.encode("utf-8")[:self.file_bytes // 8].decode("utf-8", "replace")},
                                   ensure_ascii=False) + "\n").encode("utf-8")
            with self._locked():
                if self.path.exists() and self.path.stat().st_size + len(data) > self.file_bytes:
                    for index in range(self.file_count - 1, 0, -1):
                        source = self.path if index == 1 else self.directory / f"application.log.{index - 1}"
                        if source.exists():
                            source.replace(self.directory / f"application.log.{index}")
                    if self.file_count == 1:
                        self.path.unlink()
                with self.path.open("ab") as stream:
                    stream.write(data)
        except Exception:
            self.handleError(record)


def configure_logging(directory=None, level=None):
    """Idempotently install application-owned handlers, preserving host handlers."""
    directory = Path(directory or os.getenv("IMAGE_GENERATOR_LOG_DIR") or LOG_ROOT).expanduser().resolve()
    level = (level or os.getenv("IMAGE_GENERATOR_LOG_LEVEL", "INFO")).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("IMAGE_GENERATOR_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    root = logging.getLogger()
    owned = [handler for handler in root.handlers if isinstance(handler, BoundedLogHandler)]
    if len(owned) == 1 and owned[0].directory == directory:
        owned[0].setLevel(level)
    else:
        handler = BoundedLogHandler(directory)
        handler.setLevel(level)
        for previous in owned:
            root.removeHandler(previous)
            previous.close()
        if not root.handlers:
            console = logging.StreamHandler()
            console.setFormatter(JsonFormatter())
            console.setLevel(logging.INFO)
            root.addHandler(console)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("image_generator").setLevel(level)
    event("logging_configured", directory=str(directory), max_total_bytes=FILE_BYTES * FILE_COUNT,
          file_bytes=FILE_BYTES, file_count=FILE_COUNT, python=platform.python_version(),
          platform=platform.platform(), resources=resource_snapshot())


def event(name, *, level=logging.INFO, exc_info=False, **fields):
    LOGGER.log(level, name, extra={"diagnostic_fields": {**_CONTEXT.get(), "event": name, **fields}}, exc_info=exc_info)


@contextmanager
def log_context(**fields):
    token = _CONTEXT.set({**_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def resource_snapshot(device=None):
    """Read RAM/CPU and already-imported CUDA state, without synchronizing GPU work."""
    result = {"cuda_requested": str(device).startswith("cuda") if device is not None else None}
    try:
        import psutil

        process = psutil.Process()
        with process.oneshot():
            memory, cpu = process.memory_info(), process.cpu_times()
            result.update(process_rss_bytes=memory.rss, process_vms_bytes=memory.vms,
                          process_cpu_seconds=cpu.user + cpu.system, process_threads=process.num_threads())
        ram = psutil.virtual_memory()
        result.update(ram_total_bytes=ram.total, ram_available_bytes=ram.available,
                      ram_used_percent=ram.percent, swap_used_bytes=psutil.swap_memory().used)
    except Exception as error:
        result["ram_probe_error"] = type(error).__name__
    torch = sys.modules.get("torch")
    result["torch_loaded"] = torch is not None
    if torch is not None:
        try:
            cuda = torch.cuda
            result.update(torch_version=str(torch.__version__), cuda_build=getattr(torch.version, "cuda", None),
                          cuda_available=bool(cuda.is_available()), cuda_initialized=bool(cuda.is_initialized()))
            if result["cuda_initialized"] and result["cuda_available"]:
                selected = device if device and str(device).startswith("cuda") else cuda.current_device()
                properties = cuda.get_device_properties(selected)
                result.update(cuda_device=str(selected), gpu_name=properties.name,
                              gpu_total_bytes=properties.total_memory,
                              gpu_allocated_bytes=cuda.memory_allocated(selected),
                              gpu_reserved_bytes=cuda.memory_reserved(selected),
                              gpu_peak_allocated_bytes=cuda.max_memory_allocated(selected),
                              gpu_peak_reserved_bytes=cuda.max_memory_reserved(selected))
        except Exception as error:
            result["cuda_probe_error"] = type(error).__name__
    return result


def measured_operation(name):
    """Instrument explicit runtime/model ownership boundaries, including failures."""
    def decorate(method):
        @wraps(method)
        def measured(self, *args, **kwargs):
            fields = {"workflow": getattr(self, "workflow_id", type(self).__name__),
                      "device": getattr(self, "device", None), "dtype": getattr(self, "dtype", None)}
            started = perf_counter()
            event(name + "_started", **fields, pipeline_loaded=getattr(self, "_pipeline", None) is not None,
                  resources=resource_snapshot(fields["device"]))
            try:
                result = method(self, *args, **kwargs)
            except BaseException:
                event(name + "_failed", level=logging.ERROR, exc_info=True, **fields,
                      elapsed_seconds=perf_counter() - started, resources=resource_snapshot(fields["device"]))
                raise
            event(name + "_completed", **fields, elapsed_seconds=perf_counter() - started,
                  pipeline_loaded=getattr(self, "_pipeline", None) is not None,
                  resources=resource_snapshot(fields["device"]))
            return result
        return measured
    return decorate


class StageTimer:
    def __init__(self, workflow):
        self.workflow = workflow
        self.stage = None
        self.started = self.last_step = perf_counter()

    def progress(self, stage, current=None, total=None):
        now = perf_counter()
        if stage != self.stage:
            self.finish()
            self.stage, self.started = stage, now
            event("stage_started", workflow=self.workflow, stage=stage, current=current, total=total,
                  resources=resource_snapshot())
        if now - self.last_step >= 1 or (total is not None and current == total):
            event("stage_progress", level=logging.DEBUG, workflow=self.workflow, stage=stage,
                  current=current, total=total, elapsed_seconds=now - self.started)
            self.last_step = now

    def finish(self):
        if self.stage is not None:
            event("stage_finished", workflow=self.workflow, stage=self.stage,
                  elapsed_seconds=perf_counter() - self.started)
            self.stage = None


class GenerationTelemetry:
    """Correlate a request and sample resources even while model calls block."""
    def __init__(self, *, job_id=None, workflow=None, device=None):
        self.fields = {"job_id": job_id or uuid4().hex, "workflow": workflow, "device": device}
        self.images = self.videos = self.batches = 0
        self.frames = 0
        self.stop = Event()
        self.thread = None

    def __enter__(self):
        self.context = log_context(**self.fields)
        self.context.__enter__()
        self.started = perf_counter()
        event("generation_started", resources=resource_snapshot(self.fields["device"]))
        try:
            interval = float(os.getenv("IMAGE_GENERATOR_RESOURCE_INTERVAL", "5"))
            if not math.isfinite(interval) or interval < 0:
                raise ValueError
        except ValueError:
            event("invalid_resource_interval", level=logging.WARNING, using_seconds=5)
            interval = 5
        if interval:
            def monitor():
                with log_context(**self.fields):
                    while not self.stop.wait(interval):
                        event("resource_sample", elapsed_seconds=perf_counter() - self.started,
                              resources=resource_snapshot(self.fields["device"]))
            self.thread = Thread(target=monitor, name="generation-resources", daemon=True)
            self.thread.start()
        return self

    def request(self, request):
        # Explicit allowlist: prompts, inputs and credentials never get serialized.
        names = ("width", "height", "num_frames", "fps", "num_inference_steps", "num_images_per_prompt",
                 "num_videos_per_prompt", "guidance_scale", "seed", "sampling")
        event("generation_parameters", model_id=request.model.id, mode=request.mode,
              parameters={name: request.settings[name] for name in names if name in request.settings},
              loras=[{"id": item.asset.id, "strength": item.strength} for item in request.loras])

    def batch_saved(self, batch, *, save_seconds):
        self.batches += 1
        images, videos = len(batch.images), len(batch.videos)
        self.images += images
        self.videos += videos
        runtime = batch.metadata.get("runtime", {})
        seconds = runtime.get("generation_seconds")
        frames = sum(len(video) for video in batch.videos)
        self.frames += frames
        runtime_names = ("backend", "device", "dtype", "precision", "torch_version", "diffusers_version",
                         "comfyui_revision", "model_reused", "loras_reset")
        event("batch_saved", batch=self.batches, images=images, videos=videos, frames=frames,
              seeds=batch.metadata.get("seeds"), generation_seconds=seconds, save_seconds=save_seconds,
              runtime={name: runtime[name] for name in runtime_names if name in runtime},
              outputs_per_second=(images + videos) / seconds if isinstance(seconds, (int, float)) and seconds > 0 else None,
              frames_per_second=frames / seconds if frames and isinstance(seconds, (int, float)) and seconds > 0 else None,
              resources=resource_snapshot(self.fields["device"]))

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        try:
            elapsed = perf_counter() - self.started
            event("generation_finished", level=logging.ERROR if exc_type else logging.INFO,
                  status=("interrupted" if not issubclass(exc_type, Exception) else "failed") if exc_type else "completed",
                  images=self.images, videos=self.videos, frames=self.frames,
                  batches=self.batches, elapsed_seconds=elapsed,
                  outputs_per_second=(self.images + self.videos) / elapsed if elapsed > 0 else None,
                  resources=resource_snapshot(self.fields["device"]),
                  exc_info=(exc_type, exc, traceback) if exc_type else False)
        finally:
            self.context.__exit__(exc_type, exc, traceback)


def install_request_logging(app):
    """Log route names and timing, excluding URLs, queries, bodies and headers."""
    from flask import g, request

    @app.before_request
    def start_request_timer():
        g.diagnostic_started = perf_counter()

    @app.after_request
    def record_request(response):
        started = getattr(g, "diagnostic_started", None)
        event("http_request", method=request.method, endpoint=request.endpoint,
              status_code=response.status_code,
              elapsed_seconds=perf_counter() - started if started is not None else None,
              level=logging.WARNING if response.status_code >= 400 else logging.DEBUG)
        return response

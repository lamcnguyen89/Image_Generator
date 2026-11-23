"""Exercise disk bounds, concurrent writers and generation telemetry without CUDA."""

import json
from contextlib import redirect_stdout
import io
import logging
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from image_generator import diagnostics as diag
from image_generator.jobs import JobStore
from image_generator.worker import Worker
from image_generator.generation import run_generation
from image_generator.workflows.base import WorkflowProgress
from queue_fakes import FakeWorkflow, make_request


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.logger = logging.Logger("diagnostics-test", logging.DEBUG)

    def handler(self, **kwargs):
        handler = diag.BoundedLogHandler(self.root, **kwargs)
        self.logger.addHandler(handler)
        self.addCleanup(handler.close)
        return handler

    def records(self):
        return [json.loads(line) for path in self.root.glob("application.log*")
                for line in path.read_text().splitlines()]

    def test_retention_is_bounded_in_bytes_and_keeps_latest_records_after_restart(self):
        self.handler(file_bytes=1200, file_count=3)
        for number in range(80):
            self.logger.info("entry %d %s", number, "雪" * 60)
        paths = list(self.root.glob("application.log*"))
        self.assertEqual(len(paths), 3)
        self.assertLessEqual(sum(path.stat().st_size for path in paths), 3600)
        self.assertTrue(all(path.stat().st_size <= 1200 for path in paths))
        self.assertIn("entry 79", json.loads((self.root / "application.log").read_text().splitlines()[-1])["message"])
        before = {path: path.read_bytes() for path in paths}
        diag.BoundedLogHandler(self.root, file_bytes=1200, file_count=3).close()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.assertEqual(diag.FILE_BYTES * diag.FILE_COUNT, 75_000_000)

    def test_enormous_records_stay_valid_json_within_single_file_limit(self):
        self.handler(file_bytes=512, file_count=2)
        self.logger.error("\n\"雪" * 50000)
        self.assertLessEqual((self.root / "application.log").stat().st_size, 512)
        self.assertEqual(self.records()[0]["event"], "log_record_truncated")
        self.logger.info("following entry")
        self.assertTrue(any(item["message"] == "following entry" for item in self.records()))

    def test_previous_larger_retention_is_pruned_without_touching_unrelated_files(self):
        for suffix in ("", ".0", ".01", ".1", ".2", ".9"):
            (self.root / ("application.log" + suffix)).write_bytes(b"x" * (2000 if not suffix else 10))
        other = self.root / "other.log"
        other.write_text("keep")
        self.handler(file_bytes=1000, file_count=2)
        self.assertEqual({p.name for p in self.root.glob("application.log*")}, {"application.log.1"})
        self.assertEqual(other.read_text(), "keep")

    def run_writers(self, file_bytes, file_count):
        code = """
import logging, sys
from image_generator.diagnostics import BoundedLogHandler
logger = logging.Logger('concurrent')
logger.addHandler(BoundedLogHandler(sys.argv[1], file_bytes=int(sys.argv[2]), file_count=int(sys.argv[3])))
for number in range(100):
    logger.warning('writer=%s entry=%d', sys.argv[4], number)
"""
        processes = [subprocess.Popen([sys.executable, "-S", "-c", code, str(self.root),
                                      str(file_bytes), str(file_count), str(writer)],
                                     env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE) for writer in range(3)]
        for process in processes:
            _, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(stderr, b"")

    def test_concurrent_processes_do_not_lose_or_corrupt_records(self):
        self.run_writers(200000, 3)
        records = self.records()
        self.assertEqual(len(records), 300)
        self.assertEqual(len({item["message"] for item in records}), 300)

    def test_concurrent_rotation_obeys_combined_budget(self):
        self.run_writers(1500, 3)
        paths = list(self.root.glob("application.log*"))
        self.assertEqual(len(paths), 3)
        self.assertLessEqual(sum(path.stat().st_size for path in paths), 4500)
        self.assertGreater(len(self.records()), 0)

    def test_formatting_redacts_credentials_even_with_quotes_and_newlines(self):
        self.handler()
        secret = 'private"password\nvalue'
        with patch.dict(os.environ, IMAGE_GENERATOR_ACCESS_PASSWORD=secret, HF_TOKEN="secret-token"):
            self.logger.error("%s %s hf_anotherSecret https://example.test/?token=url-secret", secret, "secret-token")
        entry = self.records()[0]
        self.assertNotIn(secret, entry["message"])
        self.assertNotIn("url-secret", entry["message"])
        self.assertNotIn("secret-token", entry["message"])
        self.assertEqual(entry["level"], "ERROR")
        self.assertIn("[REDACTED]", entry["message"])

    def test_cuda_probe_reports_selected_device_without_initializing_cuda(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.is_initialized.return_value = False
        torch = SimpleNamespace(cuda=cuda, __version__="test", version=SimpleNamespace(cuda="test-cuda"))
        with patch.dict(sys.modules, torch=torch):
            snapshot = diag.resource_snapshot("cuda:1")
            self.assertTrue(snapshot["cuda_available"])
            self.assertFalse(snapshot["cuda_initialized"])
            cuda.get_device_properties.assert_not_called()
            cuda.is_initialized.return_value = True
            cuda.get_device_properties.return_value = SimpleNamespace(name="Test GPU", total_memory=1234)
            snapshot = diag.resource_snapshot("cuda:1")
        cuda.get_device_properties.assert_called_once_with("cuda:1")
        cuda.memory_allocated.assert_called_once_with("cuda:1")
        self.assertEqual(snapshot["gpu_name"], "Test GPU")

    def test_probe_failures_do_not_break_generation_or_import_gpu_libraries(self):
        code = """
import sys
from image_generator.diagnostics import resource_snapshot
snapshot = resource_snapshot()
assert snapshot['torch_loaded'] is False
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'flask', 'transformers'))
"""
        result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        with patch.dict(sys.modules, psutil=SimpleNamespace(Process=Mock(side_effect=OSError("unavailable")))):
            self.assertEqual(diag.resource_snapshot()["ram_probe_error"], "OSError")

    def test_worker_logs_correlated_totals_and_failure_tracebacks_without_prompts(self):
        store = JobStore(self.root / "jobs.sqlite3")
        request = make_request(self.root, prompt="private prompt")
        job = store.enqueue(request)
        with self.assertLogs(diag.LOGGER, logging.INFO) as logs:
            Worker(store, self.root / "outputs", workflow_factory=lambda name, **options: FakeWorkflow(name)).run(once=True)
        payloads = [json.loads(diag.JsonFormatter().format(record)) for record in logs.records]
        summary = next(record for record in logs.records if record.diagnostic_fields.get("event") == "generation_finished")
        self.assertEqual(summary.diagnostic_fields["images"], 2)
        self.assertEqual(summary.diagnostic_fields["status"], "completed")
        self.assertEqual(summary.diagnostic_fields["job_id"], job.id)
        self.assertNotIn("private prompt", json.dumps(payloads))
        self.assertEqual(store.get(job.id).status, "completed")
        store.enqueue(make_request(self.root, failure="generation"))
        with self.assertLogs(diag.LOGGER, logging.ERROR) as logs:
            Worker(store, self.root / "outputs", workflow_factory=lambda name, **options: FakeWorkflow(name)).run(once=True)
        summary = next(record for record in logs.records if record.diagnostic_fields.get("event") == "generation_finished")
        self.assertEqual(summary.diagnostic_fields["status"], "failed")
        self.assertIn("fake generation failed", diag.JsonFormatter().format(summary))
        self.assertIsNotNone(summary.exc_info)

    def test_cli_generation_and_video_batch_statistics(self):
        request = make_request(self.root)
        with patch("image_generator.generation.create_workflow", return_value=FakeWorkflow()), \
                self.assertLogs(diag.LOGGER, logging.INFO) as logs:
            saved = run_generation(request, self.root / "outputs", device="cpu")
        self.assertEqual(len(saved), 2)
        self.assertTrue(any(record.diagnostic_fields.get("event") == "batch_saved" for record in logs.records))
        batch = SimpleNamespace(images=(), videos=([1] * 8, [1] * 8),
                                metadata={"runtime": {"generation_seconds": 4}, "seeds": [1, 2]})
        with self.assertLogs(diag.LOGGER, logging.INFO) as logs, diag.GenerationTelemetry(workflow="video") as telemetry:
            telemetry.batch_saved(batch, save_seconds=0.1)
        fields = next(record.diagnostic_fields for record in logs.records if record.diagnostic_fields["event"] == "batch_saved")
        self.assertEqual(fields["frames_per_second"], 4)
        self.assertEqual(fields["videos"], 2)

    def test_stage_timing_preserves_progress_callback_and_finishes_last_stage(self):
        workflow = FakeWorkflow()
        received = []
        workflow.set_progress_callback(received.append)
        with self.assertLogs(diag.LOGGER, logging.DEBUG) as logs:
            workflow.report_progress("loading_model")
            workflow.report_progress("denoising", 0, 2)
            workflow.report_progress("denoising", 2, 2)
            workflow.set_progress_callback(None)
        self.assertEqual(received, [WorkflowProgress("loading_model"), WorkflowProgress("denoising", 0, 2),
                                    WorkflowProgress("denoising", 2, 2)])
        finished = [record.diagnostic_fields for record in logs.records
                    if record.diagnostic_fields["event"] == "stage_finished"]
        self.assertEqual([item["stage"] for item in finished], ["loading_model", "denoising"])
        self.assertTrue(all(item["elapsed_seconds"] >= 0 for item in finished))

    def test_request_logging_excludes_queries_bodies_and_authentication(self):
        from flask import Flask

        app = Flask(__name__)
        app.testing = True
        diag.install_request_logging(app)

        @app.post("/test")
        def reject():
            return "rejected", 422

        with self.assertLogs(diag.LOGGER, logging.DEBUG) as logs:
            response = app.test_client().post("/test?prompt=private-query", data={"prompt": "private-body"},
                                              headers={"Authorization": "Bearer private-header"})
        self.assertEqual(response.status_code, 422)
        payload = json.loads(diag.JsonFormatter().format(logs.records[0]))
        self.assertEqual(payload["endpoint"], "reject")
        self.assertEqual(payload["status_code"], 422)
        self.assertGreaterEqual(payload["elapsed_seconds"], 0)
        self.assertNotIn("private-", json.dumps(payload))

    def test_cli_dry_run_reads_logging_settings_without_loading_application_environment(self):
        from image_generator.cli import run_family

        catalog = self.root / "catalog"
        shutil.copytree(ROOT / "tests/fixtures/model_families", catalog)
        (catalog / "SDXL/checkpoints/demo/missing-fixture.safetensors").write_bytes(b"test weights")
        original = dict(os.environ)
        with patch("dotenv.dotenv_values", return_value={"IMAGE_GENERATOR_LOG_DIR": str(self.root / "logs"),
                                                        "IMAGE_GENERATOR_LOG_LEVEL": "DEBUG"}), \
                patch("dotenv.load_dotenv") as load, patch("image_generator.cli.configure_logging") as configure, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_family("sdxl", ["--root", str(catalog), "--model-id", "fixture-sdxl",
                                                 "--prompt", "test", "--no-loras", "--dry-run"]), 0)
        load.assert_not_called()
        configure.assert_called_once_with(str(self.root / "logs"), "DEBUG")
        self.assertEqual(dict(os.environ), original)

    def test_periodic_sampling_stops_and_context_does_not_leak(self):
        sampled = Event()
        original = diag.event
        def observe(name, **fields):
            if name == "resource_sample":
                sampled.set()
            original(name, **fields)
        with patch.dict(os.environ, IMAGE_GENERATOR_RESOURCE_INTERVAL="0.01"), \
                patch.object(diag, "event", side_effect=observe), self.assertLogs(diag.LOGGER, logging.INFO) as logs:
            with diag.GenerationTelemetry(job_id="test-job", workflow="test") as telemetry:
                self.assertTrue(sampled.wait(2))
            self.assertFalse(telemetry.thread.is_alive())
            self.assertEqual(diag._CONTEXT.get(), {})
        self.assertTrue(all(record.diagnostic_fields["job_id"] == "test-job" for record in logs.records))

    def test_configuration_is_idempotent_and_preserves_existing_handlers(self):
        root_logger = logging.getLogger()
        previous_handlers = root_logger.handlers[:]
        previous_level = root_logger.level
        app_logger = logging.getLogger("image_generator")
        previous_app_level = app_logger.level
        try:
            diag.configure_logging(self.root, "DEBUG")
            diag.configure_logging(self.root, "DEBUG")
            handlers = [h for h in root_logger.handlers if isinstance(h, diag.BoundedLogHandler)]
            self.assertEqual(len(handlers), 1)
            for handler in previous_handlers:
                if not isinstance(handler, diag.BoundedLogHandler):
                    self.assertIn(handler, root_logger.handlers)
        finally:
            for handler in root_logger.handlers:
                if handler not in previous_handlers:
                    handler.close()
            root_logger.handlers[:] = previous_handlers
            root_logger.setLevel(previous_level)
            app_logger.setLevel(previous_app_level)


if __name__ == "__main__":
    unittest.main()

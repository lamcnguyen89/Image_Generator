"""SDXL HTTP submission through the real worker/workflow with a mocked GPU pipeline."""

from contextlib import nullcontext
from itertools import count
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation
from image_generator.jobs import JobStore
from image_generator.worker import Worker
from test_generation_ui import FormPage
from test_sdxl import AncestralScheduler, DefaultScheduler, DPMScheduler, EulerScheduler, FakeGenerator


class Pipeline:
    def __init__(self, owner, source):
        self.owner, self.source = owner, source
        self.scheduler = DefaultScheduler()
        self.to = Mock(return_value=self)
        self.adapters = []
        self.load_lora_weights = Mock(side_effect=lambda path, **options: self.adapters.append(options["adapter_name"]))
        self.unload_lora_weights = Mock(side_effect=self.adapters.clear)
        self.set_adapters = Mock()

    def __call__(self, **arguments):
        self.owner.calls.append({"pipeline": self, "source": self.source, "adapters": tuple(self.adapters),
                                 "strengths": self.set_adapters.call_args, "arguments": arguments})
        if arguments["prompt"] == "fail generation":
            raise RuntimeError("SDXL test inference failed")
        self.num_timesteps = arguments["num_inference_steps"]
        for step in range(self.num_timesteps):
            callback = arguments.get("callback_on_step_end")
            if callback:
                values = {}
                if callback(self, step, step, values) is not values:
                    raise AssertionError("Step callbacks must return their tensor dictionary unchanged")
            if self.owner.on_step:
                self.owner.on_step(step)
        return SimpleNamespace(images=[Image.new("RGB", (16, 16), "navy")
                                       for _ in range(arguments["num_images_per_prompt"])])


class SDXLExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog_root = self.root / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.catalog_root)
        for asset in discover_catalog(self.catalog_root).models.values():
            if asset.family == "sdxl":
                asset.asset_paths["weights"].write_bytes(b"fake checkpoint")
        for asset in discover_catalog(self.catalog_root).loras.values():
            if asset.family == "sdxl":
                asset.asset_paths["weights"].write_bytes(b"fake adapter")
        source = self.catalog_root / "SDXL/checkpoints/demo/model.defaults.json"
        second = json.loads(source.read_text())
        second.update(id="fixture-sdxl-second", display_name="Second SDXL checkpoint")
        directory = source.parent.parent / "second"
        directory.mkdir()
        (directory / "model.defaults.json").write_text(json.dumps(second))
        (directory / second["weights"]).write_bytes(b"second checkpoint")
        self.database, self.outputs = self.root / "jobs.sqlite3", self.root / "outputs"
        self.app = create_app({"TESTING": True, "CATALOG_ROOT": self.catalog_root,
                               "JOB_DATABASE": self.database, "JOB_OUTPUT_ROOT": self.outputs,
                               "SECRET_KEY": "execution-tests-only"})
        self.client = self.app.test_client()
        self.store = JobStore(self.database)
        self.pipelines, self.calls, self.on_step = [], [], None
        def load(source, **options):
            pipeline = Pipeline(self, source)
            self.pipelines.append(pipeline)
            return pipeline
        self.loader = SimpleNamespace(from_pretrained=Mock(side_effect=load), from_single_file=Mock(side_effect=load))
        torch = SimpleNamespace(
            __version__="test-torch", float16="fp16", bfloat16="bf16", float32="fp32",
            Generator=FakeGenerator, inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=Mock(return_value=True), is_bf16_supported=Mock(return_value=True),
                                 synchronize=Mock(), empty_cache=Mock()))
        diffusers = SimpleNamespace(
            __version__="test-diffusers", StableDiffusionXLPipeline=self.loader,
            EulerDiscreteScheduler=EulerScheduler, EulerAncestralDiscreteScheduler=AncestralScheduler,
            DPMSolverMultistepScheduler=DPMScheduler)
        runtime = patch("image_generator.workflows.sdxl.runtime.import_module", side_effect={"torch": torch, "diffusers": diffusers}.__getitem__)
        self.runtime = runtime.start()
        self.addCleanup(runtime.stop)

    def data(self, model="fixture-sdxl", **values):
        page = FormPage(self.client.get(f"/generate/{model}").get_data(as_text=True))
        data = page.data()
        for name, value in {"action": "generate", "prompt": "A mountain lake", "num_inference_steps": "4", **values}.items():
            data[name] = value
        return data

    def submit(self, model="fixture-sdxl", **values):
        response = self.client.post(f"/generate/{model}", data=self.data(model, **values))
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True)[:2500])
        return response.location.rsplit("/", 1)[1]

    def worker(self):
        return Worker(self.store, self.outputs)

    def test_registered_samplers_save_queue_and_dispatch_with_model_reuse(self):
        from image_generator.catalog.persistence import settings_revision

        manifest = self.catalog_root / "SDXL/checkpoints/demo/model.defaults.json"
        document = json.loads(manifest.read_text())
        document["requirements"]["allowed_sampling"].append({"sampler": "dpmpp_2m_sde", "noise_schedule": "default"})
        document["requirements"]["allowed_sampling"].append({"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"})
        manifest.write_text(json.dumps(document))
        html = self.client.get("/generate/fixture-sdxl").text
        for name in ("euler_ancestral", "dpmpp_2m_sde"):
            self.assertIn(f'<option value="{name}"', html)
        identifiers = []
        for sampler, schedule in (("euler_ancestral", "default"), ("dpmpp_2m_sde", "karras"),
                                  ("dpmpp_2m_sde", "default")):
            sampling = {"sampler": sampler, "noise_schedule": schedule}
            response = self.client.post("/api/models/fixture-sdxl/settings/field", json={
                "field": "sampling", "value_text": json.dumps(sampling),
                "csrf_token": self.data()["csrf_token"],
                "settings_revision": settings_revision(discover_catalog(self.catalog_root)),
            })
            self.assertEqual(response.status_code, 200, response.text)
            identifiers.append(self.submit())
            self.assertEqual(self.store.get(identifiers[-1]).request["settings"]["sampling"], sampling)
        self.runtime.assert_not_called()
        self.worker().run(until_idle=True)
        for identifier, scheduler_class, schedule in zip(
            identifiers, ("AncestralScheduler", "DPMScheduler", "DPMScheduler"), ("default", "karras", "default"), strict=True,
        ):
            job = self.store.get(identifier)
            self.assertEqual(job.status, "completed")
            metadata = json.loads(job.outputs[0].metadata_path.read_text())
            self.assertEqual(metadata["scheduler"]["class_name"], scheduler_class)
            self.assertEqual(metadata["scheduler"]["config"].get("use_karras_sigmas", False), schedule == "karras")
        self.assertEqual(len(self.pipelines), 1)
        self.assertEqual(metadata["scheduler"]["config"]["algorithm_type"], "sde-dpmsolver++")
        self.assertEqual(metadata["scheduler"]["config"]["solver_order"], 2)

    def test_generate_queues_without_loading_gpu_and_returns_resumable_job_page(self):
        identifier = self.submit()
        self.runtime.assert_not_called()
        job = self.store.get(identifier)
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.request["loras"][0]["strength"], 0.65)
        response = self.client.get(f"/jobs/{identifier}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Waiting in queue", response.get_data(as_text=True))
        self.assertIn("worker is stopped", response.get_data(as_text=True))
        self.assertIn(identifier[:8], self.client.get("/jobs").get_data(as_text=True))
        self.assertEqual(self.client.get(f"/api/jobs/{identifier}").json["status"], "queued")

    def test_denoising_progress_is_observable_before_completion_and_results_are_served(self):
        identifier = self.submit(seed=str(2**63 - 1), width="512", height="512")
        entered, release = Event(), Event()
        def step(index):
            if index == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release pipeline")
        self.on_step = step
        failures = []
        def run():
            try:
                with patch("image_generator.worker.monotonic", side_effect=lambda: next(ticks)):
                    self.worker().run(once=True)
            except BaseException as error:
                failures.append(error)
        ticks = count()
        thread = Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            response = self.client.get(f"/api/jobs/{identifier}")
            self.assertEqual(response.json["progress"]["stage"], "denoising")
            self.assertEqual((response.json["progress"]["current"], response.json["progress"]["total"]), (2, 4))
            self.assertFalse(response.json["terminal"])
            self.assertIn("no-store", response.headers["Cache-Control"])
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed")
        denoising = next(event for event in result["events"] if event["stage"] == "denoising")
        self.assertEqual((denoising["current"], denoising["total"]), (4, 4))
        saving = next(event for event in result["events"] if event["stage"] == "saving")
        self.assertEqual(saving["current"], 1)
        output = result["outputs"][0]
        self.assertEqual(output["seed"], str(2**63 - 1))
        self.assertEqual((output["width"], output["height"]), (512, 512))
        media = self.client.get(output["url"])
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.mimetype, "image/png")
        media.close()
        download = self.client.get(output["download_url"])
        self.assertIn("attachment", download.headers["Content-Disposition"])
        download.close()
        metadata = self.client.get(output["metadata_url"])
        self.assertEqual(metadata.json["seed"], 2**63 - 1)
        metadata.close()
        completed_page = self.client.get(f"/jobs/{identifier}").get_data(as_text=True)
        self.assertIn("Saved outputs", completed_page)
        self.assertIn('width="512" height="512"', completed_page)

    def test_same_checkpoint_reuses_model_reset_loras_then_switches_checkpoint(self):
        with_lora = self.submit()
        no_loras = self.data()
        no_loras.poplist("lora")
        response = self.client.post("/generate/fixture-sdxl", data=no_loras)
        self.assertEqual(response.status_code, 303)
        cleared = response.location.rsplit("/", 1)[1]
        switched = self.submit("fixture-sdxl-second")
        with self.worker() as worker:
            for identifier in (with_lora, cleared, switched):
                self.assertEqual(worker.run_once().id, identifier)
            self.assertIsNone(worker._workflow._progress_callback)
        self.assertEqual(len(self.pipelines), 2)
        self.assertIs(self.calls[0]["pipeline"], self.calls[1]["pipeline"])
        self.assertEqual(self.calls[0]["adapters"], ("adapter_0",))
        self.assertEqual(self.calls[1]["adapters"], ())
        self.pipelines[0].unload_lora_weights.assert_called_once()
        cleared_status = self.client.get(f"/api/jobs/{cleared}").json
        stages = [event["stage"] for event in cleared_status["events"]]
        self.assertIn("reusing_model", stages)
        self.assertIn("resetting_loras", stages)
        self.assertIn("switching_model", [event.stage for event in self.store.get(switched).events])
        self.assertNotIn("reusing_model", [event.stage for event in self.store.get(switched).events])
        metadata = json.loads(self.store.get(cleared).outputs[0].metadata_path.read_text())
        self.assertTrue(metadata["runtime"]["model_reused"])
        self.assertTrue(metadata["runtime"]["loras_reset"])

    def test_failed_inference_is_visible_and_next_job_loads_a_clean_pipeline(self):
        failed = self.submit(prompt="fail generation")
        following = self.submit()
        self.worker().run(until_idle=True)
        result = self.client.get(f"/api/jobs/{failed}").json
        self.assertEqual(result["status"], "failed")
        self.assertIn("inference failed", result["error"])
        self.assertEqual(result["outputs"], [])
        failed_page = self.client.get(f"/jobs/{failed}").get_data(as_text=True)
        self.assertIn("Generation could not complete", failed_page)
        self.assertIn('max="1" value="0"', failed_page)
        self.assertEqual(self.store.get(following).status, "completed")
        self.assertEqual(len(self.pipelines), 2)

    def test_duplicate_submission_is_idempotent_and_changed_values_conflict(self):
        data = self.data()
        first = self.client.post("/generate/fixture-sdxl", data=data)
        second = self.client.post("/generate/fixture-sdxl", data=data)
        self.assertEqual(first.location, second.location)
        self.assertEqual(len(self.store.list_jobs()), 1)
        data["seed"] = "99"
        response = self.client.post("/generate/fixture-sdxl", data=data)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(self.store.list_jobs()), 1)

    def test_invalid_or_cross_session_submissions_create_no_jobs(self):
        for name, value in (("csrf_token", "invalid"), ("submission_key", ""), ("prompt", ""), ("width", "770")):
            data = self.data()
            data[name] = value
            with self.subTest(name=name):
                self.assertIn(self.client.post("/generate/fixture-sdxl", data=data).status_code, (400, 422))
        other = self.app.test_client()
        self.assertEqual(other.post("/generate/fixture-sdxl", data=self.data()).status_code, 400)
        self.assertEqual(self.store.list_jobs(), ())

    def test_missing_weights_rejected_for_generation_but_preview_still_works(self):
        (self.catalog_root / "SDXL/checkpoints/demo/missing-fixture.safetensors").unlink()
        data = self.data()
        response = self.client.post("/generate/fixture-sdxl", data=data)
        self.assertEqual(response.status_code, 422)
        self.assertIn("weights not found", response.get_data(as_text=True))
        data["action"] = "preview"
        self.assertEqual(self.client.post("/generate/fixture-sdxl", data=data).status_code, 200)
        self.assertEqual(self.store.list_jobs(), ())

    def test_queue_storage_errors_preserve_form_values_and_report_unavailability(self):
        data = self.data()
        with patch("image_generator.jobs.JobStore.enqueue", side_effect=sqlite3.OperationalError("database busy")):
            response = self.client.post("/generate/fixture-sdxl", data=data)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(FormPage(response.get_data(as_text=True)).controls["prompt"][0]["value"], "A mountain lake")
        with patch("image_generator.jobs.JobStore.list_jobs", side_effect=sqlite3.OperationalError("database busy")):
            self.assertEqual(self.client.get("/jobs").status_code, 503)

    def test_flux_exposes_the_same_image_generation_action(self):
        response = self.client.get("/generate/fixture-flux")
        self.assertIn('value="generate"', response.get_data(as_text=True))
        self.assertIn("Generate images", response.get_data(as_text=True))
        data = FormPage(response.get_data(as_text=True)).data()
        data["action"] = "generate"
        # Missing fixture assets must fail validation before any GPU work.
        data["prompt"] = "A mountain lake"
        self.assertEqual(self.client.post("/generate/fixture-flux", data=data).status_code, 422)
        self.assertEqual(self.store.list_jobs(), ())

    def test_results_do_not_depend_on_live_catalog_and_output_paths_are_confined(self):
        identifier = self.submit()
        self.worker().run(once=True)
        output = self.client.get(f"/api/jobs/{identifier}").json["outputs"][0]
        shutil.rmtree(self.catalog_root)
        self.assertEqual(self.client.get(f"/jobs/{identifier}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/jobs/{identifier}").status_code, 200)
        path = self.store.get(identifier).outputs[0].file_path
        path.unlink()
        outside = self.root / "outside.png"
        Image.new("RGB", (2, 2)).save(outside)
        path.symlink_to(outside)
        self.assertEqual(self.client.get(output["url"]).status_code, 404)
        self.assertEqual(self.client.get(f"/jobs/{identifier}/outputs/unknown/media").status_code, 404)
        self.assertEqual(self.client.get("/jobs/unknown").status_code, 404)

    def test_job_output_identifiers_cannot_cross_jobs(self):
        first, second = self.submit(), self.submit()
        self.worker().run(until_idle=True)
        other_output = self.store.get(second).outputs[0]
        response = self.client.get(f"/jobs/{first}/outputs/{other_output.id}/media")
        self.assertEqual(response.status_code, 404)

    def test_version_one_database_migrates_without_losing_jobs(self):
        queued = self.submit()
        with sqlite3.connect(self.database) as connection:
            connection.execute("DROP TABLE job_events")
            connection.execute("DROP TABLE job_submissions")
            connection.execute("PRAGMA user_version=1")
        upgraded = JobStore(self.database)
        self.assertEqual(upgraded.get(queued).status, "queued")
        self.worker().run(once=True)
        self.assertEqual(upgraded.get(queued).status, "completed")

    def test_model_file_atomic_replacement_invalidates_cached_pipeline(self):
        first = self.submit()
        with self.worker() as worker:
            self.assertEqual(worker.run_once().id, first)
            path = self.catalog_root / "SDXL/checkpoints/demo/missing-fixture.safetensors"
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            second = self.submit()
            self.assertEqual(worker.run_once().id, second)
        self.assertEqual(len(self.pipelines), 2)
        self.assertFalse(json.loads(self.store.get(second).outputs[0].metadata_path.read_text())["runtime"]["model_reused"])


if __name__ == "__main__":
    unittest.main()

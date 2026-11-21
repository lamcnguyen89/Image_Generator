"""Krea2 forms through the real queue/workflow, with a synthetic model runtime."""

from contextlib import nullcontext
from io import BytesIO
from itertools import count
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
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
from image_generator.jobs import JobStore
from image_generator.worker import Worker
from test_generation_ui import FormPage


class Scheduler:
    def __init__(self, config=None):
        self.config = dict(config or {"num_train_timesteps": 1000})

    @classmethod
    def from_config(cls, config, **options):
        return cls({**config, **options})


class Generator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class Pipeline:
    def __init__(self, owner):
        self.owner, self.adapters = owner, []
        self.scheduler = Scheduler()
        self.to = Mock(return_value=self)
        self.load_lora_weights = Mock(side_effect=lambda path, **options: self.adapters.append(options["adapter_name"]))
        self.unload_lora_weights = Mock(side_effect=lambda **options: self.adapters.clear())
        self.set_adapters = Mock()
        self.set_lora_device = Mock()

    def __call__(self, **arguments):
        self.owner.calls.append({"pipeline": self, "adapters": tuple(self.adapters), "arguments": arguments})
        if arguments["prompt"] == "fail generation":
            raise RuntimeError("synthetic Krea2 inference failure")
        self.num_timesteps = arguments["num_inference_steps"]
        for step in range(self.num_timesteps):
            tensors = {}
            callback = arguments["callback_on_step_end"]
            self.owner.assertIs(callback(self, step, step, tensors), tensors)
            if self.owner.on_step:
                self.owner.on_step(step)
        return SimpleNamespace(images=[Image.new("RGB", (arguments["width"], arguments["height"]), "navy")
                                       for _ in range(arguments["num_images_per_prompt"])])


class Krea2ExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog_root = self.root / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.catalog_root)
        catalog = discover_catalog(self.catalog_root)
        for asset in (*catalog.models.values(), *catalog.loras.values()):
            if asset.family == "krea2":
                asset.asset_paths["weights"].write_bytes(b"synthetic weights")
        self.outputs, self.videos = self.root / "Images", self.root / "Videos"
        self.app = create_app({"TESTING": True, "CATALOG_ROOT": self.catalog_root,
                               "JOB_DATABASE": self.root / "jobs.sqlite3", "JOB_OUTPUT_ROOT": self.outputs,
                               "JOB_VIDEO_OUTPUT_ROOT": self.videos, "SECRET_KEY": "krea2-tests"})
        self.client = self.app.test_client()
        self.store = JobStore(self.app.config["JOB_DATABASE"])
        self.pipelines, self.calls, self.on_step = [], [], None

        def load(*args, **options):
            pipeline = Pipeline(self)
            self.pipelines.append(pipeline)
            return pipeline

        self.loader, self.transformer_loader = Mock(side_effect=load), Mock(return_value=Mock(to=Mock(return_value=object())))
        torch = SimpleNamespace(
            __version__="test-torch", float16="fp16", bfloat16="bf16", float32="fp32",
            Generator=Generator, inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                                 device=lambda device: nullcontext(), synchronize=Mock(), empty_cache=Mock()))
        diffusers = SimpleNamespace(
            __version__="test-diffusers", FlowMatchEulerDiscreteScheduler=Scheduler,
            Krea2Pipeline=SimpleNamespace(from_pretrained=self.loader),
            Krea2Transformer2DModel=SimpleNamespace(from_single_file=self.transformer_loader))
        runtime = patch("image_generator.workflows.shared.diffusers_runtime.import_module",
                        side_effect={"torch": torch, "diffusers": diffusers}.__getitem__)
        self.runtime = runtime.start()
        self.addCleanup(runtime.stop)

    def data(self, model="fixture-krea2", *, loras=(), **values):
        page = FormPage(self.client.get(f"/generate/{model}").get_data(as_text=True))
        data = page.data()
        for name, value in {"action": "generate", "prompt": "A mountain lake", "width": "64", "height": "32",
                            "num_inference_steps": "4", "seed": "42", **values}.items():
            data[name] = value
        data.setlist("lora", loras)
        return data

    def submit(self, model="fixture-krea2", **values):
        response = self.client.post(f"/generate/{model}", data=self.data(model, **values))
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True))
        return response.location.rsplit("/", 1)[1]

    def worker(self):
        return Worker(self.store, self.outputs, video_output_dir=self.videos)

    def test_form_uses_shared_sdxl_layout_and_krea2_controls(self):
        html = self.client.get("/generate/fixture-krea2").get_data(as_text=True)
        sdxl = self.client.get("/generate/fixture-sdxl").get_data(as_text=True)
        for text in ("generation-layout", "generation.css", "generation.js", "Generate images",
                     "Preview request", "Save model settings", 'name="csrf_token"', 'name="submission_key"'):
            self.assertIn(text, html)
            self.assertIn(text, sdxl)
        page = FormPage(html)
        self.assertNotIn("true_cfg_scale", page.controls)
        self.assertEqual(page.controls["max_sequence_length"][0]["max"], "256")
        self.assertEqual(page.controls["guidance_scale"][0]["value"], "3.5")
        self.assertNotIn("clip_skip", page.controls)
        self.assertNotIn("input_image", page.controls)
        self.assertIn('id="krea2-cfg-status"', html)
        self.runtime.assert_not_called()

    def test_submission_freezes_values_and_loras_without_loading_runtime_or_saving_settings(self):
        before = {path: path.read_bytes() for path in self.catalog_root.rglob("*.json")}
        identifier = self.submit(loras=("fixture-krea2-style", "fixture-krea2-detail"),
                                 **{"strength:fixture-krea2-style": "0.35", "include_trigger_words": "on"})
        job = self.store.get(identifier)
        self.assertEqual((job.workflow, job.status), ("krea2", "queued"))
        self.assertEqual([item["strength"] for item in job.request["loras"]], [0.35, 0.8])
        self.assertIn("krea2_style", job.request["prompt"])
        self.assertEqual(job.request["settings"]["num_inference_steps"], 4)
        self.assertEqual(before, {path: path.read_bytes() for path in self.catalog_root.rglob("*.json")})
        self.runtime.assert_not_called()
        self.assertIn("Waiting in queue", self.client.get(f"/jobs/{identifier}").get_data(as_text=True))

    def test_images_metadata_downloads_and_large_seed_sequence(self):
        identifier = self.submit(seed=str(2**63 - 1), num_images_per_prompt="2", negative_prompt="blurry", guidance_scale="3.5")
        self.worker().run(once=True)
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed")
        self.assertEqual([item["seed"] for item in result["outputs"]], [str(2**63 - 1), "0"])
        self.assertFalse(self.videos.exists())
        for output, record in zip(result["outputs"], self.store.get(identifier).outputs, strict=True):
            self.assertEqual(record.file_path.parent, self.outputs / identifier)
            self.assertTrue(record.file_path.name.startswith("krea2-seed-"))
            self.assertEqual((output["kind"], output["width"], output["height"]), ("image", 64, 32))
            media = self.client.get(output["url"])
            self.assertEqual((media.status_code, media.mimetype), (200, "image/png"))
            with Image.open(BytesIO(media.data)) as image:
                self.assertEqual(image.size, (64, 32))
            media.close()
            download = self.client.get(output["download_url"])
            self.assertIn("attachment", download.headers["Content-Disposition"])
            download.close()
            metadata = self.client.get(output["metadata_url"])
            self.assertEqual(metadata.json["seed"], record.seed)
            self.assertTrue(metadata.json["negative_prompt_enabled"])
            self.assertEqual(metadata.json["runtime"]["dtype"], "bfloat16")
            self.assertNotIn("callback_on_step_end", metadata.get_data(as_text=True))
            metadata.close()
        arguments = self.calls[0]["arguments"]
        self.assertEqual(arguments["guidance_scale"], 3.5)
        self.assertNotIn("true_cfg_scale", arguments)
        self.assertEqual(arguments["callback_on_step_end_tensor_inputs"], [])
        html = self.client.get(f"/jobs/{identifier}").get_data(as_text=True)
        self.assertIn("Saved outputs", html)
        self.assertIn('width="64" height="32"', html)

    def test_progress_is_visible_while_denoising_and_finishes_with_decoding_and_saving(self):
        identifier = self.submit()
        entered, release = Event(), Event()

        def on_step(step):
            if step == 1:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test did not release inference")

        self.on_step = on_step
        failures, ticks = [], count()

        def run():
            try:
                with patch("image_generator.worker.monotonic", side_effect=lambda: next(ticks)):
                    self.worker().run(once=True)
            except BaseException as error:
                failures.append(error)

        thread = Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(10))
            response = self.client.get(f"/api/jobs/{identifier}")
            progress = response.json["progress"]
            self.assertEqual((progress["stage"], progress["current"], progress["total"]), ("denoising", 2, 4))
            self.assertFalse(response.json["terminal"])
            self.assertIn("no-store", response.headers["Cache-Control"])
        finally:
            release.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        job = self.store.get(identifier)
        self.assertEqual(job.status, "completed")
        stages = [event.stage for event in job.events]
        for stage in ("loading_runtime", "loading_model", "loras_ready", "decoding", "saving"):
            self.assertIn(stage, stages)
        denoising = next(event for event in job.events if event.stage == "denoising")
        self.assertEqual((denoising.current, denoising.total), (4, 4))

    def test_adapter_reset_pipeline_reuse_and_checkpoint_switch(self):
        first = self.submit(loras=("fixture-krea2-style",))
        cleared = self.submit()
        manifest = self.catalog_root / "Krea2/checkpoints/demo/model.defaults.json"
        second = json.loads(manifest.read_text())
        second.update(id="fixture-krea2-second", display_name="Second Krea2 checkpoint")
        directory = manifest.parent.parent / "second"
        directory.mkdir()
        (directory / "model.defaults.json").write_text(json.dumps(second))
        (directory / second["weights"]).write_bytes(b"second checkpoint")
        switched = self.submit("fixture-krea2-second")
        self.worker().run(until_idle=True)
        for identifier in (first, cleared, switched):
            self.assertEqual(self.store.get(identifier).status, "completed")
        self.assertEqual(len(self.pipelines), 2)
        self.assertIs(self.calls[0]["pipeline"], self.calls[1]["pipeline"])
        self.assertEqual([call["adapters"] for call in self.calls], [("adapter_0",), (), ()])
        self.pipelines[0].unload_lora_weights.assert_called_once_with()
        self.assertIn("reusing_model", [event.stage for event in self.store.get(cleared).events])
        self.assertIn("switching_model", [event.stage for event in self.store.get(switched).events])

    def test_inference_failure_is_visible_and_next_job_loads_clean_pipeline(self):
        failed, following = self.submit(prompt="fail generation"), self.submit()
        self.worker().run(until_idle=True)
        result = self.client.get(f"/api/jobs/{failed}").json
        self.assertEqual(result["status"], "failed")
        self.assertIn("synthetic Krea2 inference failure", result["error"])
        self.assertEqual(result["outputs"], [])
        self.assertIn("Generation could not complete", self.client.get(f"/jobs/{failed}").get_data(as_text=True))
        self.assertEqual(self.store.get(following).status, "completed")
        self.assertEqual(len(self.pipelines), 2)

    def test_retries_are_idempotent_and_changed_values_conflict(self):
        data = self.data()
        first = self.client.post("/generate/fixture-krea2", data=data)
        self.assertEqual(first.status_code, 303)
        self.assertEqual(self.client.post("/generate/fixture-krea2", data=data).location, first.location)
        data["seed"] = "99"
        self.assertEqual(self.client.post("/generate/fixture-krea2", data=data).status_code, 409)
        self.assertEqual(len(self.store.list_jobs()), 1)
        self.runtime.assert_not_called()

    def test_invalid_fields_adapters_and_cross_session_submissions_never_queue(self):
        for name, value, status in (("prompt", "", 422), ("width", "65", 422), ("guidance_scale", "-0.5", 422),
                                    ("max_sequence_length", "257", 422), ("sampling.sampler", "euler", 422),
                                    ("csrf_token", "bad", 400), ("submission_key", "../escape", 400)):
            with self.subTest(name=name):
                data = self.data(**{name: value})
                self.assertEqual(self.client.post("/generate/fixture-krea2", data=data).status_code, status)
        for selected in (("fixture-sdxl-style",), ("fixture-krea2-style", "fixture-krea2-style")):
            self.assertEqual(self.client.post("/generate/fixture-krea2", data=self.data(loras=selected)).status_code, 422)
        self.assertEqual(self.app.test_client().post("/generate/fixture-krea2", data=self.data()).status_code, 400)
        self.assertEqual(self.store.list_jobs(), ())
        self.runtime.assert_not_called()

    def test_missing_weights_fail_generation_but_preview_remains_available(self):
        for relative, loras in (("checkpoints/demo/missing-fixture.safetensors", ()),
                                ("loras/style/missing-fixture.safetensors", ("fixture-krea2-style",))):
            path = self.catalog_root / "Krea2" / relative
            payload = path.read_bytes()
            path.unlink()
            data = self.data(loras=loras)
            response = self.client.post("/generate/fixture-krea2", data=data)
            self.assertEqual(response.status_code, 422)
            self.assertIn("weights not found", response.get_data(as_text=True))
            data["action"] = "preview"
            self.assertEqual(self.client.post("/generate/fixture-krea2", data=data).status_code, 200)
            path.write_bytes(payload)
        self.assertEqual(self.store.list_jobs(), ())
        self.runtime.assert_not_called()

    def test_queue_failure_preserves_form_values(self):
        with patch("image_generator.jobs.JobStore.enqueue", side_effect=sqlite3.OperationalError("busy")):
            response = self.client.post("/generate/fixture-krea2", data=self.data(guidance_scale="2.5"))
        self.assertEqual(response.status_code, 503)
        page = FormPage(response.get_data(as_text=True))
        self.assertEqual(page.controls["guidance_scale"][0]["value"], "2.5")
        self.assertEqual(page.controls["prompt"][0]["value"], "A mountain lake")
        self.assertEqual(self.store.list_jobs(), ())
        self.runtime.assert_not_called()

    def test_registered_outputs_are_confined_to_the_job_and_survive_catalog_removal(self):
        first, second = self.submit(), self.submit()
        self.worker().run(until_idle=True)
        record = self.store.get(first).outputs[0]
        other = self.store.get(second).outputs[0]
        self.assertEqual(self.client.get(f"/jobs/{first}/outputs/{other.id}/media").status_code, 404)
        shutil.rmtree(self.catalog_root)
        self.assertEqual(self.client.get(f"/jobs/{first}").status_code, 200)
        outside = self.root / "outside.png"
        Image.new("RGB", (2, 2)).save(outside)
        for part, path in (("media", record.file_path), ("metadata", record.metadata_path)):
            path.unlink()
            path.symlink_to(outside)
            self.assertEqual(self.client.get(f"/jobs/{first}/outputs/{record.id}/{part}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/jobs/{first}").status_code, 200)


    def prepared(self, **overrides):
        from image_generator.generation import prepare_generation

        return prepare_generation(discover_catalog(self.catalog_root), "fixture-krea2",
                                  overrides={"prompt": "A lake", "width": 64, "height": 32,
                                             "num_inference_steps": 2, **overrides})

    def workflow(self, **options):
        from image_generator.workflows.krea2 import Krea2Workflow

        workflow = Krea2Workflow(**options)
        workflow.set_progress_callback(lambda event: None)
        self.addCleanup(workflow.close)
        return workflow

    def test_raw_loader_uses_local_transformer_and_runtime_options(self):
        workflow = self.workflow(cache_dir=self.root / "cache", token="test-secret", local_files_only=True)
        batch = workflow.generate(self.prepared())
        options = self.transformer_loader.call_args.kwargs
        self.assertEqual(options["config"], "krea/Krea-2-Raw")
        self.assertEqual(options["subfolder"], "transformer")
        self.assertEqual(options["dtype"], "bf16")
        self.assertEqual(options["token"], "test-secret")
        self.assertTrue(options["local_files_only"])
        self.assertEqual(options["cache_dir"], str(self.root / "cache"))
        self.assertIs(self.loader.call_args.kwargs["transformer"], self.transformer_loader.return_value.to.return_value)
        self.assertIs(self.loader.call_args.kwargs["is_distilled"], False)
        self.assertEqual(self.loader.call_args.kwargs["device_map"], "cuda")
        self.assertNotIn("test-secret", json.dumps(batch.metadata))

    def test_cfg_zero_and_empty_negative_prompt(self):
        workflow = self.workflow()
        for guidance, negative, cfg, active in ((0, "blur", False, False), (3.5, "", True, False),
                                                (0.5, "blur", True, True)):
            with self.subTest(guidance=guidance, negative=negative):
                batch = workflow.generate(self.prepared(guidance_scale=guidance, negative_prompt=negative))
                self.assertIs(batch.metadata["cfg_enabled"], cfg)
                self.assertIs(batch.metadata["negative_prompt_enabled"], active)
                self.assertNotIn("true_cfg_scale", self.calls[-1]["arguments"])

    def test_backend_rejects_flux_settings_and_invalid_shapes_before_runtime(self):
        for changes in ({"true_cfg_scale": 2}, {"max_sequence_length": 257}, {"guidance_scale": -1},
                        {"max_sequence_length": True}, {"width": 65}, {"seed": 2**63},
                        {"sampling": {"sampler": "euler"}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepared(**changes)
        model = discover_catalog(self.catalog_root).get_model("fixture-krea2")
        for key, value in (("model_variant", "krea_2_turbo"), ("weights_component", "pipeline"),
                           ("base_model_id", None)):
            original = model.defaults[key]
            model.defaults[key] = value
            from dataclasses import replace
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.workflow().validate(replace(self.prepared(), model=model))
            model.defaults[key] = original
        self.runtime.assert_not_called()

    def test_pretrained_loading_and_checkpoint_replacement(self):
        workflow = self.workflow()
        request = self.prepared()
        workflow.generate(request)
        request.model.asset_paths["weights"].write_bytes(b"replaced checkpoint")
        workflow.generate(request)
        self.assertEqual(self.loader.call_count, 2)
        request.model.defaults["weights"] = None
        workflow.generate(request)
        self.assertEqual(self.transformer_loader.call_count, 2)
        self.assertNotIn("transformer", self.loader.call_args.kwargs)
        self.assertIs(self.loader.call_args.kwargs["is_distilled"], False)

    def test_multiple_adapters_are_weighted_placed_and_cleared_after_failure(self):
        workflow = self.workflow()
        request = self.prepared(loras=[{"id": "fixture-krea2-style", "enabled": True, "strength": 0.3},
                                       {"id": "fixture-krea2-detail", "enabled": True, "strength": 1.2}])
        workflow.generate(request)
        pipe = self.pipelines[0]
        pipe.set_adapters.assert_called_once_with(["adapter_0", "adapter_1"], adapter_weights=[0.3, 1.2])
        pipe.set_lora_device.assert_called_once_with(adapter_names=["adapter_0", "adapter_1"], device="cuda:0")
        pipe.load_lora_weights.side_effect = RuntimeError("bad tensors")
        with self.assertRaisesRegex(RuntimeError, "Could not load LoRA"):
            workflow.generate(request)
        self.assertIsNone(workflow._pipeline)
        workflow.generate(self.prepared(loras=[]))
        self.assertEqual(len(self.pipelines), 2)

    def test_cli_dry_run_works_without_model_libraries(self):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "scripts/generate_krea2.py"), "--root", str(self.catalog_root),
             "--model-id", "fixture-krea2", "--prompt", "A lake", "--seed", "random", "--dry-run"],
            cwd="/tmp", capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["model"]["workflow"], "krea2")
        self.assertIsNone(data["settings"]["seed"])
        self.assertNotIn("true_cfg_scale", data["settings"])
        self.runtime.assert_not_called()


    def test_cpu_loading_uses_float32_without_cuda_mapping(self):
        batch = self.workflow(device="cpu").generate(self.prepared())
        self.assertEqual(self.loader.call_args.kwargs["dtype"], "fp32")
        self.assertNotIn("device_map", self.loader.call_args.kwargs)
        self.assertEqual(batch.metadata["runtime"]["device"], "cpu")

    def test_save_and_reset_model_settings_preserve_disabled_adapters(self):
        data = self.data(guidance_scale="0", max_sequence_length="128")
        data["action"] = "save_model"
        data.setlist("configured_lora", ["fixture-krea2-style"])
        data["strength_source:fixture-krea2-style"] = "override"
        data["strength:fixture-krea2-style"] = "0.4"
        response = self.client.post("/settings/models/fixture-krea2", data=data)
        self.assertEqual(response.status_code, 303)
        saved = discover_catalog(self.catalog_root).get_model("fixture-krea2").user_overrides["settings"]
        self.assertEqual(saved["guidance_scale"], 0)
        self.assertEqual(saved["max_sequence_length"], 128)
        self.assertEqual(saved["loras"], [{"id": "fixture-krea2-style", "enabled": False, "strength": 0.4}])
        data = self.data(action="reset_model")
        self.assertEqual(self.client.post("/settings/models/fixture-krea2", data=data).status_code, 303)
        self.assertEqual(discover_catalog(self.catalog_root).get_model("fixture-krea2").user_overrides["settings"], {})
        self.runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()

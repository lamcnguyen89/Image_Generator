"""Qwen 2.1 forms, settings, queue and workflow exercised without model weights."""

from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
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
from image_generator.generation import prepare_generation
from image_generator.jobs import JobStore
from image_generator.worker import Worker
from image_generator.workflows.qwen21 import Qwen21Workflow, split_fused_lora
from test_generation_ui import FormPage


class Scheduler:
    def __init__(self, config=None):
        self.config = dict(config or {"num_train_timesteps": 1000})

    @classmethod
    def from_config(cls, config, **options):
        return cls({**config, **options})


class Pipeline:
    def __init__(self, owner):
        self.owner, self.adapters = owner, []
        self.scheduler, self.config = Scheduler(), {"sample_sigmas": owner.repo_sigmas}
        self.to = Mock(return_value=self)
        self.transformer = SimpleNamespace(get_submodule=Mock(return_value=object()))
        self.lora_state_dict = Mock(return_value=({
            "transformer.img_in.lora_A.weight": object(), "transformer.img_in.lora_B.weight": object(),
        }, None))
        self.load_lora_into_transformer = Mock(side_effect=lambda state, **options: self.adapters.append(options["adapter_name"]))
        self.unload_lora_weights = Mock(side_effect=self.adapters.clear)
        self.set_adapters, self.set_lora_device = Mock(), Mock()

    def __call__(self, **arguments):
        self.owner.calls.append({"pipeline": self, "adapters": tuple(self.adapters), "arguments": arguments})
        if arguments["prompt"] == "fail generation":
            raise RuntimeError("synthetic Qwen inference failure")
        sigmas = arguments["sigmas"] if arguments["sigmas"] is not None else self.config["sample_sigmas"]
        self.num_timesteps = len(sigmas) if sigmas is not None else arguments["num_inference_steps"]
        for step in range(self.num_timesteps):
            callback = arguments.get("callback_on_step_end")
            if callback:
                tensors = {}
                self.owner.assertIs(callback(self, step, step, tensors), tensors)
            if self.owner.on_step:
                self.owner.on_step(step)
        return SimpleNamespace(images=[Image.new("RGBA", (arguments["width"], arguments["height"]), (20, 40, 60, 127))
                                       for _ in range(arguments["num_images_per_prompt"])])


class Qwen21ExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog_root = self.root / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.catalog_root)
        for asset in self.catalog().models.values():
            if asset.family == "qwen_2_1":
                asset.asset_paths["weights"].write_bytes(b"synthetic transformer")
        for asset in self.catalog().loras.values():
            if asset.family == "qwen_2_1":
                asset.asset_paths["weights"].write_bytes(b"synthetic adapter")
        self.outputs, self.videos = self.root / "Images", self.root / "Videos"
        self.app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": self.catalog_root,
                               "JOB_DATABASE": self.root / "jobs.sqlite3", "JOB_OUTPUT_ROOT": self.outputs,
                               "JOB_VIDEO_OUTPUT_ROOT": self.videos, "SECRET_KEY": "qwen-tests"})
        self.client = self.app.test_client()
        self.store = JobStore(self.app.config["JOB_DATABASE"])
        self.pipelines, self.calls, self.on_step, self.repo_sigmas = [], [], None, None

        def load(*args, **options):
            pipeline = Pipeline(self)
            self.pipelines.append(pipeline)
            return pipeline

        self.loader = Mock(side_effect=load)
        self.transformer_loader = Mock(return_value=Mock(to=Mock(return_value=object())))
        torch = SimpleNamespace(
            __version__="test-torch", float16="fp16", bfloat16="bf16", float32="fp32",
            Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed), inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                                 device=lambda device: nullcontext(), synchronize=Mock(), empty_cache=Mock()))
        diffusers = SimpleNamespace(
            __version__="test-diffusers", FlowMatchEulerDiscreteScheduler=Scheduler,
            QwenImage21Pipeline=SimpleNamespace(from_pretrained=self.loader),
            QwenImage21Transformer2DModel=SimpleNamespace(from_single_file=self.transformer_loader))
        runtime = patch("image_generator.workflows.common.import_module",
                        side_effect={"torch": torch, "diffusers": diffusers}.__getitem__)
        self.runtime = runtime.start()
        self.addCleanup(runtime.stop)

    def catalog(self):
        return discover_catalog(self.catalog_root, initialize_missing=False)

    def data(self, model="fixture-qwen_2_1", *, loras=(), **values):
        response = self.client.get(f"/generate/{model}")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        data = FormPage(response.get_data(as_text=True)).data()
        for name, value in {"action": "generate", "prompt": "A mountain lake", "width": "64", "height": "32",
                            "num_inference_steps": "4", "seed": "42", **values}.items():
            data[name] = value
        data.setlist("lora", loras)
        return data

    def submit(self, model="fixture-qwen_2_1", **values):
        response = self.client.post(f"/generate/{model}", data=self.data(model, **values))
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True))
        return response.location.rsplit("/", 1)[1]

    def prepared(self, **overrides):
        return prepare_generation(self.catalog(), "fixture-qwen_2_1",
                                  overrides={"prompt": "A lake", "width": 64, "height": 32,
                                             "num_inference_steps": 4, **overrides})

    def workflow(self, **options):
        workflow = Qwen21Workflow(**options)
        self.addCleanup(workflow.close)
        return workflow

    def worker(self):
        return Worker(self.store, self.outputs, video_output_dir=self.videos)

    def test_card_order_and_shared_form_with_qwen_controls(self):
        html = self.client.get("/").get_data(as_text=True)
        headings = re.findall(r'<h2 id="family-\d+">(.*?)</h2>', html)
        self.assertEqual(headings[:3], ["FLUX", "QWEN 2.1", "KREA2"])
        html = self.client.get("/generate/fixture-qwen_2_1").get_data(as_text=True)
        sdxl = self.client.get("/generate/fixture-sdxl").get_data(as_text=True)
        for text in ("generation-layout", "generation.css", "generation.js", "Generate images",
                     "Preview request", "Save model settings", 'name="csrf_token"', 'name="submission_key"'):
            self.assertIn(text, html)
            self.assertIn(text, sdxl)
        page = FormPage(html)
        self.assertEqual(page.controls["width"][0]["step"], "32")
        self.assertEqual(page.controls["true_cfg_scale"][0]["value"], "1.0")
        self.assertIn("checked", page.controls["use_kv_cache"][0])
        self.assertIn("sigmas", page.controls)
        self.assertIn('id="cfg-status"', html)
        for name in ("guidance_scale", "max_sequence_length", "clip_skip", "input_image"):
            self.assertNotIn(name, page.controls)
        self.assertNotIn("min", page.controls["strength:fixture-qwen_2_1-style"][0])
        self.assertEqual(FormPage(sdxl).controls["strength:fixture-sdxl-style"][0]["min"], "0")
        self.runtime.assert_not_called()

    def test_preview_parses_sigma_list_without_touching_weights_or_runtime(self):
        self.catalog().get_model("fixture-qwen_2_1").asset_paths["weights"].unlink()
        before = {path: path.read_bytes() for path in self.catalog_root.rglob("*.json")}
        response = self.client.post("/generate/fixture-qwen_2_1", data=self.data(action="preview", sigmas="1, 0.8\n0.4"))
        self.assertEqual(response.status_code, 200)
        preview = FormPage(response.get_data(as_text=True)).preview
        self.assertEqual(preview["overrides"]["sigmas"], [1.0, 0.8, 0.4])
        self.assertEqual(before, {path: path.read_bytes() for path in self.catalog_root.rglob("*.json")})
        self.assertFalse(self.outputs.exists())
        self.runtime.assert_not_called()

    def test_submission_freezes_settings_and_multiple_signed_loras_without_loading_runtime(self):
        before = {path: path.read_bytes() for path in self.catalog_root.rglob("*.json")}
        identifier = self.submit(loras=("fixture-qwen_2_1-style", "fixture-qwen_2_1-detail"),
                                 **{"strength:fixture-qwen_2_1-style": "-0.3", "include_trigger_words": "on",
                                    "sigmas": "1,0.8,0.4"})
        job = self.store.get(identifier)
        self.assertEqual((job.workflow, job.status), ("qwen_2_1", "queued"))
        self.assertEqual([item["strength"] for item in job.request["loras"]], [-0.3, 0.8])
        self.assertEqual(job.request["settings"]["sigmas"], [1, 0.8, 0.4])
        self.assertIn("qwen21_style", job.request["prompt"])
        self.assertEqual(before, {path: path.read_bytes() for path in self.catalog_root.rglob("*.json")})
        self.runtime.assert_not_called()
        shutil.rmtree(self.catalog_root)
        # Model files are fingerprinted, so keep those files for execution in
        # the other tests; this request remains inspectable without manifests.
        self.assertEqual(self.client.get(f"/jobs/{identifier}").status_code, 200)

    def test_rgba_images_metadata_downloads_and_large_seed_sequence(self):
        identifier = self.submit(seed=str(2**63 - 1), num_images_per_prompt="2",
                                 negative_prompt="blurry", true_cfg_scale="2.5", sigmas="1,0.8,0.4")
        self.worker().run(once=True)
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([item["seed"] for item in result["outputs"]], [str(2**63 - 1), "0"])
        for output in result["outputs"]:
            media = self.client.get(output["url"])
            with Image.open(BytesIO(media.data)) as image:
                self.assertEqual((image.size, image.mode), ((64, 32), "RGBA"))
                self.assertEqual(image.getpixel((0, 0))[3], 127)
            media.close()
            metadata = self.client.get(output["metadata_url"])
            self.assertTrue(metadata.json["negative_prompt_enabled"])
            self.assertEqual(metadata.json["effective_num_inference_steps"], 3)
            self.assertEqual(metadata.json["effective_sample_sigmas"], [1, 0.8, 0.4])
            self.assertEqual(metadata.json["runtime"]["dtype"], "bfloat16")
            self.assertNotIn("callback_on_step_end", metadata.get_data(as_text=True))
            metadata.close()
            download = self.client.get(output["download_url"])
            self.assertIn("attachment", download.headers["Content-Disposition"])
            download.close()
        self.assertFalse(self.videos.exists())
        arguments = self.calls[0]["arguments"]
        self.assertNotIn("guidance_scale", arguments)
        self.assertNotIn("max_sequence_length", arguments)
        self.assertTrue(arguments["use_kv_cache"])
        self.assertEqual(arguments["callback_on_step_end_tensor_inputs"], [])

    def test_real_queue_progress_uses_sigma_count_and_finishes_with_saving(self):
        identifier = self.submit(sigmas="1,0.8,0.4")
        entered, release = Event(), Event()

        def on_step(step):
            if step == 0:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test did not release inference")

        self.on_step = on_step
        failures = []

        def run():
            try:
                self.worker().run(once=True)
            except BaseException as error:
                failures.append(error)

        thread = Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(10))
            status = self.client.get(f"/api/jobs/{identifier}").json
            self.assertEqual(status["status"], "running")
            self.assertEqual(status["progress"]["total"], 3)
        finally:
            release.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        job = self.store.get(identifier)
        self.assertEqual(job.status, "completed")
        stages = [event.stage for event in job.events]
        for stage in ("loading_runtime", "loading_model", "loras_ready", "denoising", "decoding", "saving"):
            self.assertIn(stage, stages)

    def test_adapter_reset_pipeline_reuse_and_inference_failure_recovery(self):
        first = self.submit(loras=("fixture-qwen_2_1-style",))
        cleared = self.submit()
        failed, following = self.submit(prompt="fail generation"), self.submit()
        self.worker().run(until_idle=True)
        for identifier in (first, cleared, following):
            self.assertEqual(self.store.get(identifier).status, "completed")
        self.assertEqual(self.store.get(failed).status, "failed")
        self.assertIn("synthetic Qwen inference failure", self.store.get(failed).error)
        self.assertEqual(len(self.pipelines), 2)
        self.assertEqual([call["adapters"] for call in self.calls], [("adapter_0",), (), (), ()])
        self.pipelines[0].unload_lora_weights.assert_called_once_with()
        self.assertIn("reusing_model", [event.stage for event in self.store.get(cleared).events])

    def test_queue_idempotence_csrf_and_invalid_values(self):
        data = self.data()
        response = self.client.post("/generate/fixture-qwen_2_1", data=data)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.client.post("/generate/fixture-qwen_2_1", data=data).location, response.location)
        data["seed"] = "99"
        self.assertEqual(self.client.post("/generate/fixture-qwen_2_1", data=data).status_code, 409)
        for name, value, status in (("prompt", "", 422), ("width", "48", 422), ("true_cfg_scale", "0.5", 422),
                                    ("sigmas", "1,0", 422), ("sigmas", "1,0.8,0.9", 422), ("sigmas", "nan", 422),
                                    ("sigmas", "word", 422), ("sampling.sampler", "euler", 422),
                                    ("csrf_token", "bad", 400), ("submission_key", "../escape", 400)):
            with self.subTest(name=name, value=value):
                response = self.client.post("/generate/fixture-qwen_2_1", data=self.data(**{name: value}))
                self.assertEqual(response.status_code, status)
        for loras in (("fixture-sdxl-style",), ("fixture-qwen_2_1-style",) * 2):
            self.assertEqual(self.client.post("/generate/fixture-qwen_2_1", data=self.data(loras=loras)).status_code, 422)
        self.assertEqual(len(self.store.list_jobs()), 1)
        self.runtime.assert_not_called()

    def test_local_loader_and_cpu_use_shared_runtime_options(self):
        workflow = self.workflow(cache_dir=self.root / "cache", token="test-secret", local_files_only=True)
        batch = workflow.generate(self.prepared())
        options = self.transformer_loader.call_args.kwargs
        self.assertEqual(options["config"], "Qwen/Qwen-Image-2.1")
        self.assertEqual(options["subfolder"], "transformer")
        self.assertEqual(options["dtype"], "bf16")
        self.assertEqual(options["token"], "test-secret")
        self.assertTrue(options["local_files_only"])
        self.assertEqual(options["cache_dir"], str(self.root / "cache"))
        self.assertIs(self.loader.call_args.kwargs["transformer"], self.transformer_loader.return_value.to.return_value)
        self.assertEqual(self.loader.call_args.kwargs["device_map"], "cuda")
        self.assertNotIn("is_distilled", self.loader.call_args.kwargs)
        self.assertNotIn("test-secret", json.dumps(batch.metadata))
        self.workflow(device="cpu").generate(self.prepared())
        self.assertEqual(self.loader.call_args.kwargs["dtype"], "fp32")
        self.assertNotIn("device_map", self.loader.call_args.kwargs)

    def test_hub_and_checkpoint_switching_and_replacement(self):
        workflow, request = self.workflow(), self.prepared()
        workflow.generate(request)
        workflow.generate(request)
        self.assertEqual(self.loader.call_count, 1)
        request.model.asset_paths["weights"].write_bytes(b"replacement")
        workflow.generate(request)
        self.assertEqual(self.loader.call_count, 2)
        request.model.defaults["weights"] = None
        request.model.defaults["weights_component"] = "pipeline"
        workflow.generate(request)
        self.assertEqual(self.transformer_loader.call_count, 2)
        self.assertNotIn("transformer", self.loader.call_args.kwargs)
        self.assertEqual(self.loader.call_count, 3)

    def test_repo_schedule_and_custom_override_drive_progress_and_metadata(self):
        self.repo_sigmas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        workflow = self.workflow()
        events = []
        workflow.set_progress_callback(events.append)
        batch = workflow.generate(self.prepared(num_inference_steps=40))
        self.assertEqual(batch.metadata["effective_num_inference_steps"], 8)
        self.assertEqual([event.total for event in events if event.stage == "denoising"], [8] * 9)
        batch = workflow.generate(self.prepared(sigmas=[1, 0.5], use_kv_cache=False))
        self.assertEqual(batch.metadata["effective_num_inference_steps"], 2)
        self.assertEqual(batch.metadata["effective_sample_sigmas"], [1, 0.5])
        self.assertFalse(self.calls[-1]["arguments"]["use_kv_cache"])

    def test_cfg_requires_both_scale_and_negative_prompt(self):
        workflow = self.workflow()
        for cfg, negative, active in ((1, "blur", False), (2.5, "", False), (2.5, "blur", True)):
            with self.subTest(cfg=cfg, negative=negative):
                batch = workflow.generate(self.prepared(true_cfg_scale=cfg, negative_prompt=negative))
                self.assertIs(batch.metadata["true_cfg_enabled"], active)
                self.assertEqual(self.calls[-1]["arguments"]["negative_prompt"], negative if active else None)

    def test_backend_rejects_wrong_architecture_fields_and_invalid_constraints_without_runtime(self):
        for changes in ({"guidance_scale": 3.5}, {"max_sequence_length": 512}, {"clip_skip": 1},
                        {"width": 48}, {"true_cfg_scale": 0}, {"sigmas": []}, {"sigmas": [1, False]},
                        {"sigmas": [1, float("inf")]}, {"sigmas": [1, 0.8, 0.9]}, {"sigmas": [1, 0]},
                        {"use_kv_cache": 1}, {"seed": 2**63}, {"sampling": {"sampler": "euler"}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepared(**changes)
        request = self.prepared()
        for key, value in (("model_variant", "qwen_image"), ("weights_component", "pipeline"), ("base_model_id", None)):
            model = self.catalog().get_model("fixture-qwen_2_1")
            model.defaults[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.workflow().validate(replace(request, model=model))
        request.model.defaults["requirements"]["dimension_multiple"] = 16
        with self.assertRaises(ValueError):
            self.workflow().validate(replace(request, settings={**request.settings, "width": 48}))
        self.runtime.assert_not_called()

    def test_multiple_signed_adapters_are_weighted_and_failed_load_discards_pipeline(self):
        workflow = self.workflow()
        request = self.prepared(loras=[{"id": "fixture-qwen_2_1-style", "enabled": True, "strength": -0.3},
                                       {"id": "fixture-qwen_2_1-detail", "enabled": True, "strength": 0.7}])
        workflow.generate(request)
        pipe = self.pipelines[0]
        pipe.set_adapters.assert_called_once_with(["adapter_0", "adapter_1"], adapter_weights=[-0.3, 0.7])
        pipe.set_lora_device.assert_called_once_with(adapter_names=["adapter_0", "adapter_1"], device="cuda:0")
        pipe.load_lora_into_transformer.side_effect = RuntimeError("bad tensors")
        with self.assertRaisesRegex(RuntimeError, "Could not load LoRA"):
            workflow.generate(request)
        self.assertIsNone(workflow._pipeline)
        workflow.generate(self.prepared(loras=[]))
        self.assertEqual(len(self.pipelines), 2)

    def test_unknown_adapter_targets_fail_before_peft_can_partially_apply_them(self):
        workflow = self.workflow()
        workflow.generate(self.prepared())
        pipe = self.pipelines[0]
        pipe.transformer.get_submodule.side_effect = AttributeError("unknown target")
        request = self.prepared(loras=[{"id": "fixture-qwen_2_1-style", "enabled": True}])
        with self.assertRaisesRegex(RuntimeError, "Could not load LoRA"):
            workflow.generate(request)
        pipe.load_lora_into_transformer.assert_not_called()
        self.assertIsNone(workflow._pipeline)

    def test_save_reload_and_reset_qwen_settings_and_disabled_signed_adapter(self):
        data = self.data(action="save_model", sigmas="1,0.8,0.4")
        data.pop("use_kv_cache")
        data.setlist("configured_lora", ["fixture-qwen_2_1-style"])
        data["strength:fixture-qwen_2_1-style"] = "-0.3"
        response = self.client.post("/settings/models/fixture-qwen_2_1", data=data)
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True))
        saved = self.catalog().get_model("fixture-qwen_2_1").user_overrides["settings"]
        self.assertEqual(saved["sigmas"], [1, 0.8, 0.4])
        self.assertFalse(saved["use_kv_cache"])
        self.assertEqual(saved["loras"], [{"id": "fixture-qwen_2_1-style", "enabled": False, "strength": -0.3}])
        reloaded = self.data()
        self.assertEqual(reloaded["sigmas"], "1.0\n0.8\n0.4")
        self.assertNotIn("use_kv_cache", reloaded)
        reloaded["sigmas"] = ""
        reloaded["action"] = "save_model"
        self.assertEqual(self.client.post("/settings/models/fixture-qwen_2_1", data=reloaded).status_code, 303)
        self.assertIsNone(self.catalog().get_model("fixture-qwen_2_1").resolve_settings()["sigmas"])
        reloaded = self.data()
        reloaded["action"] = "reset_model"
        self.assertEqual(self.client.post("/settings/models/fixture-qwen_2_1", data=reloaded).status_code, 303)
        self.assertEqual(self.catalog().get_model("fixture-qwen_2_1").user_overrides["settings"], {})
        self.runtime.assert_not_called()

    def test_shared_adapter_default_supports_negative_ratio_and_inherits_for_generation(self):
        data = self.data(action="save_adapter")
        data["adapter_strength:fixture-qwen_2_1-style"] = "-0.2"
        response = self.client.post("/settings/adapters/fixture-qwen_2_1-style", data=data)
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True))
        self.assertEqual(self.catalog().get_lora("fixture-qwen_2_1-style").resolve_settings()["strength"], -0.2)
        request = self.prepared(loras=[{"id": "fixture-qwen_2_1-style", "enabled": True}])
        self.assertEqual(request.loras[0].strength, -0.2)
        for value in ("nan", "inf"):
            data = self.data(action="save_adapter")
            data["adapter_strength:fixture-qwen_2_1-style"] = value
            self.assertEqual(self.client.post("/settings/adapters/fixture-qwen_2_1-style", data=data).status_code, 422)
        self.runtime.assert_not_called()

    def test_missing_weights_block_generation_but_not_preview(self):
        self.catalog().get_model("fixture-qwen_2_1").asset_paths["weights"].unlink()
        response = self.client.post("/generate/fixture-qwen_2_1", data=self.data())
        self.assertEqual(response.status_code, 422)
        self.assertIn("weights not found", response.get_data(as_text=True))
        self.assertEqual(self.client.post("/generate/fixture-qwen_2_1", data=self.data(action="preview")).status_code, 200)
        self.assertEqual(self.store.list_jobs(), ())
        self.runtime.assert_not_called()

    def test_cli_dry_run_is_stdlib_only_and_exposes_qwen_controls(self):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "scripts/generate_qwen21.py"), "--root", str(self.catalog_root),
             "--model-id", "fixture-qwen_2_1", "--prompt", "A lake", "--seed", "random", "--dry-run",
             "--sigmas", "1", "0.8", "0.4", "--no-kv-cache", "--lora", "fixture-qwen_2_1-style:-0.2"],
            cwd="/tmp", capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["model"]["workflow"], "qwen_2_1")
        self.assertIsNone(data["settings"]["seed"])
        self.assertEqual(data["settings"]["sigmas"], [1, 0.8, 0.4])
        self.assertFalse(data["settings"]["use_kv_cache"])
        self.assertEqual(data["loras"][0]["strength"], -0.2)
        self.runtime.assert_not_called()


class Matrix:
    """Small tensor substitute for testing fused LoRA algebra without Torch."""
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def chunk(self, count, dim):
        if count != 2 or dim != 0:
            raise ValueError("This fixture splits output rows only")
        middle = len(self.rows) // 2
        return Matrix(self.rows[:middle]), Matrix(self.rows[middle:])

    def multiply(self, other):
        return [[sum(a * b for a, b in zip(row, column)) for column in zip(*other.rows)] for row in self.rows]


class FusedLoRATests(unittest.TestCase):
    def test_gate_and_projection_deltas_reconstruct_the_original_fused_delta(self):
        prefix = "transformer.transformer_blocks.0.img_mlp."
        down = Matrix([[1, 2, 3], [4, 5, 6]])
        up = Matrix([[1, 2], [3, 4], [5, 6], [7, 8]])
        state = {prefix + "gate_up.lora_A.weight": down, prefix + "gate_up.lora_B.weight": up}
        converted = split_fused_lora(state)
        self.assertEqual(len(state), 2)  # The caller's original mapping is preserved.
        expected = up.multiply(down)
        for name, rows in (("gate_layer", expected[:2]), ("proj", expected[2:])):
            a = converted[prefix + name + ".lora_A.weight"]
            b = converted[prefix + name + ".lora_B.weight"]
            self.assertEqual(b.multiply(a), rows)
        self.assertFalse(any("gate_up" in key for key in converted))

    def test_conflicting_or_invalid_fused_weights_are_rejected_and_native_weights_preserved(self):
        key = "transformer.transformer_blocks.0.img_mlp.gate_up."
        down = Matrix([[1, 2]])
        state = {key + "lora_A.weight": down, key + "lora_B.weight": Matrix([[1], [2], [3]])}
        with self.assertRaisesRegex(ValueError, "dimensions"):
            split_fused_lora(state)
        state[key + "lora_B.weight"] = Matrix([[1], [2]])
        state[key.replace("gate_up", "gate_layer") + "lora_A.weight"] = down
        with self.assertRaisesRegex(ValueError, "both fused and separate"):
            split_fused_lora(state)
        state = {"transformer.img_in.lora_A.weight": down, "transformer.img_in.lora_B.weight": down}
        self.assertEqual(split_fused_lora(state), state)


if __name__ == "__main__":
    unittest.main()

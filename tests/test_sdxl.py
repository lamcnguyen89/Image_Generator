"""Validate SDXL behavior without downloading weights or requiring CUDA."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import CatalogValidationError, discover_catalog
from image_generator.generation import prepare_generation, request_snapshot, run_generation, save_batch
from image_generator.workflows.base import GeneratedBatch
from image_generator.workflows.registry import create_workflow
from image_generator.workflows.sdxl import SDXLWorkflow


class DefaultScheduler:
    def __init__(self, config=None):
        self.config = dict(config or {"prediction_type": "epsilon", "timestep_spacing": "leading"})

    @classmethod
    def from_config(cls, config, **options):
        return cls({**config, **options})


class EulerScheduler(DefaultScheduler):
    pass


class AncestralScheduler(DefaultScheduler):
    pass


class DPMScheduler(DefaultScheduler):
    def __init__(self, config=None):
        super().__init__({**(config or {}), "lambda_min_clipped": float("-inf")})


class FakeGenerator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakePipeline:
    def __init__(self):
        self.scheduler = DefaultScheduler()
        self.to = Mock(return_value=self)
        self.load_lora_weights = Mock()
        self.unload_lora_weights = Mock()
        self.set_adapters = Mock()
        self.invoke = Mock(side_effect=lambda **kwargs: SimpleNamespace(
            images=[Image.new("RGB", (16, 16), "navy") for _ in range(kwargs["num_images_per_prompt"])],
        ))

    def __call__(self, **kwargs):
        return self.invoke(**kwargs)


class SDXLTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_path = self.root / "SDXL/checkpoints/base/model.defaults.json"
        self.model = {
            "schema_version": 1, "id": "base", "display_name": "Base model", "family": "sdxl",
            "workflow": "sdxl", "weights": None, "base_model_id": "example/sdxl",
            "supported_modes": ["txt2img"], "requirements": {"dimension_multiple": 8, "scheduler_config": {}},
            "settings": {"prompt": "a lake", "seed": 42},
        }
        self.write(self.model_path, self.model)
        self.pipelines = []

        def pipeline_factory(*args, **kwargs):
            pipeline = FakePipeline()
            self.pipelines.append(pipeline)
            return pipeline

        self.loader = SimpleNamespace(
            from_pretrained=Mock(side_effect=pipeline_factory),
            from_single_file=Mock(side_effect=pipeline_factory),
        )
        self.torch = SimpleNamespace(
            __version__="test-torch", float16="fp16", float32="fp32", bfloat16="bf16",
            Generator=FakeGenerator, inference_mode=nullcontext,
            cuda=SimpleNamespace(
                is_available=Mock(return_value=True), is_bf16_supported=Mock(return_value=True),
                synchronize=Mock(), empty_cache=Mock(),
            ),
        )
        self.diffusers = SimpleNamespace(
            __version__="test-diffusers", StableDiffusionXLPipeline=self.loader,
            EulerDiscreteScheduler=EulerScheduler, EulerAncestralDiscreteScheduler=AncestralScheduler,
            DPMSolverMultistepScheduler=DPMScheduler,
        )
        self.runtime_import = patch(
            "image_generator.workflows.sdxl.runtime.import_module",
            side_effect={"torch": self.torch, "diffusers": self.diffusers}.__getitem__,
        ).start()
        self.addCleanup(patch.stopall)
        self.workflow = SDXLWorkflow()
        self.addCleanup(self.workflow.close)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def catalog(self):
        return discover_catalog(self.root)

    def request(self, **overrides):
        return prepare_generation(self.catalog(), "base", overrides=overrides)

    def add_lora(self, identifier="style", **changes):
        path = self.root / f"SDXL/loras/{identifier}/lora.defaults.json"
        self.write(path, {
            "schema_version": 1, "id": identifier, "display_name": identifier,
            "family": "sdxl", "adapter_type": "lora", "weights": "adapter.safetensors",
            "compatibility": {"verified_checkpoint_ids": []},
            "settings": {"strength": 0.8, "trigger_words": ["style-trigger"],
                         "positive_prefix": "adapter-prefix", "recommended_generation": {"width": 512}},
            **changes,
        })
        path.with_name("adapter.safetensors").write_bytes(b"mock weights")
        return path

    def test_base_model_default_scheduler_and_exact_pipeline_arguments(self):
        request = self.request()
        batch = self.workflow.generate(request)
        self.loader.from_pretrained.assert_called_once_with(
            "example/sdxl", dtype="fp16", use_safetensors=True, local_files_only=False, variant="fp16",
        )
        pipeline = self.pipelines[0]
        self.assertIsInstance(pipeline.scheduler, DefaultScheduler)
        self.assertEqual(batch.metadata["seeds"], [42])
        args = pipeline.invoke.call_args.kwargs
        self.assertEqual(args["prompt"], "a lake")
        self.assertEqual(args["width"], 1024)
        self.assertEqual(args["generator"][0].seed, 42)
        self.assertNotIn("seed", args)
        self.assertNotIn("sampling", args)
        self.assertNotIn("clip_skip", args)
        pipeline.load_lora_weights.assert_not_called()
        pipeline.set_adapters.assert_not_called()

    def test_single_file_and_required_noobai_scheduler_options(self):
        checkpoint = self.model_path.with_name("checkpoint.safetensors")
        checkpoint.write_bytes(b"mock weights")
        self.write(self.model_path, {**self.model, "weights": checkpoint.name,
            "requirements": {"dimension_multiple": 8,
                "scheduler_config": {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True},
                "allowed_sampling": [{"sampler": "euler", "noise_schedule": "default"}]},
            "settings": {**self.model["settings"], "sampling": {"sampler": "euler", "noise_schedule": "default"}},
        })
        batch = self.workflow.generate(self.request())
        self.loader.from_single_file.assert_called_once_with(
            str(checkpoint), dtype="fp16", use_safetensors=True, local_files_only=False, config="example/sdxl",
        )
        self.assertIsInstance(self.pipelines[0].scheduler, EulerScheduler)
        self.assertEqual(batch.metadata["scheduler"]["config"]["prediction_type"], "v_prediction")
        self.assertTrue(batch.metadata["scheduler"]["config"]["rescale_betas_zero_snr"])
        with self.assertRaisesRegex(ValueError, "prohibited"):
            self.request(sampling={"sampler": "model_default"})

    def test_scheduler_switches_do_not_leak_config_and_pipeline_is_reused(self):
        batch = self.workflow.generate(self.request(sampling={"sampler": "dpmpp_2m_sde"}))
        self.assertEqual(batch.metadata["scheduler"]["config"]["algorithm_type"], "sde-dpmsolver++")
        self.assertEqual(batch.metadata["scheduler"]["config"]["solver_order"], 2)
        self.workflow.generate(self.request(sampling={"sampler": "euler_ancestral"}))
        self.assertIsInstance(self.pipelines[0].scheduler, AncestralScheduler)
        self.workflow.generate(self.request(sampling={"sampler": "model_default"}))
        self.assertIsInstance(self.pipelines[0].scheduler, DefaultScheduler)
        self.assertNotIn("algorithm_type", self.pipelines[0].scheduler.config)
        self.loader.from_pretrained.assert_called_once()

    def test_metadata_handles_scheduler_infinity_sentinels_as_strict_json(self):
        batch = self.workflow.generate(self.request(sampling={"sampler": "dpmpp_2m_sde"}))
        json.dumps(batch.metadata, allow_nan=False)
        saved = save_batch(batch, self.root / "images")
        metadata = json.loads(saved[0].metadata_path.read_text())
        self.assertEqual(metadata["scheduler"]["config"]["lambda_min_clipped"], "-Infinity")

    def test_multiple_adapters_strength_precedence_and_reset(self):
        path = self.add_lora()
        self.write(path.with_name("lora.user.json"), {"schema_version": 1, "settings": {"strength": 0.6}})
        self.add_lora("detail")
        links = [{"id": "style", "enabled": True, "strength": 0.3}, {"id": "detail", "enabled": True}]
        batch = self.workflow.generate(self.request(loras=links))
        pipeline = self.pipelines[0]
        self.assertEqual(pipeline.load_lora_weights.call_count, 2)
        pipeline.set_adapters.assert_called_once_with(["adapter_0", "adapter_1"], adapter_weights=[0.3, 0.8])
        self.assertEqual([item["strength"] for item in batch.metadata["loras"]], [0.3, 0.8])
        self.assertEqual(batch.metadata["settings"]["width"], 1024)
        self.assertNotIn("style-trigger", batch.metadata["prompt"])
        self.assertNotIn("adapter-prefix", batch.metadata["prompt"])
        self.workflow.generate(self.request(loras=[]))
        pipeline.unload_lora_weights.assert_called_once()
        self.loader.from_pretrained.assert_called_once()

    def test_disabled_adapters_do_not_require_local_weights(self):
        path = self.add_lora()
        path.with_name("adapter.safetensors").unlink()
        batch = self.workflow.generate(self.request(loras=[{"id": "style", "enabled": False}]))
        self.assertEqual(batch.metadata["loras"], [])
        self.pipelines[0].load_lora_weights.assert_not_called()

    def test_prompt_assembly_and_opt_in_trigger_words(self):
        self.add_lora()
        self.write(self.model_path, {**self.model, "settings": {
            "prompt": "a lake", "positive_prefix": "quality,", "positive_suffix": "sunlight",
            "trigger_words": ["model-trigger", "style-trigger"],
        }})
        request = prepare_generation(self.catalog(), "base", include_trigger_words=True,
            overrides={"loras": [{"id": "style", "enabled": True}]})
        self.assertEqual(request.prompt, "quality, model-trigger, style-trigger a lake sunlight")

    def test_random_seeds_and_fixed_seed_sequence_are_recorded(self):
        with patch("image_generator.workflows.sdxl.runtime.secrets.randbelow", side_effect=[11, 23]):
            batch = self.workflow.generate(self.request(seed=None, num_images_per_prompt=2))
        self.assertEqual(batch.metadata["seeds"], [11, 23])
        batch = self.workflow.generate(self.request(seed=2**63 - 1, num_images_per_prompt=2, clip_skip=1))
        self.assertEqual(batch.metadata["seeds"], [2**63 - 1, 0])
        self.assertEqual(self.pipelines[0].invoke.call_args.kwargs["clip_skip"], 1)

    def test_preparation_snapshots_catalog_and_preserves_files(self):
        catalog = self.catalog()
        original = self.model_path.read_bytes()
        request = prepare_generation(catalog, "base")
        catalog.models["base"].defaults["requirements"]["dimension_multiple"] = 32
        catalog.models["base"].defaults["settings"]["prompt"] = "changed"
        self.assertEqual(request.model.defaults["requirements"]["dimension_multiple"], 8)
        self.assertEqual(request.prompt, "a lake")
        self.assertEqual(self.model_path.read_bytes(), original)

    def test_invalid_settings_fail_before_runtime_import(self):
        for changes in (
            {"prompt": " "}, {"width": 1025}, {"height": 0}, {"num_inference_steps": -1},
            {"num_images_per_prompt": 0}, {"seed": -1}, {"seed": 2**63},
            {"guidance_scale": -0.1}, {"guidance_scale": float("nan")}, {"clip_skip": -1},
            {"sampling": {"sampler": "unknown"}}, {"sampling": {"noise_schedule": "karras"}},
            {"not_a_setting": True},
        ):
            with self.subTest(changes=changes), self.assertRaises((ValueError, CatalogValidationError)):
                self.request(**changes)
        self.runtime_import.assert_not_called()

    def test_missing_checkpoint_and_adapter_weights_fail_before_runtime_import(self):
        self.write(self.model_path, {**self.model, "weights": "missing.safetensors"})
        with self.assertRaisesRegex(ValueError, "weights not found"):
            self.request()
        self.write(self.model_path, self.model)
        path = self.add_lora()
        path.with_name("adapter.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "weights not found"):
            self.request(loras=[{"id": "style", "enabled": True}])
        self.runtime_import.assert_not_called()

    def test_wrong_adapter_family_negative_strength_and_unknown_id_are_rejected(self):
        self.add_lora(family="flux")
        with self.assertRaisesRegex(ValueError, "SDXL compatibility"):
            self.request(loras=[{"id": "style", "enabled": True}])
        self.add_lora()
        with self.assertRaisesRegex(ValueError, "strength"):
            self.request(loras=[{"id": "style", "enabled": True, "strength": -0.1}])
        with self.assertRaisesRegex(ValueError, "Unknown LoRA"):
            self.request(loras=[{"id": "missing", "enabled": True}])

    def test_unknown_workflow_model_and_unsupported_mode_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            create_workflow("unregistered-workflow")
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            prepare_generation(self.catalog(), "missing")
        with self.assertRaisesRegex(ValueError, "txt2img"):
            prepare_generation(self.catalog(), "base", mode="img2img")

    def test_cuda_failure_does_not_load_or_download_models(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            self.workflow.generate(self.request())
        self.loader.from_pretrained.assert_not_called()
        self.assertEqual(self.runtime_import.call_args_list[0].args, ("torch",))
        self.assertEqual(self.runtime_import.call_count, 1)

    def test_cpu_uses_float32_and_forwards_offline_cache_options(self):
        workflow = SDXLWorkflow(device="cpu", local_files_only=True, cache_dir=self.root / "cache", token="private-token")
        self.addCleanup(workflow.close)
        batch = workflow.generate(self.request())
        options = self.loader.from_pretrained.call_args.kwargs
        self.assertEqual(options["dtype"], "fp32")
        self.assertNotIn("variant", options)
        self.assertTrue(options["local_files_only"])
        self.assertEqual(options["cache_dir"], str(self.root / "cache"))
        self.assertNotIn("private-token", json.dumps(batch.metadata))
        with self.assertRaises(ValueError):
            SDXLWorkflow(device="cpu", dtype="float16")

    def test_failed_adapter_load_discards_pipeline_before_next_request(self):
        self.add_lora()
        self.workflow.generate(self.request())
        self.pipelines[0].load_lora_weights.side_effect = RuntimeError("incompatible tensors")
        with self.assertRaisesRegex(RuntimeError, "Could not load LoRA"):
            self.workflow.generate(self.request(loras=[{"id": "style", "enabled": True}]))
        self.assertIsNone(self.workflow._pipeline)
        self.workflow.generate(self.request())
        self.assertEqual(len(self.pipelines), 2)

    def test_failed_inference_discards_pipeline(self):
        self.workflow.generate(self.request())
        self.pipelines[0].invoke.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.workflow.generate(self.request())
        self.assertIsNone(self.workflow._pipeline)

    def test_model_file_replacement_and_model_switch_reload_pipeline(self):
        path = self.model_path.with_name("checkpoint.safetensors")
        path.write_bytes(b"first")
        self.write(self.model_path, {**self.model, "weights": path.name})
        self.workflow.generate(self.request())
        path.write_bytes(b"replacement with different size")
        self.workflow.generate(self.request())
        self.assertEqual(self.loader.from_single_file.call_count, 2)
        self.write(self.model_path, self.model)
        self.workflow.generate(self.request())
        self.loader.from_pretrained.assert_called_once()

    def test_saves_actual_png_and_complete_per_image_metadata_without_overwriting(self):
        request = self.request(num_images_per_prompt=2)
        directory = self.root / "images"
        first = run_generation(request, directory)
        second = run_generation(request, directory)
        self.assertEqual(len(list(directory.glob("*.png"))), 4)
        self.assertNotEqual(first[0].image_path, second[0].image_path)
        with Image.open(first[0].image_path) as image:
            self.assertEqual(image.format, "PNG")
        metadata = json.loads(first[1].metadata_path.read_text())
        self.assertEqual(metadata["seed"], 43)
        self.assertEqual(metadata["image_index"], 1)
        self.assertEqual(metadata["model"]["id"], "base")
        self.assertEqual(metadata["runtime"]["torch_version"], "test-torch")
        self.assertEqual(metadata["image_file"], first[1].image_path.name)

    def test_save_errors_do_not_leave_orphaned_image(self):
        batch = self.workflow.generate(self.request())
        original_write = Path.write_text

        def fail_metadata(path, *args, **kwargs):
            if path.suffix == ".json":
                raise OSError("disk full")
            return original_write(path, *args, **kwargs)

        with patch.object(Path, "write_text", fail_metadata), self.assertRaisesRegex(OSError, "disk full"):
            save_batch(batch, self.root / "images")
        self.assertEqual(list((self.root / "images").iterdir()), [])

    def test_invalid_batch_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "count"):
            save_batch(GeneratedBatch((), {"seeds": [1]}), self.root / "images")

    def test_request_snapshot_validates_without_loading_runtime_or_saving_outputs(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        request = self.request(prompt="a forest", negative_prompt="blurry", seed=None)
        data = json.loads(json.dumps(request_snapshot(request), allow_nan=False))
        self.assertEqual(data["model"]["id"], "base")
        self.assertEqual(data["model"]["workflow"], "sdxl")
        self.assertEqual(data["prompt"], "a forest")
        self.assertEqual(data["negative_prompt"], "blurry")
        self.assertIsNone(data["settings"]["seed"])
        self.assertEqual(data["loras"], [])
        self.runtime_import.assert_not_called()
        after = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, {path: after[path] for path in before})
        for path in after.keys() - before.keys():
            self.assertEqual(path.name, "model.user.json")
            self.assertEqual(json.loads(after[path]), {"schema_version": 1, "settings": {}})

    def test_run_generation_releases_pipeline_on_success_and_save_failure(self):
        request = self.request(prompt="a forest", seed=99)
        self.workflow = SDXLWorkflow(device="cpu", local_files_only=True)
        self.addCleanup(self.workflow.close)
        with patch("image_generator.generation.create_workflow", return_value=self.workflow) as factory:
            outputs = run_generation(request, self.root / "output", device="cpu", local_files_only=True)
        factory.assert_called_once_with("sdxl", device="cpu", local_files_only=True)
        self.assertIsNone(self.workflow._pipeline)
        self.assertEqual(self.loader.from_pretrained.call_args.kwargs["dtype"], "fp32")
        self.assertTrue(self.loader.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertEqual(outputs[0].seed, 99)
        with Image.open(outputs[0].image_path) as image:
            self.assertEqual(image.format, "PNG")
        metadata = json.loads(outputs[0].metadata_path.read_text())
        self.assertEqual((metadata["prompt"], metadata["seed"]), ("a forest", 99))

        failed_directory = self.root / "failed-output"
        with patch("image_generator.generation.create_workflow", return_value=self.workflow), \
                patch("image_generator.generation.save_batch", side_effect=OSError("disk full")), \
                self.assertRaisesRegex(OSError, "disk full"):
            run_generation(request, failed_directory)
        self.assertIsNone(self.workflow._pipeline)
        self.assertFalse(failed_directory.exists())
        self.assertEqual(len(self.pipelines), 2)

    def test_unexpected_pipeline_image_count_discards_pipeline_and_recovers(self):
        request = self.request(num_images_per_prompt=2)
        self.workflow.generate(request)
        self.pipelines[0].invoke.side_effect = lambda **kwargs: SimpleNamespace(
            images=[Image.new("RGB", (16, 16))])
        with self.assertRaisesRegex(RuntimeError, "unexpected number of images"):
            self.workflow.generate(request)
        self.assertIsNone(self.workflow._pipeline)
        batch = self.workflow.generate(request)
        self.assertEqual((len(batch.images), batch.metadata["seeds"]), (2, [42, 43]))
        self.assertEqual(len(self.pipelines), 2)

    def test_saved_loras_can_be_overridden_or_explicitly_disabled_per_request(self):
        self.add_lora()
        self.add_lora("detail")
        user_path = self.write(self.model_path.with_name("model.user.json"), {
            "schema_version": 1, "settings": {"loras": [{"id": "style", "enabled": True}]},
        })
        saved_settings = user_path.read_bytes()
        self.assertEqual([selection.asset.id for selection in self.request().loras], ["style"])
        request = self.request(loras=[{"id": "style", "enabled": True, "strength": 0.4},
                                      {"id": "detail", "enabled": True}])
        self.assertEqual([(selection.asset.id, selection.strength) for selection in request.loras],
                         [("style", 0.4), ("detail", 0.8)])
        self.assertEqual(self.request(loras=[]).loras, ())
        self.assertEqual(user_path.read_bytes(), saved_settings)
        self.runtime_import.assert_not_called()


if __name__ == "__main__":
    unittest.main()

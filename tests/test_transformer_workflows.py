"""FLUX/WAN request, loader, adapter, streaming, and CLI checks without weights."""

from contextlib import nullcontext, redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import discover_catalog
from image_generator.cli import run_family
from image_generator.generation import prepare_generation, run_generation
from image_generator.workflows.flux import FluxWorkflow
from image_generator.workflows.wan import WanWorkflow


class FlowScheduler:
    def __init__(self, config=None):
        self.config = config or {"num_train_timesteps": 1000}

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
    def __init__(self, family, vae=None):
        self.family = family
        self.scheduler = FlowScheduler()
        self.transformer = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)), set_adapters=Mock())
        self.transformer_2 = SimpleNamespace(set_adapters=Mock())
        self.config = SimpleNamespace(boundary_ratio=0.9)
        self.vae_scale_factor_spatial, self.vae_scale_factor_temporal = 8, 4
        self.vae = vae or SimpleNamespace(dtype="fp32", enable_tiling=Mock(), disable_tiling=Mock())
        self.to = Mock(return_value=self)
        self.load_lora_weights, self.unload_lora_weights, self.set_adapters = Mock(), Mock(), Mock()
        self.invoke = Mock(side_effect=self.result)

    def result(self, **kwargs):
        if self.family == "flux":
            return SimpleNamespace(images=[Image.new("RGB", (16, 16)) for _ in range(kwargs["num_images_per_prompt"])])
        return SimpleNamespace(frames=[[Image.new("RGB", (kwargs["width"], kwargs["height"])) for _ in range(kwargs["num_frames"])]])

    def __call__(self, **kwargs):
        return self.invoke(**kwargs)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input.png"
        Image.new("RGB", (300, 180), "navy").save(self.source)
        self.flux_path = self.root / "FLUX/checkpoints/demo/model.defaults.json"
        self.wan_path = self.root / "WAN/checkpoints/demo/model.defaults.json"
        self.flux_model = {
            "schema_version": 1, "id": "flux-model", "display_name": "FLUX test", "family": "flux", "workflow": "flux",
            "weights": "transformer.safetensors", "weights_component": "transformer", "model_variant": "flux_1_dev",
            "base_model_id": "example/flux", "supported_modes": ["txt2img"],
            "requirements": {"dimension_multiple": 16, "max_sequence_length_limit": 512, "scheduler_config": {}},
            "settings": {"prompt": "a lake", "seed": 42, "width": 64, "height": 32, "num_inference_steps": 2},
        }
        self.wan_model = {
            "schema_version": 1, "id": "wan-model", "display_name": "WAN test", "family": "wan", "workflow": "wan",
            "weights": None, "weights_component": "pipeline", "model_variant": "wan_2_2_i2v_a14b",
            "base_model_id": "example/wan", "supported_modes": ["img2vid"],
            "requirements": {"dimension_multiple": 16, "frame_count": {"minimum": 5, "step": 4, "offset": 1}, "scheduler_config": {}},
            "settings": {"prompt": "a moving lake", "seed": 42, "width": 64, "height": 32, "num_frames": 5, "num_inference_steps": 2},
        }
        self.write(self.flux_path, self.flux_model)
        self.flux_path.with_name("transformer.safetensors").write_bytes(b"mock weights")
        self.write(self.wan_path, self.wan_model)
        self.pipelines = []

        def factory(family):
            def build(*args, **kwargs):
                pipe = Pipeline(family, kwargs.get("vae"))
                self.pipelines.append(pipe)
                return pipe
            return build

        self.flux_loader = Mock(side_effect=factory("flux"))
        self.wan_loader = Mock(side_effect=factory("wan"))
        self.transformer_loader = Mock(return_value=SimpleNamespace())
        self.vae_loader = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(
            dtype=kwargs["dtype"], enable_tiling=Mock(), disable_tiling=Mock(),
        ))
        self.torch = SimpleNamespace(
            __version__="test-torch", float16="fp16", bfloat16="bf16", float32="fp32",
            Generator=Generator, inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=Mock(return_value=True), is_bf16_supported=Mock(return_value=True),
                                 device=lambda device: nullcontext(), synchronize=Mock(), empty_cache=Mock()),
        )
        self.diffusers = SimpleNamespace(
            __version__="test-diffusers", FlowMatchEulerDiscreteScheduler=FlowScheduler,
            FluxPipeline=SimpleNamespace(from_pretrained=self.flux_loader),
            FluxTransformer2DModel=SimpleNamespace(from_single_file=self.transformer_loader),
            WanImageToVideoPipeline=SimpleNamespace(from_pretrained=self.wan_loader),
            AutoencoderKLWan=SimpleNamespace(from_pretrained=self.vae_loader),
        )
        self.runtime = patch("image_generator.workflows.shared.diffusers_runtime.import_module", side_effect={"torch": self.torch, "diffusers": self.diffusers}.__getitem__).start()
        self.encoder = patch("image_generator.workflows.wan.runtime.check_video_encoder").start()
        self.exporter = patch("image_generator.generation.export_video", side_effect=lambda frames, path, fps: path.write_bytes(b"mock MP4")).start()
        self.addCleanup(patch.stopall)
        self.flux_workflow, self.wan_workflow = FluxWorkflow(), WanWorkflow()
        self.addCleanup(self.flux_workflow.close)
        self.addCleanup(self.wan_workflow.close)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def request(self, family, **overrides):
        return prepare_generation(discover_catalog(self.root), f"{family}-model", overrides=overrides,
                                  inputs={"input_image": self.source} if family == "wan" else None)

    def add_lora(self, family, name="style", target="high_noise", variant=None):
        path = self.root / f"{family.upper()}/loras/{name}/lora.defaults.json"
        self.write(path, {
            "schema_version": 1, "id": name, "display_name": name, "family": family,
            "adapter_type": "lora", "weights": "adapter.safetensors",
            "compatibility": {"verified_checkpoint_ids": [], "model_variant": variant or ("flux_1_dev" if family == "flux" else "wan_2_2_i2v_a14b"), "expert_target": target},
            "settings": {"strength": 0.8, "trigger_words": ["style"], "recommended_generation": {"guidance_scale": 6.0}},
        })
        path.with_name("adapter.safetensors").write_bytes(b"mock LoRA")
        return path

    def test_flux_transformer_loading_bfloat16_and_separate_guidance(self):
        batch = self.flux_workflow.generate(self.request("flux", guidance_scale=3.5, true_cfg_scale=2.0, negative_prompt="blur"))
        args = self.transformer_loader.call_args
        self.assertEqual(args.args, (str(self.flux_path.with_name("transformer.safetensors")),))
        self.assertEqual(args.kwargs["subfolder"], "transformer")
        self.assertEqual(args.kwargs["dtype"], "bf16")
        self.assertIs(self.flux_loader.call_args.kwargs["transformer"], self.transformer_loader.return_value)
        call = self.pipelines[0].invoke.call_args.kwargs
        self.assertEqual(call["guidance_scale"], 3.5)
        self.assertEqual(call["true_cfg_scale"], 2.0)
        self.assertEqual(call["negative_prompt"], "blur")
        self.assertEqual(call["max_sequence_length"], 512)
        self.assertTrue(batch.metadata["true_cfg_enabled"])
        self.assertEqual(batch.metadata["seeds"], [42])

    def test_flux_true_cfg_disabled_without_negative_prompt(self):
        batch = self.flux_workflow.generate(self.request("flux", true_cfg_scale=2.0, negative_prompt=""))
        self.assertFalse(batch.metadata["true_cfg_enabled"])

    def test_flux_pretrained_without_transformer_file(self):
        self.write(self.flux_path, {**self.flux_model, "weights": None})
        self.flux_workflow.generate(self.request("flux"))
        self.transformer_loader.assert_not_called()
        self.assertNotIn("transformer", self.flux_loader.call_args.kwargs)

    def test_flux_multiple_adapters_and_overwritten_parameter_reset(self):
        self.add_lora("flux")
        self.add_lora("flux", "detail")
        request = self.request("flux", loras=[{"id": "style", "enabled": True, "strength": 0.3}, {"id": "detail", "enabled": True}])
        batch = self.flux_workflow.generate(request)
        pipe = self.pipelines[0]
        pipe.set_adapters.assert_called_once_with(["adapter_0", "adapter_1"], adapter_weights=[0.3, 0.8])
        self.assertEqual(batch.metadata["settings"]["guidance_scale"], 3.5)
        self.flux_workflow.generate(self.request("flux", loras=[]))
        pipe.unload_lora_weights.assert_called_once_with(reset_to_overwritten_params=True)
        self.flux_loader.assert_called_once()

    def test_flux_validation_before_runtime(self):
        for changes in ({"max_sequence_length": 513}, {"max_sequence_length": 0}, {"true_cfg_scale": 0.9},
                        {"width": 63}, {"sampling": {"sampler": "euler"}}, {"seed": -1}, {"unknown": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.request("flux", **changes)
        self.runtime.assert_not_called()

    def test_flux_checkpoint_replacement_reloads_pipeline(self):
        self.flux_workflow.generate(self.request("flux"))
        self.flux_path.with_name("transformer.safetensors").write_bytes(b"different weights")
        self.flux_workflow.generate(self.request("flux"))
        self.assertEqual(self.flux_loader.call_count, 2)

    def test_wan_vae_float32_input_hash_alignment_and_metadata(self):
        batch = self.wan_workflow.generate(self.request("wan"))
        self.assertEqual(self.vae_loader.call_args.kwargs["dtype"], "fp32")
        self.assertEqual(self.wan_loader.call_args.kwargs["dtype"], "bf16")
        args = self.pipelines[0].invoke.call_args.kwargs
        self.assertEqual(args["num_videos_per_prompt"], 1)
        self.assertEqual(args["output_type"], "np")
        self.assertEqual(args["generator"].seed, 42)
        self.assertEqual(args["image"].size, (48, 32))
        self.assertEqual(batch.metadata["input_image"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(batch.metadata["input_image"]["conditioning_size"], [48, 32])
        self.assertEqual(batch.metadata["frame_count"], 5)
        for field in ("fps", "preserve_aspect_ratio", "enable_vae_tiling"):
            self.assertNotIn(field, args)
        self.pipelines[0].vae.enable_tiling.assert_called_once()

    def test_wan_center_crop_and_tiling_can_be_disabled(self):
        self.wan_workflow.generate(self.request("wan", preserve_aspect_ratio=False, enable_vae_tiling=False))
        pipe = self.pipelines[0]
        self.assertEqual(pipe.invoke.call_args.kwargs["image"].size, (64, 32))
        pipe.vae.disable_tiling.assert_called_once()

    def test_wan_expert_targeted_multiple_adapter_strengths_and_reset(self):
        self.add_lora("wan", "high", target="high_noise")
        self.add_lora("wan", "low", target="low_noise")
        self.wan_workflow.generate(self.request("wan", loras=[{"id": "high", "enabled": True, "strength": 0.4}, {"id": "low", "enabled": True, "strength": 0.7}]))
        pipe = self.pipelines[0]
        self.assertFalse(pipe.load_lora_weights.call_args_list[0].kwargs["load_into_transformer_2"])
        self.assertTrue(pipe.load_lora_weights.call_args_list[1].kwargs["load_into_transformer_2"])
        pipe.transformer.set_adapters.assert_called_once_with(["adapter_0"], weights=[0.4])
        pipe.transformer_2.set_adapters.assert_called_once_with(["adapter_1"], weights=[0.7])
        pipe.set_adapters.assert_not_called()
        self.wan_workflow.generate(self.request("wan", loras=[]))
        pipe.unload_lora_weights.assert_called_once_with()

    def test_wan_unknown_expert_and_variant_are_rejected_before_runtime(self):
        path = self.add_lora("wan", target=None)
        with self.assertRaisesRegex(ValueError, "expert_target"):
            self.request("wan", loras=[{"id": "style", "enabled": True}])
        data = json.loads(path.read_text())
        data["compatibility"].update(expert_target="high_noise", model_variant=None)
        self.write(path, data)
        with self.assertRaisesRegex(ValueError, "model_variant"):
            self.request("wan", loras=[{"id": "style", "enabled": True}])
        self.runtime.assert_not_called()

    def test_disabled_unknown_wan_adapter_is_ignored(self):
        self.add_lora("wan", target=None)
        self.assertEqual(self.request("wan", loras=[{"id": "style", "enabled": False}]).loras, ())

    def test_wan_frames_guidance_and_input_validation(self):
        for changes in ({"num_frames": 4}, {"num_frames": 6}, {"fps": 0}, {"guidance_scale_2": 1.0},
                        {"guidance_scale_2": -1}, {"sampling": {"sampler": "euler"}}, {"num_videos_per_prompt": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.request("wan", **changes)
        with self.assertRaisesRegex(ValueError, "input_image"):
            prepare_generation(discover_catalog(self.root), "wan-model")
        self.runtime.assert_not_called()

    def test_invalid_input_image_content_fails_before_loading(self):
        self.source.write_bytes(b"invalid image")
        with self.assertRaises(OSError):
            self.wan_workflow.generate(self.request("wan"))
        self.runtime.assert_not_called()

    def test_bad_encoder_fails_before_loading(self):
        self.encoder.side_effect = RuntimeError("encoder unavailable")
        with self.assertRaisesRegex(RuntimeError, "encoder"):
            self.wan_workflow.generate(self.request("wan"))
        self.runtime.assert_not_called()

    def test_wan_checks_loaded_experts_boundary_and_temporal_alignment(self):
        for change in ("missing-expert", "bad-boundary", "temporal", "spatial"):
            with self.subTest(change=change):
                self.wan_workflow.close()
                def build(*args, **kwargs):
                    pipe = Pipeline("wan", kwargs["vae"])
                    if change == "missing-expert":
                        pipe.transformer_2 = None
                    elif change == "bad-boundary":
                        pipe.config.boundary_ratio = None
                    elif change == "temporal":
                        pipe.vae_scale_factor_temporal = 8
                    else:
                        pipe.vae_scale_factor_spatial = 0
                    return pipe
                self.wan_loader.side_effect = build
                with self.assertRaises(ValueError):
                    self.wan_workflow.generate(self.request("wan"))
                self.assertIsNone(self.wan_workflow._pipeline)

    def test_wan_local_pipeline_directory_and_config_replacement(self):
        local = self.wan_path.parent / "pipeline"
        self.write(local / "model_index.json", {})
        self.write(self.wan_path, {**self.wan_model, "weights": "pipeline"})
        self.wan_workflow.generate(self.request("wan"))
        self.assertEqual(self.wan_loader.call_args.args, (str(local),))
        self.assertTrue(self.wan_loader.call_args.kwargs["local_files_only"])
        self.write(local / "model_index.json", {"changed": True})
        self.wan_workflow.generate(self.request("wan"))
        self.assertEqual(self.wan_loader.call_count, 2)

    def test_wan_rejects_single_file_checkpoint(self):
        self.wan_path.with_name("wrong.safetensors").write_bytes(b"weights")
        self.write(self.wan_path, {**self.wan_model, "weights": "wrong.safetensors"})
        with self.assertRaisesRegex(ValueError, "pipeline directory"):
            self.request("wan")

    def test_wan_streams_generation_and_saving_in_order(self):
        events = []
        def build(*args, **kwargs):
            pipe = Pipeline("wan", kwargs["vae"])
            def result(**arguments):
                events.append("generate")
                return pipe.result(**arguments)
            pipe.invoke.side_effect = result
            return pipe
        self.wan_loader.side_effect = build
        def export(frames, path, fps):
            events.append("save")
            path.write_bytes(b"MP4")
        self.exporter.side_effect = export
        outputs = run_generation(self.request("wan", num_videos_per_prompt=2), self.root / "videos")
        self.assertEqual(events, ["generate", "save", "generate", "save"])
        self.assertEqual([output.seed for output in outputs], [42, 43])
        self.assertNotEqual(outputs[0].video_path, outputs[1].video_path)
        for index, output in enumerate(outputs):
            data = json.loads(output.metadata_path.read_text())
            self.assertEqual(data["video_index"], index)
            self.assertEqual(data["frame_count"], 5)
            with Image.open(output.conditioning_path) as image:
                self.assertEqual(image.size, (48, 32))
        with self.assertRaisesRegex(ValueError, "stream"):
            self.wan_workflow.generate(self.request("wan", num_videos_per_prompt=2))

    def test_export_failure_removes_partial_output_and_closes_workflow(self):
        request = self.request("wan")
        def failed_export(frames, path, fps):
            path.write_bytes(b"partial")
            raise RuntimeError("encoder failed")
        self.exporter.side_effect = failed_export
        with patch("image_generator.generation.create_workflow", return_value=self.wan_workflow):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                run_generation(request, self.root / "videos")
        self.assertEqual(list((self.root / "videos").iterdir()), [])
        self.assertIsNone(self.wan_workflow._pipeline)

    def test_flux_png_prefix_and_metadata(self):
        output = run_generation(self.request("flux"), self.root / "images")[0]
        self.assertTrue(output.image_path.name.startswith("flux-seed-42-"))
        self.assertEqual(json.loads(output.metadata_path.read_text())["model"]["workflow"], "flux")

    def test_failed_adapter_load_discards_pipeline_for_both_families(self):
        for family, workflow in (("flux", self.flux_workflow), ("wan", self.wan_workflow)):
            with self.subTest(family=family):
                self.add_lora(family, f"{family}-adapter")
                workflow.generate(self.request(family))
                pipe = self.pipelines[-1]
                pipe.load_lora_weights.side_effect = RuntimeError("bad tensors")
                with self.assertRaisesRegex(RuntimeError, "Could not load LoRA"):
                    workflow.generate(self.request(family, loras=[{"id": f"{family}-adapter", "enabled": True}]))
                self.assertIsNone(workflow._pipeline)
                workflow.generate(self.request(family))

    def test_cuda_unavailable_and_bf16_unsupported_fail_before_model_loading(self):
        self.torch.cuda.is_available.return_value = False
        for family, workflow in (("flux", self.flux_workflow), ("wan", self.wan_workflow)):
            with self.subTest(family=family), self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                workflow.generate(self.request(family))
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.is_bf16_supported.return_value = False
        with self.assertRaisesRegex(RuntimeError, "bfloat16"):
            self.flux_workflow.generate(self.request("flux"))
        self.flux_loader.assert_not_called()
        self.wan_loader.assert_not_called()

    def test_cli_dry_runs_initialize_manifests_without_site_packages_or_generation_outputs(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for family in ("flux", "wan"):
            args = [sys.executable, "-S", str(ROOT / f"scripts/generate_{family}.py"), "--root", str(self.root),
                    "--model-id", f"{family}-model", "--prompt", "a forest", "--seed", "random", "--dry-run"]
            if family == "wan":
                args.extend(["--input-image", str(self.source)])
            result = subprocess.run(args, cwd="/tmp", capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["model"]["workflow"], family)
            self.assertIsNone(data["settings"]["seed"])
        after = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, {path: after[path] for path in before})
        for path in after.keys() - before.keys():
            self.assertIn(path.name, {"model.user.json", "lora.user.json"})
            self.assertEqual(json.loads(after[path]), {"schema_version": 1, "settings": {}})

    def test_cli_generates_both_media_types(self):
        for family in ("flux", "wan"):
            args = ["--root", str(self.root), "--model-id", f"{family}-model", "--prompt", "a forest", "--seed", "99", "--output-dir", str(self.root / family)]
            if family == "wan":
                args.extend(["--input-image", str(self.source), "--no-preserve-aspect-ratio", "--no-vae-tiling"])
            with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                code = run_family(family, args)
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("Seed: 99", stdout.getvalue())
            self.assertIn("Video:" if family == "wan" else "Image:", stdout.getvalue())

    def test_cli_rejects_wrong_family_and_invalid_settings(self):
        for args in (["--model-id", "wan-model"], ["--model-id", "flux-model", "--max-sequence-length", "513"]):
            with redirect_stderr(io.StringIO()):
                code = run_family("flux", ["--root", str(self.root), "--prompt", "test", "--dry-run", *args])
            self.assertEqual(code, 1)
        self.runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()

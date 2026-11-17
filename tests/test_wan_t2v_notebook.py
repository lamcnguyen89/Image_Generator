"""Run the WAN T2V notebook against native pipeline doubles, without CUDA."""

from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import discover_catalog

FAMILY = ROOT / "Models/Model_Families/WAN"
NOTEBOOK = FAMILY / "WAN_T2V_Video_Gen.ipynb"


class WanT2VNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.settings = "".join(self.notebook["cells"][4]["source"])
        self.generation = "".join(self.notebook["cells"][5]["source"])
        self.ns = {}
        exec(compile(self.settings, str(NOTEBOOK), "exec"), self.ns)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ns.update(project_root=self.root, checkpoint_root=self.root / "checkpoints",
                       lora_root=self.root / "loras", prompts=["A mountain lake"],
                       width=64, height=32, num_frames=5, seed=42, show_videos=False)

        class WanPipeline:
            def to(pipe, device):
                return pipe

            def __call__(pipe, **kwargs):
                return pipe.generate(**kwargs)

        self.pipe = WanPipeline()
        for name in ("transformer", "transformer_2", "text_encoder", "vae"):
            setattr(self.pipe, name, SimpleNamespace(
                device="cuda:0", dtype="fp32" if name == "vae" else "bf16",
                config=SimpleNamespace(in_channels=16, out_channels=16, patch_size=(1, 2, 2))))
        self.pipe.vae.enable_tiling, self.pipe.vae.disable_tiling = Mock(), Mock()
        self.pipe.vae_scale_factor_spatial, self.pipe.vae_scale_factor_temporal = 8, 4
        self.pipe.config = SimpleNamespace(boundary_ratio=0.875)
        self.pipe.scheduler = SimpleNamespace(config={"flow_shift": 12.0})
        self.pipe.generate = Mock(side_effect=lambda **kw: SimpleNamespace(frames=[
            [object() for _ in range(kw["num_frames"])]]))
        self.pipe.unload_lora_weights, self.pipe.load_lora_weights, self.pipe.set_adapters = Mock(), Mock(), Mock()
        self.loader = Mock(return_value=self.pipe)
        WanPipeline.from_pretrained = self.loader
        self.vae_loader = Mock(return_value=self.pipe.vae)
        self.export = Mock(side_effect=lambda frames, path, **kw: Path(path).write_bytes(b"synthetic mp4"))
        self.torch = SimpleNamespace(
            __version__="test", bfloat16="bf16", float32="fp32", device=str,
            inference_mode=nullcontext, version=SimpleNamespace(cuda="test"),
            Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed),
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                mem_get_info=lambda device: (2**30, 2**31), get_device_name=lambda device: "test",
                empty_cache=Mock(), synchronize=Mock(), reset_peak_memory_stats=Mock(),
                max_memory_allocated=lambda device: 0))
        self.modules = {
            "torch": self.torch,
            "diffusers": SimpleNamespace(__version__="test", WanPipeline=WanPipeline,
                AutoencoderKLWan=SimpleNamespace(from_pretrained=self.vae_loader)),
            "diffusers.utils": SimpleNamespace(export_to_video=self.export),
            "dotenv": SimpleNamespace(load_dotenv=Mock()),
            "IPython.display": SimpleNamespace(Video=Mock(), display=Mock()),
            "imageio_ffmpeg": SimpleNamespace(get_ffmpeg_exe=Mock(return_value="ffmpeg")),
        }

    def run_generation(self):
        with patch.dict(sys.modules, self.modules), patch.dict("os.environ", {"CACHE_LOCATION": ""}), \
                redirect_stdout(io.StringIO()):
            exec(compile(self.generation, str(NOTEBOOK), "exec"), self.ns)

    def test_code_cells_compile(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                # Package-install magics are parsed by IPython, not Python's compiler.
                source = "".join("pass\n" if line.startswith("%") else line for line in cell["source"])
                compile(source, cell["id"], "exec")

    def test_manifest_is_discoverable_and_matches_notebook_defaults(self):
        folder = self.root / "families/WAN/checkpoints/wan2_2_t2v"
        shutil.copytree(FAMILY / "checkpoints/wan2_2_t2v", folder)
        catalog = discover_catalog(self.root / "families", initialize_missing=False)
        model, = catalog.models.values()
        self.assertEqual(model.defaults["base_model_id"], self.ns["model_id"])
        self.assertEqual(model.defaults["model_variant"], "wan_2_2_t2v_a14b")
        self.assertEqual(model.defaults["supported_modes"], ["txt2vid"])
        self.assertFalse(model.defaults["requirements"]["requires_input_image"])
        self.assertEqual(model.user_overrides["settings"], {})
        defaults = {}
        exec(compile(self.settings, "settings", "exec"), defaults)
        for name in ("width", "height", "num_frames", "fps", "num_inference_steps", "guidance_scale",
                     "guidance_scale_2", "negative_prompt", "num_videos_per_prompt", "seed", "enable_vae_tiling"):
            self.assertEqual(model.defaults["settings"][name], defaults[name], name)
        self.assertFalse(defaults["Use_LoRA"])
        self.assertEqual(model.defaults["settings"]["loras"], [])
        self.assertNotIn("input_image", defaults)

    def test_generation_omits_image_and_exports_text_to_video_metadata(self):
        self.ns["num_videos_per_prompt"] = 2
        self.run_generation()
        self.assertEqual(self.loader.call_args.args, ("Wan-AI/Wan2.2-T2V-A14B-Diffusers",))
        self.assertEqual(self.vae_loader.call_args.kwargs["torch_dtype"], "fp32")
        self.assertEqual(self.pipe.generate.call_count, 2)
        for call in self.pipe.generate.call_args_list:
            self.assertNotIn("image", call.kwargs)
            self.assertEqual((call.kwargs["width"], call.kwargs["height"], call.kwargs["num_frames"]), (64, 32, 5))
        self.assertEqual([call.kwargs["generator"] for call in self.pipe.generate.call_args_list], [42, 43])
        for path in self.ns["saved_paths"]:
            self.assertTrue(path.name.startswith("wan22-t2v-seed-"))
            metadata = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(metadata["mode"], "txt2vid")
            self.assertEqual(metadata["frame_count"], 5)
            self.assertNotIn("input_image", metadata)
            self.assertNotIn("conditioning_image", metadata)
            self.assertFalse(path.with_suffix(".input.png").exists())

    def test_reruns_reuse_pipeline_and_reset_optional_expert_adapters(self):
        self.run_generation()
        self.ns["Use_LoRA"] = True
        for expert, field in (("high", "LoRA_High_Noise_Weights"), ("low", "LoRA_Low_Noise_Weights")):
            path = self.root / f"loras/{expert}.safetensors"
            path.parent.mkdir(exist_ok=True)
            path.touch()
            self.ns[field] = path.name
        self.run_generation()
        self.assertEqual([call.kwargs["load_into_transformer_2"]
                          for call in self.pipe.load_lora_weights.call_args_list], [False, True])
        self.ns["Use_LoRA"] = False
        self.run_generation()
        self.loader.assert_called_once()
        self.assertIsNone(self.pipe._notebook_lora_key)
        self.assertEqual(self.pipe.unload_lora_weights.call_count, 2)

    def test_local_override_requires_t2v_pipeline_before_model_loading(self):
        folder = self.root / "checkpoints/local"
        folder.mkdir(parents=True)
        self.ns["checkpoint_dir"] = "local"
        index = folder / "model_index.json"
        index.write_text(json.dumps({"_class_name": "WanImageToVideoPipeline"}))
        with self.assertRaisesRegex(ValueError, "T2V WanPipeline"):
            self.run_generation()
        self.loader.assert_not_called()
        index.write_text(json.dumps({"_class_name": "WanPipeline"}))
        self.run_generation()
        self.assertEqual(self.loader.call_args.args, (str(folder),))
        self.assertTrue(self.loader.call_args.kwargs["local_files_only"])

    def test_invalid_frame_counts_and_settings_fail_before_loading(self):
        for name, value in (("num_frames", 4), ("width", 48 + 1), ("guidance_scale_2", 1),
                            ("seed", -1), ("fps", 0), ("Use_LoRA", True)):
            with self.subTest(name=name, value=value):
                previous = self.ns[name]
                self.ns[name] = value
                with self.assertRaises(ValueError):
                    self.run_generation()
                self.ns[name] = previous
        self.loader.assert_not_called()

    def test_image_conditioned_experts_and_invalid_runtime_alignment_are_rejected(self):
        self.pipe.transformer.config.in_channels = 36
        with self.assertRaisesRegex(ValueError, "16 input/output channels"):
            self.run_generation()
        self.pipe.transformer.config.in_channels = 16
        self.pipe.vae_scale_factor_temporal = 8
        with self.assertRaisesRegex(ValueError, "temporal alignment"):
            self.run_generation()
        self.export.assert_not_called()

    def test_unexpected_decoded_frame_count_is_not_exported(self):
        self.pipe.generate.side_effect = lambda **kw: SimpleNamespace(frames=[[object()]])
        with self.assertRaisesRegex(RuntimeError, "Expected 5 decoded frames"):
            self.run_generation()
        self.export.assert_not_called()


if __name__ == "__main__":
    unittest.main()

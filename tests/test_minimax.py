"""MiniMax integration using fixture manifests and mocked native ComfyUI nodes."""

from contextlib import nullcontext
from io import BytesIO
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import discover_catalog, initialize_asset
from image_generator.generation import export_video, prepare_generation, run_generation, save_video_batch
from image_generator.workflows.minimax import MiniMaxWorkflow, protect_quantized_transfers


class RootModule:
    def modules(self):
        return []


class MiniMaxFixture:
    """Native node doubles shared by backend and HTTP/worker integration tests."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(ROOT / "tests/fixtures/model_families/MiniMax", self.root / "MiniMax")
        self.model_path = self.root / "MiniMax/checkpoints/demo/model.defaults.json"
        self.model = json.loads(self.model_path.read_text())
        self.model["settings"].update(prompt="A lake", width=64, height=32, num_frames=5, seed=42)
        self.model["components"] = {"vae": {"weights": "../../vae/video.safetensors"},
                                    "text_encoder": {"weights": "../../text_encoders/encoder.safetensors"}}
        self.write(self.model_path, self.model)
        self.checkpoint = self.model_path.parent / self.model["weights"]
        self.header(self.checkpoint, {"__metadata__": {"config": json.dumps({"transformer": {"image_model": "minimax_h3"}})}})
        self.vae = self.root / "MiniMax/vae/video.safetensors"
        self.header(self.vae, {"latents_mean": {}, "decoder.transformer_blocks.0.weight": {}})
        self.encoder = self.root / "MiniMax/text_encoders/encoder.safetensors"
        self.encoder.parent.mkdir()
        self.encoder.write_bytes(b"fake encoder")
        self.adapter = self.root / "MiniMax/loras/demo/missing-fixture.safetensors"
        self.adapter.write_bytes(b"fake adapter")
        self.source = self.root / "source.png"
        Image.new("RGB", (96, 64), "navy").save(self.source)
        self.workflow = MiniMaxWorkflow()
        self.addCleanup(self.workflow.close)
        self.torch = SimpleNamespace(
            __version__="fake-torch", inference_mode=Mock(side_effect=lambda flag: nullcontext()),
            no_grad=nullcontext, from_numpy=lambda array: SimpleNamespace(unsqueeze=lambda axis: np.expand_dims(array, axis)),
            cuda=SimpleNamespace(synchronize=Mock(), empty_cache=Mock(), is_available=Mock(return_value=True)))
        self.base_models = []

        def base_model(*args, **kwargs):
            base = SimpleNamespace(model=RootModule(), patches={})
            self.base_models.append(base)
            return base

        def apply_adapter(model, clip, weights, strength, clip_strength, **kwargs):
            return SimpleNamespace(model=model.model, patches={**model.patches, str(len(model.patches)): [strength]}), None

        node = lambda value: SimpleNamespace(execute=Mock(return_value=(value,)))
        frames = np.zeros((5, 32, 32, 3), dtype=np.float32)
        self.runtime = SimpleNamespace(
            sd=SimpleNamespace(load_diffusion_model=Mock(side_effect=base_model),
                               load_clip=Mock(return_value=SimpleNamespace(cond_stage_model=RootModule())),
                               CLIPType=SimpleNamespace(MINIMAX="minimax"),
                               VAE=Mock(return_value=SimpleNamespace(first_stage_model=RootModule())),
                               load_lora_for_models=Mock(side_effect=apply_adapter)),
            utils=SimpleNamespace(load_torch_file=Mock(return_value=({}, {})), PROGRESS_BAR_HOOK=None),
            management=SimpleNamespace(unload_all_models=Mock()),
            samplers=SimpleNamespace(SAMPLER_NAMES=["res_multistep", "euler"], SCHEDULER_NAMES=["simple"]),
            h3=SimpleNamespace(MiniMaxH3ImageToVideo=SimpleNamespace(execute=Mock(return_value=SimpleNamespace(result=("positive", "latent"))))),
            sampling=SimpleNamespace(BasicGuider=node("guider"), KSamplerSelect=node("sampler"),
                                     BasicScheduler=node("sigmas"), RandomNoise=node("noise"), SamplerCustomAdvanced=node("sampled")),
            nodes=SimpleNamespace(VAEDecodeTiled=Mock(return_value=SimpleNamespace(decode=Mock(return_value=(frames,)))),
                                  VAEDecode=Mock(return_value=SimpleNamespace(decode=Mock(return_value=(frames,))))))
        self.runtime.utils.set_progress_bar_global_hook = Mock(
            side_effect=lambda hook: setattr(self.runtime.utils, "PROGRESS_BAR_HOOK", hook))

        def load_runtime():
            self.workflow._torch, self.workflow._runtime = self.torch, self.runtime

        runtime_patch = patch.object(self.workflow, "_load_runtime", side_effect=load_runtime)
        self.load_runtime = runtime_patch.start()
        self.addCleanup(runtime_patch.stop)
        encoder_patch = patch("image_generator.workflows.minimax.runtime.check_video_encoder")
        self.video_encoder = encoder_patch.start()
        self.addCleanup(encoder_patch.stop)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def header(self, path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document).encode()
        path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"data")

    def request(self, *, mode="img2vid", **overrides):
        return prepare_generation(discover_catalog(self.root), "fixture-minimax", mode=mode, overrides=overrides,
                                  inputs={"input_image": self.source} if mode == "img2vid" else None)


class MiniMaxTests(MiniMaxFixture, unittest.TestCase):
    def test_euler_dispatches_native_sampler_in_both_modes_and_retains_simple(self):
        for mode in ("txt2vid", "img2vid"):
            batch = self.workflow.generate(self.request(mode=mode, sampling={"sampler": "euler", "noise_schedule": "simple"}))
            self.runtime.sampling.KSamplerSelect.execute.assert_called_with(sampler_name="euler")
            self.assertEqual(self.runtime.sampling.BasicScheduler.execute.call_args.kwargs["scheduler"], "simple")
            self.assertEqual(batch.metadata["settings"]["sampling"], {"sampler": "euler", "noise_schedule": "simple"})
        self.workflow.generate(self.request())
        self.runtime.sampling.KSamplerSelect.execute.assert_called_with(sampler_name="res_multistep")
        self.assertEqual(len(self.base_models), 1)

    def test_text_to_video_uses_requested_canvas_without_image_conditioning(self):
        self.source.unlink()
        events = []
        self.workflow.set_progress_callback(events.append)
        batch = self.workflow.generate(self.request(mode="txt2vid"))
        arguments = self.runtime.h3.MiniMaxH3ImageToVideo.execute.call_args.kwargs
        self.assertEqual((arguments["width"], arguments["height"], arguments["length"]), (64, 32, 5))
        self.assertNotIn("first_frame", arguments)
        self.assertNotIn("last_frame", arguments)
        self.assertIsNone(batch.conditioning_image)
        self.assertEqual(batch.metadata["mode"], "txt2vid")
        self.assertEqual(batch.metadata["inputs"], {})
        self.assertNotIn("input_image", batch.metadata)
        self.assertNotIn("conditioning_image", [event.stage for event in events])
        self.assertIn("encoding_prompt", [event.stage for event in events])

    def test_switching_modes_reuses_components_and_rebuilds_conditioning(self):
        for mode in ("img2vid", "txt2vid", "img2vid"):
            self.workflow.generate(self.request(mode=mode))
        self.runtime.sd.load_diffusion_model.assert_called_once()
        self.runtime.sd.load_clip.assert_called_once()
        self.runtime.sd.VAE.assert_called_once()
        calls = self.runtime.h3.MiniMaxH3ImageToVideo.execute.call_args_list
        self.assertEqual(["first_frame" in call.kwargs for call in calls], [True, False, True])

    def test_text_to_video_export_failure_removes_partial_files(self):
        batch = self.workflow.generate(self.request(mode="txt2vid"))
        def fail_export(frames, path, fps):
            path.write_bytes(b"partial video")
            raise OSError("encoder failed")
        directory = self.root / "failed-output"
        with patch("image_generator.generation.export_video", side_effect=fail_export):
            with self.assertRaisesRegex(OSError, "encoder failed"):
                save_video_batch(batch, directory)
        self.assertEqual(list(directory.iterdir()), [])

    def test_modes_reject_missing_extra_and_undeclared_inputs_before_runtime(self):
        catalog = discover_catalog(self.root)
        for mode, inputs in (("img2vid", {}), ("txt2vid", {"input_image": self.source}),
                             ("txt2vid", {"other": self.source}), ("unsupported", {})):
            with self.subTest(mode=mode, inputs=inputs), self.assertRaises(ValueError):
                prepare_generation(catalog, "fixture-minimax", mode=mode, inputs=inputs)
        catalog.get_model("fixture-minimax").defaults["supported_modes"] = ["img2vid"]
        with self.assertRaises(ValueError):
            prepare_generation(catalog, "fixture-minimax", mode="txt2vid")
        self.load_runtime.assert_not_called()

    def test_native_nodes_receive_h3_settings_and_conditioning(self):
        batch = self.workflow.generate(self.request())
        self.runtime.sd.load_diffusion_model.assert_called_once_with(str(self.checkpoint))
        self.runtime.sd.load_clip.assert_called_once_with([str(self.encoder)], clip_type="minimax")
        arguments = self.runtime.h3.MiniMaxH3ImageToVideo.execute.call_args.kwargs
        self.assertEqual((arguments["width"], arguments["height"], arguments["length"]), (32, 32, 5))
        self.assertEqual(arguments["first_frame"].shape, (1, 32, 32, 3))
        self.assertNotIn("negative_prompt", arguments)
        self.runtime.sampling.KSamplerSelect.execute.assert_called_once_with(sampler_name="res_multistep")
        self.assertEqual(self.runtime.sampling.BasicScheduler.execute.call_args.kwargs["scheduler"], "simple")
        self.assertEqual(self.runtime.sampling.BasicScheduler.execute.call_args.kwargs["steps"], 20)
        self.runtime.sampling.RandomNoise.execute.assert_called_once_with(noise_seed=42)
        self.assertEqual(batch.metadata["seeds"], [42])
        self.assertEqual(batch.metadata["component_paths"]["vae"], str(self.vae))
        self.assertEqual(batch.metadata["model"]["components"], self.model["components"])
        self.assertFalse(batch.metadata["audio_exported"])
        self.assertTrue(batch.metadata["runtime"]["quantized_transfer_guard"])
        self.torch.inference_mode.assert_called_with(False)

    def test_sampling_progress_uses_numeric_hook_and_restores_previous_on_success_and_failure(self):
        previous = Mock()
        self.runtime.utils.PROGRESS_BAR_HOOK = previous
        for fail in (False, True):
            events = []
            self.workflow.set_progress_callback(events.append)

            def sample(**kwargs):
                hook = self.runtime.utils.PROGRESS_BAR_HOOK
                self.assertIsNot(hook, previous)
                hook(1, 20, object(), node_id="sampler")
                self.assertEqual((events[-1].stage, events[-1].current, events[-1].total), ("denoising", 1, 20))
                if fail:
                    raise RuntimeError("sampling failed")
                return ("sampled",)

            self.runtime.sampling.SamplerCustomAdvanced.execute.side_effect = sample
            with self.subTest(fail=fail):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                        self.workflow.generate(self.request())
                    self.assertIsNone(self.workflow._pipeline)
                else:
                    self.workflow.generate(self.request())
                self.assertIs(self.runtime.utils.PROGRESS_BAR_HOOK, previous)
        previous.assert_not_called()

    def test_progress_reports_component_and_adapter_reuse_reset_and_loading(self):
        events = []
        self.workflow.set_progress_callback(lambda event: events.append(event.stage))
        self.workflow.generate(self.request())
        for stage in ("loading_runtime", "resolving_text_encoder", "loading_model", "loading_text_encoder",
                      "loading_vae", "resetting_loras", "loading_loras", "loras_ready", "conditioning_image",
                      "encoding_prompt", "video_variation", "denoising", "decoding_video"):
            self.assertIn(stage, events)
        events.clear()
        self.workflow.generate(self.request())
        self.assertIn("reusing_model", events)
        self.assertIn("reusing_loras", events)
        self.assertNotIn("loading_model", events)
        self.assertNotIn("loading_loras", events)
        events.clear()
        self.workflow.generate(self.request(loras=[]))
        self.assertIn("resetting_loras", events)
        self.assertIs(self.workflow._model, self.base_models[0])

    def test_base_reuse_and_adapter_changes_start_from_unpatched_model(self):
        self.workflow.generate(self.request())
        self.workflow.generate(self.request())
        self.assertEqual(self.runtime.sd.load_diffusion_model.call_count, 1)
        self.assertEqual(self.runtime.sd.load_lora_for_models.call_count, 1)
        self.assertEqual(self.base_models[0].patches, {})
        self.workflow.generate(self.request(loras=[]))
        self.assertIs(self.workflow._model, self.base_models[0])
        self.workflow.generate(self.request(loras=[{"id": "fixture-minimax-style", "enabled": True, "strength": 0.2}]))
        self.assertIs(self.runtime.sd.load_lora_for_models.call_args.args[0], self.base_models[0])
        self.assertEqual(self.runtime.sd.load_lora_for_models.call_args.args[3], 0.2)

    def test_checkpoint_replacement_reloads_all_components(self):
        self.workflow.generate(self.request())
        self.checkpoint.write_bytes(self.checkpoint.read_bytes() + b"changed")
        self.workflow.generate(self.request())
        self.assertEqual(self.runtime.sd.load_diffusion_model.call_count, 2)
        self.assertEqual(self.runtime.sd.load_clip.call_count, 2)
        self.runtime.management.unload_all_models.assert_called_once()

    def test_multiple_adapters_stack_and_zero_strength_skips_loading(self):
        path = self.root / "MiniMax/loras/second/lora.defaults.json"
        second = json.loads((self.adapter.parent / "lora.defaults.json").read_text())
        second.update(id="second", weights="second.safetensors")
        self.write(path, second)
        path.with_name("second.safetensors").write_bytes(b"adapter")
        self.workflow.generate(self.request(loras=[
            {"id": "fixture-minimax-style", "enabled": True, "strength": 0.3},
            {"id": "second", "enabled": True, "strength": 0.5},
        ]))
        self.assertEqual(len(self.workflow._model.patches), 2)
        self.assertEqual(self.base_models[0].patches, {})
        self.workflow.generate(self.request(loras=[{"id": "second", "enabled": True, "strength": 0}]))
        self.assertIs(self.workflow._model, self.base_models[0])
        self.assertEqual(self.runtime.sd.load_lora_for_models.call_count, 2)

    def test_unmatched_lora_failed_decoder_and_bad_image_release_or_avoid_runtime(self):
        self.runtime.sd.load_lora_for_models.side_effect = lambda model, *args, **kwargs: (model, None)
        with self.assertRaisesRegex(ValueError, "did not match"):
            self.workflow.generate(self.request())
        self.assertIsNone(self.workflow._pipeline)
        self.runtime.management.unload_all_models.assert_called_once()
        self.runtime.nodes.VAEDecodeTiled.return_value.decode.side_effect = RuntimeError("decode failed")
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            self.workflow.generate(self.request(loras=[]))
        self.assertIsNone(self.workflow._pipeline)
        self.source.write_bytes(b"not an image")
        count = self.load_runtime.call_count
        with self.assertRaises(OSError):
            self.workflow.generate(self.request())
        self.assertEqual(self.load_runtime.call_count, count)

    def test_tiling_and_aspect_ratio_can_be_disabled(self):
        batch = self.workflow.generate(self.request(enable_vae_tiling=False, preserve_aspect_ratio=False))
        self.assertEqual(batch.conditioning_image.size, (64, 32))
        self.runtime.nodes.VAEDecode.return_value.decode.assert_called_once()
        self.runtime.nodes.VAEDecodeTiled.assert_not_called()

    def test_videos_stream_and_save_with_family_prefix_and_actual_seeds(self):
        events = []
        self.runtime.sampling.SamplerCustomAdvanced.execute.side_effect = lambda **kwargs: (events.append("generate") or "sampled",)
        def export(frames, path, fps):
            events.append("save")
            path.write_bytes(b"mp4")
        with patch("image_generator.generation.create_workflow", return_value=self.workflow):
            with patch("image_generator.generation.export_video", side_effect=export):
                outputs = run_generation(self.request(num_videos_per_prompt=2), self.root / "outputs")
        self.assertEqual(events, ["generate", "save", "generate", "save"])
        self.assertEqual([item.seed for item in outputs], [42, 43])
        self.assertTrue(all(item.video_path.name.startswith("minimax-seed-") for item in outputs))
        self.assertEqual(json.loads(outputs[1].metadata_path.read_text())["video_index"], 1)
        self.assertTrue(outputs[0].conditioning_path.is_file())
        self.assertIsNone(self.workflow._pipeline)

    def test_invalid_settings_fail_before_runtime_or_download(self):
        for overrides in ({"num_frames": 49}, {"num_frames": 3600}, {"fps": 16}, {"guidance_scale": 3.5},
                          {"negative_prompt": "blur"}, {"width": 48}, {"seed": -1}, {"num_frames": True},
                          {"sampling": {"sampler": "heun"}}, {"sampling": {"noise_schedule": "default"}},
                          {"sampling": {"sampler": "euler", "noise_schedule": "simple", "extra": True}},
                          {"guidance_scale_2": 1.0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.request(**overrides)
        self.load_runtime.assert_not_called()

    def test_wrong_checkpoint_vae_and_lora_metadata_are_rejected(self):
        self.header(self.checkpoint, {"__metadata__": {"config": "{}"}})
        with self.assertRaisesRegex(ValueError, "image_model"):
            self.request()
        self.header(self.checkpoint, {"__metadata__": {"config": json.dumps({"transformer": {"image_model": "minimax_h3"}})}})
        self.header(self.vae, {"audio": {}})
        with self.assertRaisesRegex(ValueError, "video VAE"):
            self.request()
        self.header(self.vae, {"latents_mean": {}, "decoder.blocks.0": {}})
        path = self.adapter.parent / "lora.defaults.json"
        lora = json.loads(path.read_text())
        lora["compatibility"]["model_variant"] = None
        self.write(path, lora)
        with self.assertRaisesRegex(ValueError, "compatibility"):
            self.request()

    def test_remote_encoder_resolution_honors_cache_offline_and_token(self):
        download = Mock(return_value=str(self.encoder))
        self.workflow.local_files_only = True
        self.workflow._token = "private-token"
        source = {"repo_id": "Comfy-Org/MiniMax-H3", "filename": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"}
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            self.assertEqual(self.workflow._resolve_text_encoder(source), self.encoder)
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        self.assertEqual(download.call_args.kwargs["token"], "private-token")

    def test_cached_encoder_snapshot_symlink_reaches_native_loader(self):
        # HF snapshots retain the format suffix; their blob targets do not.
        cache = self.root / "hub/models--Comfy-Org--MiniMax-H3"
        blob = cache / "blobs/encoder-hash"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"fake encoder")
        filename = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        snapshot = cache / "snapshots/revision" / filename
        snapshot.parent.mkdir(parents=True)
        snapshot.symlink_to(blob)
        self.model["components"]["text_encoder"] = {
            "repo_id": "Comfy-Org/MiniMax-H3", "filename": filename,
        }
        self.write(self.model_path, self.model)
        downloader = Mock(return_value=str(snapshot))
        with patch.dict(sys.modules, {
            "huggingface_hub": SimpleNamespace(hf_hub_download=downloader)
        }):
            self.workflow.generate(self.request())
        self.runtime.sd.load_clip.assert_called_once_with([str(snapshot)], clip_type="minimax")
        self.assertNotEqual(snapshot, snapshot.resolve())

    def test_missing_native_runtime_fails_with_setup_instructions(self):
        workflow = MiniMaxWorkflow(comfyui_dir=self.root / "not-installed")
        with self.assertRaisesRegex(RuntimeError, "native H3 support"):
            workflow._load_runtime()
        with self.assertRaisesRegex(ValueError, "cuda:0"):
            MiniMaxWorkflow(device="cpu")
        with self.assertRaisesRegex(ValueError, "precision"):
            MiniMaxWorkflow(dtype="float16")

    def test_cli_dry_run_is_stdlib_only_and_records_components(self):
        code = f"""
import runpy, sys
sys.argv = [{str(ROOT / 'scripts/generate_minimax.py')!r}, '--root', {str(self.root)!r},
            '--model-id', 'fixture-minimax', '--prompt', 'A lake', '--input-image', {str(self.source)!r}, '--dry-run']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'comfy', 'huggingface_hub'))
"""
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd="/tmp", capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["model"]["components"], self.model["components"])

    def test_initialization_uses_minimax_templates_without_assuming_lora_compatibility(self):
        for folder, kind in (("checkpoints", "model"), ("loras", "lora")):
            directory = self.root / "MiniMax" / folder / "new"
            directory.mkdir()
            (directory / "new.safetensors").write_bytes(b"weights")
            for layer in ("defaults", "user"):
                name = f"{kind}.{layer}.example.json"
                shutil.copyfile(ROOT / "Models/Model_Families/MiniMax" / folder / name, directory.parent / name)
            self.assertEqual(len(initialize_asset(directory)), 2)
            defaults = json.loads((directory / f"{kind}.defaults.json").read_text())
            self.assertEqual(defaults["family"], "minimax")
            if kind == "lora":
                self.assertIsNone(defaults["compatibility"]["model_variant"])

    def test_quantized_transfer_guard_is_local_and_idempotent(self):
        parameter = SimpleNamespace(__tensor_flatten__=lambda: None)
        module = SimpleNamespace(parameters=lambda recurse: [parameter], _apply=Mock(return_value="moved"))
        root = SimpleNamespace(modules=lambda: [module])
        original = module._apply
        protect_quantized_transfers(root, self.torch)
        wrapped = module._apply
        protect_quantized_transfers(root, self.torch)
        self.assertIs(module._apply, wrapped)
        self.assertEqual(module._apply("conversion"), "moved")
        original.assert_called_once_with("conversion", recurse=True)
        self.torch.inference_mode.assert_called_with(False)


class MiniMaxFormTests(unittest.TestCase):
    def setUp(self):
        from image_generator import create_app
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.root)
        self.client = create_app({"TESTING": True, "CATALOG_ROOT": self.root}).test_client()

    def test_form_and_preview_use_h3_controls_without_wan_experts(self):
        from test_generation_ui import FormPage
        response = self.client.get("/generate/fixture-minimax")
        self.assertEqual(response.status_code, 200)
        page = FormPage(response.get_data(as_text=True))
        self.assertEqual(page.controls["num_frames"][0]["step"], "17")
        self.assertEqual(page.controls["width"][0]["step"], "32")
        self.assertIn("readonly", page.controls["guidance_scale"][0])
        self.assertIn("readonly", page.controls["fps"][0])
        self.assertIn("readonly", page.controls["negative_prompt"][0])
        self.assertNotIn("guidance_scale_2", page.controls)
        self.assertNotIn("High-noise adapters", response.get_data(as_text=True))
        data = page.data()
        data["prompt"] = "A lake"
        source = BytesIO()
        Image.new("RGB", (32, 32), "navy").save(source, format="PNG")
        data["input_image"] = (BytesIO(source.getvalue()), "lake.png")
        response = self.client.post("/generate/fixture-minimax", data=data)
        self.assertEqual(response.status_code, 200)
        preview = FormPage(response.get_data(as_text=True)).preview
        self.assertEqual(preview["overrides"]["sampling"], {"sampler": "res_multistep", "noise_schedule": "simple"})
        self.assertEqual(preview["overrides"]["loras"][0]["strength"], 0.7)

    def test_forged_fixed_controls_and_unknown_variant_adapter_are_rejected(self):
        from test_generation_ui import FormPage
        for key, value in (("fps", "16"), ("guidance_scale", "2"), ("negative_prompt", "blur"), ("num_frames", "49")):
            data = FormPage(self.client.get("/generate/fixture-minimax").get_data(as_text=True)).data()
            data[key], data["prompt"] = value, "A lake"
            data["input_image"] = (BytesIO(b"preview"), "lake.png")
            with self.subTest(key=key):
                self.assertEqual(self.client.post("/generate/fixture-minimax", data=data).status_code, 422)
        path = self.root / "MiniMax/loras/demo/lora.defaults.json"
        lora = json.loads(path.read_text())
        lora["compatibility"]["model_variant"] = None
        path.write_text(json.dumps(lora))
        page = FormPage(self.client.get("/generate/fixture-minimax").get_data(as_text=True))
        self.assertIn("disabled", page.controls["lora"][0])


class VideoExportTests(unittest.TestCase):
    def test_frames_stream_without_diffusers_and_convert_float_pixels(self):
        writer = Mock()
        def frames():
            yield np.full((2, 2, 3), 0.5, dtype=np.float32)
            self.assertEqual(writer.append_data.call_count, 1)
            yield Image.new("RGB", (2, 2), "navy")
        from unittest.mock import MagicMock
        context = MagicMock()
        context.__enter__.return_value = writer
        with patch("imageio.v2.get_writer", return_value=context):
            export_video(frames(), Path("preview.mp4"), 24)
        self.assertEqual(writer.append_data.call_count, 2)
        np.testing.assert_array_equal(writer.append_data.call_args_list[0].args[0], np.full((2, 2, 3), 128, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

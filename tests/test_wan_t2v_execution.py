"""WAN T2V and grouped mode selection through the real workflow and worker."""

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import discover_catalog
from image_generator.catalog.validation import CatalogValidationError
from image_generator.generation import prepare_generation
from image_generator.workflows.wan import WanWorkflow
from test_generation_ui import FormPage
from test_wan_execution import VideoPipeline, WanExecutionFixture


class WanT2VExecutionTests(WanExecutionFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.i2v_path = self.catalog / "WAN/checkpoints/demo/model.defaults.json"
        self.i2v = json.loads(self.i2v_path.read_text())
        self.t2v_path = self.catalog / "WAN/checkpoints/t2v/model.defaults.json"
        model = json.loads((ROOT / "Models/Model_Families/WAN/checkpoints/wan2_2_t2v/model.defaults.json").read_text())
        model.update(id="fixture-wan-t2v", notes_file=None)
        model["settings"].update(prompt="A lake at sunrise", width=64, height=32, num_frames=5, seed=42)
        self.group = model["generation_group"]
        self.i2v["generation_group"] = self.group
        self.i2v_path.write_text(json.dumps(self.i2v))
        self.t2v_path.parent.mkdir()
        self.t2v_path.write_text(json.dumps(model))
        self.i2v_user = self.i2v_path.with_name("model.user.json")
        self.i2v_user_bytes = self.i2v_user.read_bytes()
        self.t2v_loader = Mock(side_effect=self.load_t2v)
        self.diffusers.WanPipeline = SimpleNamespace(from_pretrained=self.t2v_loader)

    def load_t2v(self, *args, **options):
        pipe = VideoPipeline(self, options["vae"])
        pipe.transformer.config.in_channels = pipe.transformer.config.out_channels = 16
        pipe.transformer_2.config = SimpleNamespace(in_channels=16, out_channels=16)
        self.pipelines.append(pipe)
        return pipe

    def t2v_data(self, **values):
        data = self.data("fixture-wan-t2v", **values)
        data.pop("input_image", None)
        return data

    def submit_t2v(self, **values):
        response = self.client.post("/generate/fixture-wan-t2v", data=self.t2v_data(**values))
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True))
        return response.location.rsplit("/", 1)[1]

    def prepared_t2v(self, **overrides):
        return prepare_generation(discover_catalog(self.catalog, initialize_missing=False),
                                  "fixture-wan-t2v", mode="txt2vid", overrides=overrides)

    def add_t2v_lora(self, target):
        directory = self.catalog / f"WAN/loras/t2v-{target}"
        directory.mkdir()
        source = self.catalog / "WAN/loras/high/lora.defaults.json"
        document = json.loads(source.read_text())
        document.update(id=f"t2v-{target}", weights="adapter.safetensors")
        document["compatibility"].update(model_variant="wan_2_2_t2v_a14b", expert_target=target,
                                           verified_checkpoint_ids=[])
        document["settings"]["trigger_words"] = []
        (directory / "lora.defaults.json").write_text(json.dumps(document))
        (directory / "adapter.safetensors").write_bytes(b"test adapter")
        return document["id"]

    def make_shared_lora(self, identifier):
        asset = discover_catalog(self.catalog, initialize_missing=False).get_lora(identifier)
        document = json.loads(asset.manifest_path.read_text())
        document["compatibility"]["model_variants"] = ["wan_2_2_i2v_a14b", "wan_2_2_t2v_a14b"]
        asset.manifest_path.write_text(json.dumps(document))
        return asset

    def test_shared_i2v_lora_can_be_enabled_and_loaded_in_both_modes(self):
        identifier = "fixture-wan-high"
        self.make_shared_lora(identifier)
        for model_id in ("fixture-wan", "fixture-wan-t2v"):
            page = FormPage(self.client.get(f"/generate/{model_id}").text)
            checkbox = next(item for item in page.controls["lora"] if item["value"] == identifier)
            self.assertNotIn("disabled", checkbox)
            self.assertEqual(page.controls[f"expert_target:{identifier}"][0]["value"], "high_noise")
        first = self.submit()
        data = self.t2v_data()
        data.setlist("lora", [identifier])
        response = self.client.post("/generate/fixture-wan-t2v", data=data)
        self.assertEqual(response.status_code, 303, response.text)
        second = response.location.rsplit("/", 1)[1]
        self.worker().run(until_idle=True)
        self.assertEqual(self.store.get(first).status, "completed")
        self.assertEqual(self.store.get(second).status, "completed")
        self.pipelines[1].transformer.set_adapters.assert_called_once_with(["adapter_0"], weights=[0.8])
        self.pipelines[1].transformer_2.set_adapters.assert_not_called()

    def test_manual_expert_override_survives_queue_and_drives_loader_and_metadata(self):
        identifier = "fixture-wan-high"
        asset = self.make_shared_lora(identifier)
        data = self.t2v_data()
        data.setlist("lora", [identifier])
        data[f"expert_target:{identifier}"] = "low_noise"
        data[f"strength:{identifier}"] = "0.6"
        response = self.client.post("/generate/fixture-wan-t2v", data=data)
        self.assertEqual(response.status_code, 303, response.text)
        job_id = response.location.rsplit("/", 1)[1]
        queued = self.store.get(job_id)
        self.assertEqual(queued.request["loras"][0]["settings"]["expert_target"], "low_noise")
        self.worker().run(until_idle=True)
        job = self.store.get(job_id)
        self.assertEqual(job.status, "completed", job.error)
        pipe = self.pipelines[0]
        self.assertTrue(pipe.load_lora_weights.call_args.kwargs["load_into_transformer_2"])
        pipe.transformer.set_adapters.assert_not_called()
        pipe.transformer_2.set_adapters.assert_called_once_with(["adapter_0"], weights=[0.6])
        metadata = json.loads(job.outputs[0].metadata_path.read_text())
        self.assertEqual(metadata["loras"][0]["expert_target"], "low_noise")
        self.assertEqual(metadata["loras"][0]["compatibility"]["expert_target"], "high_noise")
        self.assertEqual(json.loads(asset.manifest_path.read_text())["compatibility"]["expert_target"], "high_noise")

    def test_unknown_expert_can_be_chosen_and_saved_separately_for_each_mode(self):
        identifier = "fixture-wan-unknown"
        asset = self.make_shared_lora(identifier)
        for model_id, target in (("fixture-wan", "high_noise"), ("fixture-wan-t2v", "low_noise")):
            page = FormPage(self.client.get(f"/generate/{model_id}").text)
            data = page.data()
            data.setlist("lora", [identifier])
            data[f"expert_target:{identifier}"] = target
            data["action"] = "save_model"
            response = self.client.post(f"/settings/models/{model_id}", data=data)
            self.assertEqual(response.status_code, 303, response.text)
        for model_id, target in (("fixture-wan", "high_noise"), ("fixture-wan-t2v", "low_noise")):
            page = FormPage(self.client.get(f"/generate/{model_id}").text)
            self.assertEqual(page.controls[f"expert_target:{identifier}"][0]["value"], target)
            data = self.data() if model_id == "fixture-wan" else self.t2v_data()
            data[f"expert_target:{identifier}"] = target
            response = self.client.post(f"/generate/{model_id}", data=data)
            self.assertEqual(response.status_code, 303, response.text)
        self.worker().run(until_idle=True)
        self.assertTrue(all(job.status == "completed" for job in self.store.list_jobs()))
        self.assertIsNone(json.loads(asset.manifest_path.read_text())["compatibility"]["expert_target"])
        self.assertFalse(self.pipelines[0].load_lora_weights.call_args.kwargs["load_into_transformer_2"])
        self.assertTrue(self.pipelines[1].load_lora_weights.call_args.kwargs["load_into_transformer_2"])

    def test_missing_or_forged_manual_expert_is_rejected_without_loading_models(self):
        identifier = "fixture-wan-unknown"
        self.make_shared_lora(identifier)
        for target in ("", "both", "invalid"):
            data = self.t2v_data()
            data.setlist("lora", [identifier])
            data[f"expert_target:{identifier}"] = target
            response = self.client.post("/generate/fixture-wan-t2v", data=data)
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("Choose the high-noise or low-noise expert", response.text)
        with self.assertRaises(ValueError):
            self.prepared_t2v(loras=[{"id": identifier, "enabled": True, "expert_target": "both"}])
        self.assertEqual(self.store.list_jobs(), ())
        self.runtime.assert_not_called()

    def test_catalog_has_one_wan_card_and_both_modes_resolve_to_their_checkpoint(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertEqual(html.count('>Wan 2.2</a></h3>'), 1)
        for model_id, mode, image in (("fixture-wan", "img2vid", True),
                                      ("fixture-wan-t2v", "txt2vid", False)):
            response = self.client.get(f"/generate/{model_id}")
            self.assertEqual(response.status_code, 200)
            page = FormPage(response.get_data(as_text=True))
            self.assertEqual([option["value"] for option in page.options["mode"]], ["img2vid", "txt2vid"])
            self.assertEqual(page.controls["mode"][0]["value"], mode)
            self.assertEqual("input_image" in page.controls, image)
            self.assertEqual("preserve_aspect_ratio" in page.controls, image)
            self.assertIn('<h1 id="generation-title">Wan 2.2</h1>', response.text)
        response = self.client.get("/generate/fixture-wan?mode=txt2vid")
        page = FormPage(response.text)
        self.assertEqual(page.controls["model_id"][0]["value"], "fixture-wan-t2v")
        self.assertEqual(page.controls["guidance_scale"][0]["value"], "4.0")
        self.assertEqual(page.controls["guidance_scale_2"][0]["value"], "3.0")
        self.runtime.assert_not_called()

    def test_mode_switch_keeps_prompt_and_loads_target_defaults_and_adapters(self):
        data = self.data(prompt="Keep this prompt", action="change_mode")
        data["mode"] = "txt2vid"
        response = self.client.post("/generate/fixture-wan", data=data)
        self.assertEqual(response.status_code, 200)
        page = FormPage(response.text)
        self.assertEqual(page.controls["prompt"][0]["value"], "Keep this prompt")
        self.assertEqual(page.controls["num_frames"][0]["value"], "5")
        self.assertEqual(page.controls["guidance_scale_2"][0]["value"], "3.0")
        self.assertNotIn("input_image", page.controls)
        self.assertEqual(page.controls["model_id"][0]["value"], "fixture-wan-t2v")
        self.assertFalse(any("checked" in item for item in page.controls["lora"]))
        back = page.data()
        back["mode"], back["action"] = "img2vid", "change_mode"
        response = self.client.post("/generate/fixture-wan-t2v", data=back)
        self.assertEqual(response.status_code, 200)
        page = FormPage(response.text)
        self.assertEqual(page.controls["prompt"][0]["value"], "Keep this prompt")
        self.assertEqual(page.controls["num_frames"][0]["value"], "49")
        self.assertEqual(page.controls["model_id"][0]["value"], "fixture-wan")
        self.assertIn("required", page.controls["input_image"][0])
        self.assertEqual(self.store.list_jobs(), ())
        self.assertEqual(self.staged_files(), [])
        self.runtime.assert_not_called()

    def test_grouping_keeps_custom_checkpoints_separate_and_rejects_ambiguous_modes(self):
        custom = dict(self.i2v)
        custom.pop("generation_group")
        custom.update(id="custom-wan", display_name="Custom WAN checkpoint")
        path = self.catalog / "WAN/checkpoints/custom/model.defaults.json"
        path.parent.mkdir()
        path.write_text(json.dumps(custom))
        html = self.client.get("/").text
        self.assertEqual(html.count('>Wan 2.2</a></h3>'), 1)
        self.assertIn('>Custom WAN checkpoint</a></h3>', html)
        custom["generation_group"] = self.group
        path.write_text(json.dumps(custom))
        response = self.client.get("/generate/fixture-wan")
        self.assertEqual(response.status_code, 422)
        self.assertIn("multiple checkpoints for img2vid", response.text)

    def test_invalid_group_metadata_is_rejected_by_catalog_validation(self):
        model = json.loads(self.t2v_path.read_text())
        for value in ("wan", {"id": "", "display_name": "Wan 2.2"},
                      {"id": "wan", "display_name": None}):
            model["generation_group"] = value
            self.t2v_path.write_text(json.dumps(model))
            with self.subTest(value=value), self.assertRaises(CatalogValidationError):
                discover_catalog(self.catalog, initialize_missing=False)

    def test_t2v_job_exports_real_mp4s_without_source_and_retains_retry_identity(self):
        import imageio.v2 as imageio

        data = self.t2v_data(num_videos_per_prompt="2")
        first = self.client.post("/generate/fixture-wan-t2v", data=data)
        self.assertEqual(first.status_code, 303)
        self.assertEqual(self.client.post("/generate/fixture-wan-t2v", data=data).location, first.location)
        data["seed"] = "100"
        self.assertEqual(self.client.post("/generate/fixture-wan-t2v", data=data).status_code, 409)
        identifier = first.location.rsplit("/", 1)[1]
        self.runtime.assert_not_called()
        self.worker().run(until_idle=True)
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed", result["error"])
        self.assertEqual(result["mode"], "txt2vid")
        self.assertIsNone(result["source_url"])
        self.assertEqual(len(result["outputs"]), 2)
        self.t2v_loader.assert_called_once()
        self.loader.assert_not_called()
        self.assertEqual(self.t2v_loader.call_args.args, ("Wan-AI/Wan2.2-T2V-A14B-Diffusers",))
        self.assertEqual(self.t2v_loader.call_args.kwargs["dtype"], "bf16")
        self.assertEqual(self.t2v_loader.call_args.kwargs["vae"].dtype, "fp32")
        self.assertNotIn("image", self.pipelines[0].invoke.call_args.kwargs)
        for output in result["outputs"]:
            self.assertEqual((output["width"], output["height"], output["frame_count"]), (64, 32, 5))
            self.assertIsNone(output["conditioning_url"])
            self.assertFalse(output["audio_exported"])
            response = self.client.get(output["url"], headers={"Range": "bytes=10-29"})
            self.assertEqual(response.status_code, 206)
            response.close()
        job = self.store.get(identifier)
        self.assertEqual([item.seed for item in job.outputs], [42, 43])
        metadata = json.loads(job.outputs[0].metadata_path.read_text())
        self.assertEqual(metadata["inputs"], {})
        self.assertNotIn("input_image", metadata)
        self.assertIsNone(metadata["conditioning_image"])
        self.assertEqual(list((self.videos / identifier).glob("*.png")), [])
        self.assertNotIn("conditioning_image", [event.stage for event in job.events])
        with imageio.get_reader(str(job.outputs[0].file_path)) as reader:
            self.assertEqual(reader.count_frames(), 5)
            self.assertEqual(reader.get_data(0).shape, (32, 64, 3))
        self.assertEqual(self.staged_files(), [])
        self.assertEqual(self.client.get(f"/jobs/{identifier}/source").status_code, 404)
        html = self.client.get(f"/jobs/{identifier}").text
        self.assertIn("<video", html)
        self.assertNotIn("Uploaded source", html)
        self.assertNotIn("Exact conditioning image", html)

    def test_worker_reuses_t2v_and_switches_pipeline_when_mode_changes(self):
        first = self.submit()
        second = self.submit_t2v()
        third = self.submit_t2v(seed="43")
        fourth = self.submit(seed="44")
        self.worker().run(until_idle=True)
        for identifier in (first, second, third, fourth):
            self.assertEqual(self.store.get(identifier).status, "completed")
        self.assertEqual(self.loader.call_count, 2)
        self.t2v_loader.assert_called_once()
        self.assertEqual(len(self.pipelines), 3)
        self.assertIn("reusing_model", [event.stage for event in self.store.get(third).events])
        self.assertIn("switching_model", [event.stage for event in self.store.get(fourth).events])

    def test_t2v_rejects_images_invalid_settings_and_i2v_adapters_before_runtime(self):
        image = self.t2v_data()
        image["input_image"] = (BytesIO(self.source), "image.png")
        self.assertEqual(self.client.post("/generate/fixture-wan-t2v", data=image).status_code, 422)
        for field, value in (("num_frames", "4"), ("width", "63"), ("fps", "0"),
                              ("guidance_scale_2", "1"), ("lora", "fixture-wan-high")):
            data = self.t2v_data(**{field: value})
            with self.subTest(field=field):
                self.assertEqual(self.client.post("/generate/fixture-wan-t2v", data=data).status_code, 422)
        request = self.prepared_t2v()
        for invalid in (replace(request, inputs={"input_image": Path(__file__)}), replace(request, mode="img2vid")):
            with self.assertRaises(ValueError):
                WanWorkflow().validate(invalid)
        self.assertEqual(self.store.list_jobs(), ())
        self.assertEqual(self.staged_files(), [])
        self.runtime.assert_not_called()

    def test_t2v_adapters_target_their_experts_and_reset_on_the_next_request(self):
        high, low = self.add_t2v_lora("high_noise"), self.add_t2v_lora("low_noise")
        data = self.t2v_data()
        data.setlist("lora", [high, low])
        data[f"strength:{high}"], data[f"strength:{low}"] = "0.4", "0.7"
        self.assertEqual(self.client.post("/generate/fixture-wan-t2v", data=data).status_code, 303)
        cleared = self.submit_t2v(seed="44")
        self.worker().run(until_idle=True)
        pipe = self.pipelines[0]
        pipe.transformer.set_adapters.assert_called_once_with(["adapter_0"], weights=[0.4])
        pipe.transformer_2.set_adapters.assert_called_once_with(["adapter_1"], weights=[0.7])
        pipe.unload_lora_weights.assert_called_once()
        self.assertEqual(self.calls[-1][1], {False: (), True: ()})
        self.assertEqual(self.store.get(cleared).status, "completed")

    def test_local_t2v_pipeline_checks_class_and_reloads_changed_configs(self):
        directory = self.t2v_path.parent / "pipeline"
        directory.mkdir()
        index = directory / "model_index.json"
        index.write_text(json.dumps({"_class_name": "WanImageToVideoPipeline"}))
        model = json.loads(self.t2v_path.read_text())
        model["weights"] = "pipeline"
        self.t2v_path.write_text(json.dumps(model))
        with self.assertRaisesRegex(ValueError, "WanPipeline"):
            self.prepared_t2v()
        self.runtime.assert_not_called()
        workflow = WanWorkflow()
        self.addCleanup(workflow.close)
        for revision in (1, 2):
            index.write_text(json.dumps({"_class_name": "WanPipeline", "revision": revision}))
            workflow.generate(self.prepared_t2v())
        self.assertEqual(self.t2v_loader.call_count, 2)
        self.assertEqual(self.t2v_loader.call_args.args, (str(directory),))
        self.assertTrue(self.t2v_loader.call_args.kwargs["local_files_only"])

    def test_loaded_i2v_experts_and_invalid_t2v_alignment_fail_with_cleanup(self):
        for change in ("channels", "temporal", "spatial"):
            def load(*args, **kwargs):
                pipe = self.load_t2v(*args, **kwargs)
                if change == "channels":
                    pipe.transformer_2.config.in_channels = 36
                elif change == "temporal":
                    pipe.vae_scale_factor_temporal = 8
                else:
                    pipe.vae_scale_factor_spatial = 32
                return pipe
            self.t2v_loader.side_effect = load
            workflow = WanWorkflow()
            with self.subTest(change=change), self.assertRaises(ValueError):
                workflow.generate(self.prepared_t2v())
            self.assertIsNone(workflow._pipeline)

    def test_later_t2v_failure_preserves_first_output_and_recovers_for_i2v(self):
        failed = self.submit_t2v(num_videos_per_prompt="2")
        following = self.submit()
        self.fail_on_call = 2
        self.worker().run(until_idle=True)
        job = self.store.get(failed)
        self.assertEqual(job.status, "failed")
        self.assertEqual(len(job.outputs), 1)
        self.assertIsNone(job.outputs[0].conditioning_path)
        self.assertTrue(job.outputs[0].file_path.is_file())
        self.assertEqual(self.store.get(following).status, "completed")

    def test_saving_t2v_preferences_does_not_change_i2v_preferences(self):
        page = FormPage(self.client.get("/generate/fixture-wan?mode=txt2vid").text)
        data = page.data()
        data["action"], data["prompt"], data["num_frames"] = "save_model", "A new T2V default", "9"
        response = self.client.post("/settings/models/fixture-wan-t2v", data=data)
        self.assertEqual(response.status_code, 303, response.text)
        self.assertEqual(self.i2v_user.read_bytes(), self.i2v_user_bytes)
        saved = json.loads(self.t2v_path.with_name("model.user.json").read_text())
        self.assertEqual(saved["settings"]["num_frames"], 9)
        self.assertEqual(saved["settings"]["prompt"], "A new T2V default")

    def test_t2v_contract_and_request_preparation_do_not_import_gpu_libraries(self):
        code = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation
catalog = discover_catalog({str(self.catalog)!r}, initialize_missing=False)
request = prepare_generation(catalog, 'fixture-wan-t2v', mode='txt2vid')
assert not request.inputs
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'flask', 'transformers'))
"""
        result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

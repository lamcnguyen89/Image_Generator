"""Render and submit fixture forms without weights, GPU libraries, or jobs."""

from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from werkzeug.datastructures import MultiDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app


class FormPage(HTMLParser):
    """Inspect actual controls and submit the successful values a browser would."""
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.controls, self.options, self.preview_text = {}, {}, []
        self.active, self.select, self.option, self.in_preview = None, None, None, False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"input", "textarea", "select"} and "name" in attrs:
            self.controls.setdefault(attrs["name"], []).append(attrs)
            if tag == "textarea":
                self.active = attrs
                attrs["value"] = ""
            elif tag == "select":
                self.select = attrs
                self.options[attrs["name"]] = []
        elif tag == "option" and self.select is not None:
            self.options[self.select["name"]].append(attrs)
            if "selected" in attrs or "value" not in self.select:
                self.select["value"] = attrs.get("value", "")
        elif tag == "pre" and attrs.get("id") == "request-preview":
            self.in_preview = True

    def handle_data(self, data):
        if self.active is not None:
            self.active["value"] += data
        if self.in_preview:
            self.preview_text.append(data)

    def handle_endtag(self, tag):
        if tag == "textarea":
            self.active = None
        elif tag == "select":
            self.select = None
        elif tag == "pre":
            self.in_preview = False

    def data(self):
        form = MultiDict()
        for name, controls in self.controls.items():
            for control in controls:
                if "disabled" in control or control.get("type") == "file":
                    continue
                if control.get("type") == "checkbox" and "checked" not in control:
                    continue
                form.add(name, control.get("value", "on" if control.get("type") == "checkbox" else ""))
        return form

    @property
    def preview(self):
        return json.loads("".join(self.preview_text)) if self.preview_text else None


class GenerationUITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "model_families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.root)
        self.app = create_app({"TESTING": True, "CATALOG_ROOT": self.root})
        self.client = self.app.test_client()

    def page(self, family):
        response = self.client.get(f"/generate/fixture-{family}")
        self.assertEqual(response.status_code, 200)
        return FormPage(response.get_data(as_text=True))

    def data(self, family):
        data = self.page(family).data()
        data["prompt"] = "A mountain lake at sunrise"
        if family == "wan":
            from PIL import Image

            source = BytesIO()
            Image.new("RGB", (32, 32), "navy").save(source, format="PNG")
            source.seek(0)
            data["input_image"] = (source, "lake.png")
        return data

    def post(self, family, data):
        response = self.client.post(f"/generate/fixture-{family}", data=data)
        return response, FormPage(response.get_data(as_text=True))

    def test_all_three_forms_render_family_controls_and_shared_navigation(self):
        fields = {}
        for family in ("sdxl", "flux", "wan"):
            page = self.page(family)
            fields[family] = page.controls
            for name in ("prompt", "negative_prompt", "seed", "sampling.sampler", "include_trigger_words"):
                self.assertIn(name, page.controls)
            html = self.client.get(f"/generate/fixture-{family}").get_data(as_text=True)
            self.assertIn("Preview request", html)
            self.assertIn(f'href="/models/fixture-{family}"', html)
        self.assertIn("clip_skip", fields["sdxl"])
        self.assertNotIn("true_cfg_scale", fields["sdxl"])
        self.assertIn("true_cfg_scale", fields["flux"])
        self.assertNotIn("input_image", fields["flux"])
        self.assertIn("input_image", fields["wan"])
        self.assertIn("guidance_scale_2", fields["wan"])
        self.assertNotIn("num_images_per_prompt", fields["wan"])

    def test_resolved_values_nested_settings_nulls_and_checkpoint_constraints(self):
        page = self.page("sdxl")
        self.assertEqual(page.controls["width"][0]["value"], "768")
        self.assertEqual(page.controls["seed"][0]["value"], "42")
        self.assertEqual(page.controls["clip_skip"][0]["value"], "")
        self.assertEqual(page.controls["sampling.sampler"][0]["value"], "euler_ancestral")
        self.assertEqual(page.controls["width"][0]["step"], "16")
        self.assertEqual([option["value"] for option in page.options["sampling.sampler"]], ["euler", "euler_ancestral"])
        self.assertEqual(self.page("flux").controls["max_sequence_length"][0]["max"], "256")
        self.assertEqual(self.page("wan").controls["num_frames"][0]["value"], "49")

    def test_successful_previews_for_all_families_do_not_mutate_files(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for family in ("sdxl", "flux", "wan"):
            with self.subTest(family=family):
                response, page = self.post(family, self.data(family))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(page.preview["model_id"], f"fixture-{family}")
                self.assertEqual(page.preview["overrides"]["prompt"], "A mountain lake at sunrise")
                self.assertIsInstance(page.preview["overrides"]["width"], int)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_multiple_loras_strength_precedence_and_opt_in_triggers(self):
        data = self.data("sdxl")
        self.assertEqual(data["strength:fixture-sdxl-style"], "0.65")
        data.add("lora", "fixture-sdxl-detail")
        data["strength:fixture-sdxl-detail"] = "0.9"
        data["include_trigger_words"] = "on"
        response, page = self.post("sdxl", data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(page.preview["overrides"]["loras"], [
            {"id": "fixture-sdxl-style", "enabled": True, "strength": 0.65},
            {"id": "fixture-sdxl-detail", "enabled": True, "strength": 0.9},
        ])
        self.assertTrue(page.preview["include_trigger_words"])
        self.assertEqual(page.preview["overrides"]["num_inference_steps"], 24)

    def test_nullable_values_booleans_and_large_seeds_keep_json_types(self):
        data = self.data("sdxl")
        data["seed"] = str(2**63 - 1)
        response, page = self.post("sdxl", data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(page.preview["overrides"]["seed"], 2**63 - 1)
        self.assertIsNone(page.preview["overrides"]["clip_skip"])
        self.assertEqual(page.controls["seed"][0]["type"], "text")
        data = self.data("wan")
        data["seed"] = ""
        del data["preserve_aspect_ratio"]
        del data["enable_vae_tiling"]
        response, page = self.post("wan", data)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(page.preview["overrides"]["seed"])
        self.assertIs(page.preview["overrides"]["preserve_aspect_ratio"], False)
        self.assertIs(page.preview["overrides"]["enable_vae_tiling"], False)
        self.assertEqual(page.preview["input_files"], {"input_image": "lake.png"})

    def test_invalid_fields_are_rejected_and_user_values_are_preserved(self):
        for family, name, value in (
            ("sdxl", "prompt", ""), ("sdxl", "width", "770"),
            ("sdxl", "num_inference_steps", "3.5"), ("sdxl", "seed", str(2**63)),
            ("sdxl", "sampling.sampler", "unsupported"), ("flux", "true_cfg_scale", "0.5"),
            ("flux", "max_sequence_length", "257"), ("flux", "guidance_scale", "nan"),
            ("wan", "fps", "0"), ("wan", "num_frames", "50"),
            ("wan", "guidance_scale_2", "0.5"),
        ):
            with self.subTest(family=family, name=name):
                data = self.data(family)
                data[name] = value
                response, page = self.post(family, data)
                self.assertEqual(response.status_code, 422)
                self.assertIsNone(page.preview)
                self.assertIn("Check these values", response.get_data(as_text=True))
                self.assertEqual(page.controls["prompt"][0]["value"], data["prompt"])

    def test_forged_unknown_duplicate_and_unavailable_loras_are_rejected(self):
        for family, identifier in (("sdxl", "missing"), ("sdxl", "fixture-sdxl-style"),
                                   ("wan", "fixture-wan-unknown"), ("sdxl", "fixture-flux-style")):
            with self.subTest(family=family, identifier=identifier):
                data = self.data(family)
                data.add("lora", identifier)
                response, page = self.post(family, data)
                self.assertEqual(response.status_code, 422)
                self.assertIsNone(page.preview)
        data = self.data("sdxl")
        data["strength:fixture-sdxl-style"] = "-1"
        self.assertEqual(self.post("sdxl", data)[0].status_code, 422)

    def test_wan_groups_adapters_and_requires_source_image(self):
        html = self.client.get("/generate/fixture-wan").get_data(as_text=True)
        for label in ("High-noise adapters", "Low-noise adapters", "Unavailable or unconfigured adapters"):
            self.assertIn(label, html)
        page = FormPage(html)
        unknown = next(control for control in page.controls["lora"] if control["value"] == "fixture-wan-unknown")
        self.assertNotIn("disabled", unknown)
        self.assertEqual(page.controls["expert_target:fixture-wan-unknown"][0]["value"], "")
        data = self.data("wan")
        del data["input_image"]
        response, _ = self.post("wan", data)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Select a source image.", response.get_data(as_text=True))

    def test_model_switching_uses_the_selected_models_own_values(self):
        data = self.data("sdxl")
        data["width"] = "512"
        self.assertEqual(self.post("sdxl", data)[0].status_code, 200)
        self.assertEqual(self.page("flux").controls["width"][0]["value"], "1024")
        self.assertEqual(self.page("sdxl").controls["width"][0]["value"], "768")

    def test_unknown_model_and_unsupported_modes_and_workflows(self):
        self.assertEqual(self.client.get("/generate/missing").status_code, 404)
        self.assertEqual(self.client.get("/generate/fixture-sdxl-style").status_code, 404)
        self.assertEqual(self.client.get("/generate/fixture-sdxl?mode=img2img").status_code, 400)
        data = self.data("sdxl")
        data["mode"] = "img2img"
        self.assertEqual(self.post("sdxl", data)[0].status_code, 400)
        path = self.root / "SDXL/checkpoints/demo/model.defaults.json"
        model = json.loads(path.read_text())
        model["workflow"] = "future-workflow"
        path.write_text(json.dumps(model))
        self.assertEqual(self.client.get("/generate/fixture-sdxl").status_code, 422)

    def test_html_is_escaped_and_extra_metadata_cannot_override_identity(self):
        data = self.data("sdxl")
        data["prompt"] = '<script>alert("unsafe")</script>'
        data["weights"] = "unexpected.safetensors"
        data["model_id"] = "fixture-flux"
        response, page = self.post("sdxl", data)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('<script>alert("unsafe")</script>', response.get_data(as_text=True))
        self.assertEqual(page.preview["model_id"], "fixture-sdxl")
        self.assertNotIn("weights", page.preview["overrides"])
        self.assertEqual(page.preview["overrides"]["prompt"], data["prompt"])

    def test_previews_never_call_full_asset_validation_or_generate(self):
        for family, module, class_name in (("sdxl", "sdxl", "SDXLWorkflow"), ("flux", "flux", "FluxWorkflow"), ("wan", "wan", "WanWorkflow")):
            with self.subTest(family=family):
                prefix = f"image_generator.workflows.{module}.{class_name}"
                with patch(f"{prefix}.validate", side_effect=AssertionError("must not check absent weights")):
                    with patch(f"{prefix}.generate", side_effect=AssertionError("must not generate")):
                        self.assertEqual(self.post(family, self.data(family))[0].status_code, 200)

    def test_karras_choices_and_preview_validate_exact_sampling_pairs(self):
        manifest = self.root / "SDXL/checkpoints/demo/model.defaults.json"
        document = json.loads(manifest.read_text())
        document["requirements"]["allowed_sampling"].extend([
            {"sampler": "dpmpp_2m_sde", "noise_schedule": "default"},
            {"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"},
        ])
        manifest.write_text(json.dumps(document))
        page = self.page("sdxl")
        control = page.controls["sampling.noise_schedule"][0]
        self.assertEqual(control["data-choice-source"], "sampling.sampler")
        self.assertEqual(json.loads(control["data-choices-by-source"]), {
            "euler": ["default"], "euler_ancestral": ["default"], "dpmpp_2m_sde": ["default", "karras"],
        })
        karras = next(option for option in page.options["sampling.noise_schedule"] if option["value"] == "karras")
        self.assertIn("disabled", karras)
        data = self.data("sdxl")
        for name, value in {"action": "preview", "sampling.sampler": "euler", "sampling.noise_schedule": "karras"}.items():
            data[name] = value
        response = self.client.post("/generate/fixture-sdxl", data=data)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Choose a noise schedule supported by the selected sampler", response.text)
        data["sampling.sampler"] = "dpmpp_2m_sde"
        response = self.client.post("/generate/fixture-sdxl", data=data)
        self.assertEqual(response.status_code, 200)
        page = FormPage(response.text)
        self.assertEqual(page.preview["overrides"]["sampling"], {"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"})
        selected = next(option for option in page.options["sampling.noise_schedule"] if "selected" in option)
        self.assertEqual(selected["value"], "karras")
        self.assertNotIn("disabled", selected)

    def test_navigation_static_assets_and_repository_models(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('href="/generate/fixture-sdxl"', html)
        for asset, mimetype in (("generation.css", "text/css"), ("generation.js", "text/javascript")):
            response = self.client.get(f"/catalog-assets/{asset}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, mimetype)
            response.close()
        app = create_app({"TESTING": True, "CATALOG_ROOT": ROOT / "Models/Model_Families"})
        from image_generator.catalog import discover_catalog
        from image_generator.workflows.registry import WORKFLOWS
        with app.test_client() as client:
            for model in discover_catalog(app.config["CATALOG_ROOT"]).models.values():
                with self.subTest(model=model.id):
                    response = client.get(f"/generate/{model.id}")
                    if model.defaults["workflow"] in WORKFLOWS:
                        self.assertEqual(response.status_code, 200)
                    else:
                        self.assertEqual(response.status_code, 422)
                        self.assertIn("Generation form unavailable", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()

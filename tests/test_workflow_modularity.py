"""Exercise a new integration through real services without editing core code."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/fixtures/workflows"))

from test_generation_ui import FormPage

from image_generator import create_app
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation
from image_generator.job_requests import decode_request
from image_generator.jobs import JobStore
from image_generator.worker import Worker
from image_generator.workflows.adapters import AdapterPolicy
from image_generator.workflows.parameters import Parameter
from image_generator.workflows.registry import WORKFLOWS, get_definition


class WorkflowModularityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.package = self.root / "demo_pipeline"
        shutil.copytree(ROOT / "tests/fixtures/workflows/demo_pipeline", self.package)
        self.registration = patch.dict(WORKFLOWS, {"demo_pipeline": ("demo_pipeline.definition", "DEFINITION")})
        self.registration.start()
        self.addCleanup(self.registration.stop)
        self.catalog = self.root / "models"
        self.model_dir = self.catalog / "SDXL/checkpoints/demo"
        self.model_dir.mkdir(parents=True)
        self.manifest = self.model_dir / "model.defaults.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": "demo-model",
                    "display_name": "Custom SDXL model",
                    "family": "sdxl",
                    "workflow": "demo_pipeline",
                    "weights": None,
                    "base_model_id": "test/no-download",
                    "supported_modes": ["reference_image"],
                    "requirements": {},
                    "settings": {},
                }
            )
        )
        (self.model_dir / "model.user.json").write_text('{"schema_version":1,"settings":{}}')
        self.app = create_app(
            {
                "TESTING": True,
                "ACCESS_PASSWORD": "",
                "SECRET_KEY": "modularity-tests",
                "CATALOG_ROOT": self.catalog,
                "JOB_DATABASE": self.root / "jobs.sqlite3",
                "JOB_OUTPUT_ROOT": self.root / "Images",
                "JOB_VIDEO_OUTPUT_ROOT": self.root / "Videos",
                "JOB_INPUT_ROOT": self.root / "inputs",
            }
        )
        self.client = self.app.test_client()

    def form(self):
        page = self.client.get("/generate/demo-model")
        self.assertEqual(page.status_code, 200, page.text)
        return FormPage(page.text).data()

    def upload(self, color):
        stream = BytesIO()
        Image.new("RGB", (8, 8), color).save(stream, "PNG")
        stream.seek(0)
        return stream, "image.png"

    def submission(self, action="generate", **settings):
        data = self.form()
        for name, value in {"prompt": "A custom pipeline", "action": action, **settings}.items():
            data[name] = value
        data["input_image"] = self.upload("white")
        data["mask_image"] = self.upload("black")
        return data

    def test_custom_family_workflow_and_mode_render_package_assets(self):
        page = self.client.get("/generate/demo-model")
        self.assertIn("Custom temperature", page.text)
        self.assertIn("custom-workflow-hint", page.text)
        for filename in ("controls.js", "controls.css"):
            url = f"/workflows/demo_pipeline/static/{filename}"
            self.assertIn(url, page.text)
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            response.close()
        self.assertEqual(self.client.get("/workflows/demo_pipeline/static/../definition.py").status_code, 404)
        self.assertEqual(self.client.get("/workflows/unregistered/static/controls.js").status_code, 404)

    def test_multiple_inputs_queue_snapshot_execution_and_downloads(self):
        response = self.client.post("/generate/demo-model", data=self.submission())
        self.assertEqual(response.status_code, 303, response.text)
        identifier = response.location.rsplit("/", 1)[1]
        store = JobStore(self.app.config["JOB_DATABASE"])
        snapshot = store.get(identifier).request
        self.assertEqual(set(snapshot["inputs"]), {"input_image", "mask_image"})
        worker = Worker(store, self.root / "Images", video_output_dir=self.root / "Videos")
        worker.run(until_idle=True)
        payload = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(payload["status"], "completed", payload["error"])
        self.assertEqual(payload["mode"], "reference_image")
        self.assertEqual(set(payload["sources"]), {"input_image", "mask_image"})
        for url in [*payload["sources"].values(), payload["outputs"][0]["url"]]:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            response.close()
        self.assertEqual(self.client.get(f"/jobs/{identifier}/inputs/unregistered").status_code, 404)
        self.assertEqual(store.get(identifier).request, snapshot)

    def test_preview_and_settings_use_schema_without_importing_runtime(self):
        sys.modules.pop("demo_pipeline.runtime", None)
        response = self.client.post("/generate/demo-model", data=self.submission("preview", temperature="3"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("demo_pipeline.runtime", sys.modules)
        self.assertFalse((self.root / "inputs").exists())
        data = self.form()
        data["action"] = "save_model"
        data["temperature"] = "3"
        response = self.client.post("/settings/models/demo-model", data=data)
        self.assertEqual(response.status_code, 303, response.text)
        self.assertEqual(json.loads((self.model_dir / "model.user.json").read_text())["settings"]["temperature"], 3)
        self.assertNotIn("demo_pipeline.runtime", sys.modules)

    def test_input_and_asset_failures_leave_no_files_or_jobs(self):
        for change in ({"temperature": "4"}, {"temperature": "99"}):
            response = self.client.post("/generate/demo-model", data=self.submission(**change))
            self.assertEqual(response.status_code, 422)
            root = self.root / "inputs"
            self.assertFalse(root.exists() and any(p.is_file() for p in root.glob("*/*") if p.parent.name != ".locks"))
        data = self.submission()
        del data["mask_image"]
        self.assertEqual(self.client.post("/generate/demo-model", data=data).status_code, 422)
        self.assertEqual(JobStore(self.app.config["JOB_DATABASE"]).list_jobs(), ())

    def test_enqueue_failure_rolls_back_all_named_uploads(self):
        with patch("image_generator.jobs.JobStore.enqueue", side_effect=OSError("Synthetic enqueue failure")):
            response = self.client.post("/generate/demo-model", data=self.submission())
        self.assertEqual(response.status_code, 422)
        self.assertEqual(list((self.root / "inputs").glob("*/source.png")), [])
        self.assertEqual(list((self.root / "inputs").glob("*/input-mask_image.png")), [])

    def test_cross_field_validation_returns_form_errors(self):
        data = self.submission(temperature="5")
        del data["use_filter"]
        response = self.client.post("/generate/demo-model", data=data)
        self.assertEqual(response.status_code, 422)
        self.assertIn("High temperature requires the filter", response.text)

    def test_derived_workflow_does_not_change_standard_family_definition(self):
        standard = get_definition("sdxl")
        custom = get_definition("demo_pipeline")
        self.assertEqual(custom.family, standard.family)
        self.assertNotIn("temperature", standard.defaults)
        runtime = custom.create_runtime()
        runtime.defaults["temperature"] = 99
        self.assertEqual(custom.defaults["temperature"], 2)

    def test_derived_builtin_runtime_uses_custom_definition_without_changing_family(self):
        shutil.copytree(ROOT / "tests/fixtures/model_families/FLUX", self.catalog / "FLUX")
        path = self.catalog / "FLUX/checkpoints/demo/model.defaults.json"
        document = json.loads(path.read_text())
        document["workflow"] = "custom_flux"
        path.write_text(json.dumps(document))
        standard = get_definition("flux")

        def parameters(model, mode):
            ui = standard.parameters(model, mode)
            return replace(
                ui,
                fields=ui.fields
                + (Parameter("custom_strength", "Custom strength", "number", "Advanced", minimum=0, maximum=1),),
            )

        custom = replace(
            standard, name="custom_flux", defaults={**standard.defaults, "custom_strength": 0.25}, parameters=parameters
        )
        with patch.dict(WORKFLOWS, {"custom_flux": custom}):
            catalog = discover_catalog(self.catalog, initialize_missing=False)
            model = catalog.get_model("fixture-flux")
            model.asset_paths["weights"].write_bytes(b"synthetic checkpoint")
            request = prepare_generation(catalog, model.id, overrides={"prompt": "Custom model", "loras": []})
            runtime = custom.create_runtime()
            runtime.validate(request)
            self.assertEqual(runtime.get_ui_definition(model, "txt2img").workflow, "custom_flux")
            self.assertIsNone(runtime._torch)
            self.assertEqual(runtime.workflow_id, "custom_flux")
        self.assertNotIn("custom_strength", standard.defaults)

    def test_sdxl_pipeline_construction_can_be_overridden_without_replacing_cache_logic(self):
        from types import SimpleNamespace

        from image_generator.workflows.sdxl import SDXLWorkflow

        class CustomRuntime(SDXLWorkflow):
            builds = 0

            def _build_pipeline(self, request):
                self.builds += 1
                pipeline = SimpleNamespace(scheduler=SimpleNamespace(config={}), custom_component=True)
                pipeline.to = lambda device: pipeline
                return pipeline

        model = discover_catalog(self.catalog, initialize_missing=False).get_model("demo-model")
        runtime = CustomRuntime(device="cpu")
        request = SimpleNamespace(model=model)
        self.assertFalse(runtime._load_pipeline(request))
        self.assertTrue(runtime._pipeline.custom_component)
        self.assertTrue(runtime._load_pipeline(request))
        self.assertEqual(runtime.builds, 1)
        runtime.close()

    def test_workflow_declares_typed_adapter_link_controls(self):
        directory = self.catalog / "SDXL/loras/custom"
        directory.mkdir(parents=True)
        (directory / "adapter.safetensors").write_bytes(b"synthetic adapter weights")
        (directory / "lora.defaults.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": "custom-adapter",
                    "display_name": "Custom adapter",
                    "family": "sdxl",
                    "adapter_type": "lora",
                    "weights": "adapter.safetensors",
                    "compatibility": {},
                    "settings": {"strength": 0.8, "component_scale": 1.0},
                }
            )
        )
        policy = AdapterPolicy(
            link_fields=(Parameter("component_scale", "Component scale", "number", "Adapters", minimum=0, maximum=2),)
        )
        definition = replace(get_definition("demo_pipeline"), adapter_policy=policy)
        with patch.dict(WORKFLOWS, {"demo_pipeline": definition}):
            data = self.submission()
            data["lora"] = "custom-adapter"
            data["component_scale:custom-adapter"] = "0.25"
            response = self.client.post("/generate/demo-model", data=data)
        self.assertEqual(response.status_code, 303, response.text)
        identifier = response.location.rsplit("/", 1)[1]
        snapshot = JobStore(self.app.config["JOB_DATABASE"]).get(identifier).request
        self.assertEqual(snapshot["settings"]["loras"][0]["component_scale"], 0.25)
        self.assertEqual(snapshot["loras"][0]["settings"]["component_scale"], 0.25)
        request = decode_request(snapshot)
        invalid = replace(request, loras=(replace(request.loras[0], settings={"component_scale": 9}),))
        with self.assertRaisesRegex(ValueError, "component_scale"):
            definition.validate(invalid)

    def test_definition_only_change_can_enable_preview_only_mode(self):
        definition = get_definition("demo_pipeline")
        disabled = replace(definition, modes=(replace(definition.modes[0], executable=False),))
        with patch.dict(WORKFLOWS, {"demo_pipeline": disabled}):
            page = self.client.get("/generate/demo-model")
            self.assertNotIn('value="generate"', page.text)
            self.assertEqual(self.client.post("/generate/demo-model", data=self.submission()).status_code, 400)

    def test_cli_uses_custom_parameters_aliases_inputs_and_boolean_switches(self):
        source = self.root / "source.png"
        Image.new("RGB", (8, 8)).save(source)
        code = """
from image_generator.workflows.registry import WORKFLOWS
WORKFLOWS['demo_pipeline'] = ('demo_pipeline.definition', 'DEFINITION')
from image_generator.cli import run_family
raise SystemExit(run_family('demo_pipeline'))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                code,
                "--root",
                str(self.catalog),
                "--prompt",
                "A custom pipeline",
                "--heat",
                "3",
                "--no-use-filter",
                "--input-image",
                str(source),
                "--mask-image",
                str(source),
                "--dry-run",
            ],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests/fixtures/workflows")]),
            },
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = json.loads(result.stdout)
        self.assertEqual(snapshot["settings"]["temperature"], 3)
        self.assertFalse(snapshot["settings"]["use_filter"])
        self.assertEqual(set(snapshot["inputs"]), {"input_image", "mask_image"})

    def test_generic_cli_text_to_video_requires_no_source_image(self):
        directory = self.catalog / "WAN/checkpoints/text_to_video"
        directory.mkdir(parents=True)
        defaults = json.loads(
            (ROOT / "Models/Model_Families/WAN/checkpoints/wan2_2_t2v/model.defaults.json").read_text()
        )
        defaults["id"] = "cli-wan-t2v"
        (directory / "model.defaults.json").write_text(json.dumps(defaults))
        (directory / "model.user.json").write_text('{"schema_version":1,"settings":{}}')
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "image_generator.cli",
                "wan",
                "--root",
                str(self.catalog),
                "--model-id",
                "cli-wan-t2v",
                "--prompt",
                "Clouds move over mountains",
                "--dry-run",
            ],
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = json.loads(result.stdout)
        self.assertEqual(snapshot["mode"], "txt2vid")
        self.assertEqual(snapshot["inputs"], {})

    def test_builtin_definitions_do_not_import_execution_modules(self):
        code = """
import sys
from image_generator.workflows.registry import WORKFLOWS, get_definition
for name in WORKFLOWS:
    definition = get_definition(name)
    assert definition.modes
assert not any(name.endswith('.runtime') and '.workflows.' in name and name.count('.') > 2 for name in sys.modules)
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'transformers', 'flask'))
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", code],
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

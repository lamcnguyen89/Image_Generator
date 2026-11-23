"""Exercise new image sampling choices through forms, saves and queued requests."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.jobs import JobStore
from image_generator.workflows.shared.samplers import available_samplers
from image_generator.workflows.shared.schedules import available_schedules
from test_generation_ui import FormPage


class ImageSamplerFormTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.catalog)
        for manifest in self.catalog.rglob("model.defaults.json"):
            document = json.loads(manifest.read_text())
            workflow = document["workflow"]
            if workflow not in ("sdxl", "flux", "qwen_2_1", "krea2"):
                continue
            document["requirements"]["allowed_sampling"] = [
                {"sampler": spec.name, "noise_schedule": schedule.name}
                for spec in available_samplers(workflow, {})
                for schedule in available_schedules(spec.schedule_backend)
            ]
            manifest.write_text(json.dumps(document))
            weights = document.get("weights")
            if weights:
                (manifest.parent / weights).write_bytes(b"synthetic image checkpoint")
        self.database = self.root / "jobs.sqlite3"
        self.app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": self.catalog,
                               "JOB_DATABASE": self.database, "JOB_OUTPUT_ROOT": self.root / "images"})
        self.client = self.app.test_client()

    def test_each_image_sampler_previews_saves_and_queues_exact_pair(self):
        for workflow in ("sdxl", "flux", "qwen_2_1", "krea2"):
            identifier = "fixture-" + workflow
            for spec in available_samplers(workflow, {}):
                with self.subTest(workflow=workflow, sampler=spec.name):
                    page = FormPage(self.client.get(f"/generate/{identifier}").text)
                    self.assertIn(spec.name, [option["value"] for option in page.options["sampling.sampler"]])
                    data = page.data()
                    data.setlist("lora", [])
                    pair = {"sampler": spec.name, "noise_schedule": "default"}
                    for name, value in {"sampling.sampler": pair["sampler"], "sampling.noise_schedule": "default",
                                        "prompt": "A mountain lake", "action": "preview"}.items():
                        data[name] = value
                    response = self.client.post(f"/generate/{identifier}", data=data)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(FormPage(response.text).preview["overrides"]["sampling"], pair)
                    data["action"] = "save_model"
                    response = self.client.post(f"/settings/models/{identifier}", data=data)
                    self.assertEqual(response.status_code, 303)
                    page = FormPage(self.client.get(f"/generate/{identifier}").text)
                    self.assertEqual(page.controls["sampling.sampler"][0]["value"], spec.name)
                    data = page.data()
                    data.setlist("lora", [])
                    data["prompt"], data["action"] = "A mountain lake", "generate"
                    response = self.client.post(f"/generate/{identifier}", data=data)
                    self.assertEqual(response.status_code, 303)
                    job_id = response.location.rsplit("/", 1)[1]
                    self.assertEqual(JobStore(self.database).get(job_id).request["settings"]["sampling"], pair)


if __name__ == "__main__":
    unittest.main()

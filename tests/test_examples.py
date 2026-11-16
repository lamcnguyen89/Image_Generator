"""Example metadata editing must preserve settings and respect editor revisions."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.catalog import discover_catalog
from image_generator.catalog.persistence import SettingsConflict, settings_revision, update_examples
from test_generation_ui import FormPage


class ExampleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.root)
        self.app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": self.root})
        self.client = self.app.test_client()
        self.model = self.root / "FLUX/checkpoints/demo/model.defaults.json"
        self.lora = self.root / "FLUX/loras/style/lora.defaults.json"

    def form(self):
        return FormPage(self.client.get("/generate/fixture-flux").text).data()

    def edit(self, asset="fixture-flux", **changes):
        form = self.form()
        payload = {"csrf_token": form["csrf_token"], "settings_revision": form["settings_revision"],
                   "field": "example_prompts", "action": "add", "text": "A mountain lake"}
        payload.update(changes)
        return self.client.post(f"/api/assets/{asset}/examples", json=payload)

    def test_model_and_lora_positive_and_negative_examples_round_trip(self):
        for asset, path in (("fixture-flux", self.model), ("fixture-flux-style", self.lora)):
            for field in ("example_prompts", "example_negative_prompts"):
                with self.subTest(asset=asset, field=field):
                    before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
                    original = json.loads(path.read_text())
                    text = "A quiet lake, café colors\nsoft lighting"
                    response = self.edit(asset, field=field, text=f"  {text}  ")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json["examples"], [text])
                    self.assertEqual(json.loads(path.read_text()), {**original, field: [text]})
                    self.assertEqual({p: p.read_bytes() for p in before if p != path},
                                     {p: content for p, content in before.items() if p != path})
                    catalog = discover_catalog(self.root, initialize_missing=False)
                    self.assertEqual(response.json["settings_revision"], settings_revision(catalog))
                    response = self.edit(asset, field=field, action="remove", index=0)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json["examples"], [])
                    self.assertEqual(json.loads(path.read_text()), {**original, field: []})

    def test_dropdowns_render_saved_examples_and_escape_prompt_text(self):
        payload = '<script>alert("test")</script>\nA lake'
        self.assertEqual(self.edit(text=payload).status_code, 200)
        self.assertEqual(self.edit("fixture-flux-style", field="example_negative_prompts", text="blurry").status_code, 200)
        html = self.client.get("/generate/fixture-flux").text
        self.assertIn('id="examples-fixture-flux-example_prompts-select"', html)
        self.assertIn('id="examples-fixture-flux-style-example_negative_prompts-select"', html)
        self.assertIn('data-example-text="blurry"', html)
        self.assertIn("Use as prompt", html)
        self.assertIn("Use as negative prompt", html)
        self.assertIn("Add example", html)
        self.assertIn("Remove selected", html)
        self.assertIn("/catalog-assets/examples.js", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn(payload, html)
        form = FormPage(html).data()
        self.assertEqual(form["prompt"], "")
        self.assertEqual(form["negative_prompt"], "")

    def test_missing_legacy_example_fields_can_be_added_and_cleared(self):
        document = json.loads(self.model.read_text())
        document.pop("example_prompts")
        document.pop("example_negative_prompts")
        self.model.write_text(json.dumps(document))
        self.assertEqual(self.edit().status_code, 200)
        self.assertEqual(self.edit(action="remove", index=0).status_code, 200)
        saved = json.loads(self.model.read_text())
        self.assertEqual(saved["example_prompts"], [])
        self.assertNotIn("example_negative_prompts", saved)

    def test_invalid_changes_leave_all_documents_unchanged(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        changes = ({"field": "settings"}, {"field": []}, {"action": "replace"}, {"action": {}},
                   {"text": " \n "}, {"text": None}, {"text": ["invalid"]},
                   *({"action": "remove", "index": value} for value in (None, -1, 0, "0", True, 1.5)))
        for change in changes:
            with self.subTest(change=change):
                response = self.edit(**change)
                self.assertEqual(response.status_code, 422)
                self.assertIn("error", response.json)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_duplicate_example_does_not_write_or_remove_other_entries(self):
        self.assertEqual(self.edit(text="First").status_code, 200)
        self.assertEqual(self.edit(text="Second").status_code, 200)
        self.assertEqual(self.edit(text="First").status_code, 422)
        self.assertEqual(self.edit(action="remove", index=0).json["examples"], ["Second"])

    def test_csrf_unknown_assets_and_malformed_requests(self):
        before = self.model.read_bytes()
        self.assertEqual(self.edit(csrf_token="wrong").status_code, 400)
        self.assertEqual(self.edit(csrf_token=None).status_code, 400)
        self.assertEqual(self.edit("missing").status_code, 404)
        for payload in ([], None, "invalid"):
            self.assertEqual(self.client.post("/api/assets/fixture-flux/examples", json=payload).status_code, 400)
        self.assertEqual(self.model.read_bytes(), before)

    def test_stale_or_missing_revision_does_not_overwrite_edits(self):
        stale = self.form()["settings_revision"]
        self.assertEqual(self.edit().status_code, 200)
        before = self.model.read_bytes()
        for revision in (stale, None, ""):
            self.assertEqual(self.edit(settings_revision=revision, text="Other").status_code, 409)
        self.assertEqual(self.model.read_bytes(), before)

    def test_returned_revision_allows_saving_the_existing_generation_form(self):
        form = self.form()
        form["prompt"] = "Unsaved prompt"
        form["width"] = "512"
        response = self.edit()
        form["settings_revision"] = response.json["settings_revision"]
        form["action"] = "save_model"
        response = self.client.post("/settings/models/fixture-flux", data=form)
        self.assertEqual(response.status_code, 303)
        saved = json.loads(self.model.with_name("model.user.json").read_text())
        self.assertEqual(saved["settings"]["prompt"], "Unsaved prompt")
        self.assertEqual(saved["settings"]["width"], 512)

    def test_failed_atomic_replace_preserves_defaults_and_permissions(self):
        before = self.model.read_bytes()
        permissions = self.model.stat().st_mode
        with patch("image_generator.catalog.persistence.os.replace", side_effect=OSError("disk failure")):
            self.assertEqual(self.edit().status_code, 422)
        self.assertEqual(self.model.read_bytes(), before)
        self.assertEqual(list(self.model.parent.glob(".model.defaults.json.*")), [])
        self.assertEqual(self.edit().status_code, 200)
        self.assertEqual(self.model.stat().st_mode, permissions)

    def test_concurrent_metadata_changes_accept_only_one_editor_revision(self):
        catalog = discover_catalog(self.root, initialize_missing=False)
        revision = settings_revision(catalog)

        def save(text):
            try:
                update_examples(catalog, "fixture-flux", "example_prompts", "add",
                                text=text, expected_revision=revision)
                return "saved"
            except SettingsConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertCountEqual(executor.map(save, ["First", "Second"]), ["saved", "conflict"])
        self.assertEqual(len(json.loads(self.model.read_text())["example_prompts"]), 1)


if __name__ == "__main__":
    unittest.main()

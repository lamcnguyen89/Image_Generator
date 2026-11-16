"""Inline edits save one setting with workflow validation and editor revisions."""

from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
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
from image_generator.catalog.persistence import save_model_settings, settings_revision
from test_web import SettingsTable


class InlinePage(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.panel = {}
        self.fields = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-settings-panel" in attrs:
            self.panel = attrs
        if "data-setting-row" in attrs:
            self.fields.append(attrs["data-field"])


class InlineSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.root)
        self.app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": self.root})
        self.client = self.app.test_client()

    def page(self, model="fixture-sdxl"):
        response = self.client.get(f"/models/{model}")
        self.assertEqual(response.status_code, 200)
        return InlinePage(response.text)

    def edit(self, field="negative_prompt", text="blurry", model="fixture-sdxl", **changes):
        page = self.page(model)
        payload = {"field": field, "value_text": text, "csrf_token": page.panel["data-csrf-token"],
                   "settings_revision": page.panel["data-settings-revision"]}
        payload.update(changes)
        return self.client.post(f"/api/models/{model}/settings/field", json=payload)

    def user(self, family="SDXL"):
        return self.root / family / "checkpoints/demo/model.user.json"

    def test_every_resolved_setting_gets_an_edit_button_without_changing_values(self):
        catalog = discover_catalog(self.root, initialize_missing=False)
        for model in catalog.models.values():
            response = self.client.get(f"/models/{model.id}")
            html = response.text
            page = InlinePage(html)
            self.assertEqual(page.fields, list(model.resolve_settings()))
            self.assertEqual(SettingsTable(html).values, model.resolve_settings())
            self.assertEqual(html.count('data-setting-edit aria-label="Edit '), len(page.fields))
            self.assertIn("/catalog-assets/settings.js", html)
            self.assertNotIn("Read only", html)

    def test_prompt_edit_preserves_every_other_setting_and_source_document(self):
        user = self.user()
        user.write_text(json.dumps({"schema_version": 1, "settings": {"seed": 19, "custom": {"keep": True}}}))
        before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        text = '<script>alert("x")</script>\nblurry, café'
        response = self.edit(text=text)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.json["display_value"]), text)
        self.assertEqual(response.json["editor_value"], text)
        self.assertEqual(json.loads(user.read_text())["settings"], {"seed": 19, "custom": {"keep": True}, "negative_prompt": text})
        for path, content in before.items():
            if path != user:
                self.assertEqual(path.read_bytes(), content)
        html = self.client.get("/models/fixture-sdxl").text
        self.assertNotIn(text, html)
        self.assertIn("&lt;script&gt;", html)
        self.assertEqual(SettingsTable(html).values["negative_prompt"], text)

    def test_text_edits_work_in_all_six_families_without_weights(self):
        for family in ("sdxl", "flux", "qwen_2_1", "krea2", "wan", "minimax"):
            with self.subTest(family=family):
                response = self.edit("positive_prefix", "A gentle light", model=f"fixture-{family}")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json["editor_value"], "A gentle light")

    def test_numeric_boolean_nullable_list_and_nested_object_edits(self):
        edits = (("fixture-sdxl", "width", "512", 512),
                 ("fixture-sdxl", "seed", "9223372036854775807", 9223372036854775807),
                 ("fixture-sdxl", "seed", "null", None),
                 ("fixture-sdxl", "trigger_words", '["soft light", "portrait"]', ["soft light", "portrait"]),
                 ("fixture-sdxl", "sampling", '{"sampler": "euler_ancestral", "noise_schedule": "default"}',
                  {"sampler": "euler_ancestral", "noise_schedule": "default"}),
                 ("fixture-wan", "preserve_aspect_ratio", "false", False))
        for model, field, text, value in edits:
            with self.subTest(model=model, field=field):
                response = self.edit(field, text, model=model)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(json.loads(response.json["display_value"]), value)
                family = "WAN" if model == "fixture-wan" else "SDXL"
                self.assertEqual(json.loads(response.json["overrides_json"]), json.loads(self.user(family).read_text())["settings"])
                self.assertEqual(SettingsTable(self.client.get(f"/models/{model}").text).values[field], value)

    def test_returned_revision_can_save_the_next_row(self):
        original = json.loads(self.user().read_text())["settings"]
        page = self.page()
        response = self.edit(text="blurry")
        self.assertEqual(response.status_code, 200)
        response = self.client.post(page.panel["data-save-url"], json={
            "field": "width", "value_text": "512", "csrf_token": page.panel["data-csrf-token"],
            "settings_revision": response.json["settings_revision"],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(self.user().read_text())["settings"], {**original, "negative_prompt": "blurry", "width": 512})

    def test_stale_revision_rejects_edit_without_losing_newer_settings(self):
        stale = self.page().panel["data-settings-revision"]
        self.assertEqual(self.edit(text="newer edit").status_code, 200)
        before = self.user().read_bytes()
        self.assertEqual(self.edit("width", "512", settings_revision=stale).status_code, 409)
        self.assertEqual(self.user().read_bytes(), before)

    def test_invalid_values_shapes_modes_and_fields_leave_files_unchanged(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        edits = (("width", "513"), ("width", "true"), ("seed", "-1"), ("seed", "9223372036854775808"),
                 ("guidance_scale", "NaN"), ("trigger_words", '[""]'), ("sampling", "[]"),
                 ("sampling", '{"sampler":"bad", "noise_schedule":"default"}'),
                 ("sampling", "{invalid"), ("loras", "{}"), ("weights", '"changed"'),
                 ("sampling.sampler", '"euler"'), (None, "1"), ([], "1"))
        for field, text in edits:
            with self.subTest(field=field, text=text):
                response = self.edit(field, text)
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.edit(model="fixture-minimax", text="unsupported").status_code, 422)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_csrf_missing_revision_unknown_model_and_malformed_requests(self):
        for token in (None, "wrong"):
            self.assertEqual(self.edit(csrf_token=token).status_code, 400)
        for revision in (None, ""):
            self.assertEqual(self.edit(settings_revision=revision).status_code, 409)
        for raw in (None, []):
            self.assertEqual(self.edit(value_text=raw).status_code, 422)
        page = self.page()
        response = self.client.post("/api/models/missing/settings/field", json={"csrf_token": page.panel["data-csrf-token"]})
        self.assertEqual(response.status_code, 404)
        for payload in ([], None, "bad"):
            self.assertEqual(self.client.post(page.panel["data-save-url"], json=payload).status_code, 400)

    def test_unknown_and_incompatible_lora_links_are_rejected(self):
        for links in ([{"id": "missing", "enabled": True}], [{"id": "fixture-flux-style", "enabled": True}]):
            self.assertEqual(self.edit("loras", json.dumps(links)).status_code, 422)
        links = [{"id": "fixture-sdxl-style", "enabled": True, "strength": 0.3}]
        self.assertEqual(self.edit("loras", json.dumps(links)).status_code, 200)
        self.assertIn("Saved strength: <code>0.3</code>", self.client.get("/models/fixture-sdxl").text)

    def test_existing_extension_field_and_resetting_to_inherited_value(self):
        user = self.user()
        user.write_text(json.dumps({"schema_version": 1, "settings": {"custom": {"value": 1}, "width": 512}}))
        self.assertEqual(self.edit("custom", '{"value":2}').status_code, 200)
        self.assertEqual(self.edit("width", "1024").status_code, 200)
        self.assertEqual(json.loads(user.read_text())["settings"], {"custom": {"value": 2}})

    def test_reset_optional_setting_to_workflow_default_removes_row_without_error(self):
        self.user().write_text(json.dumps({"schema_version": 1, "settings": {"clip_skip": 1}}))
        response = self.edit("clip_skip", "null")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["removed"])
        self.assertEqual(response.json["display_value"], "null")
        self.assertNotIn("clip_skip", SettingsTable(self.client.get("/models/fixture-sdxl").text).values)

    def test_atomic_write_failure_preserves_previous_values(self):
        before = self.user().read_bytes()
        with patch("image_generator.catalog.persistence.os.replace", side_effect=OSError("disk failure")):
            response = self.edit()
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.user().read_bytes(), before)
        self.assertEqual(list(self.user().parent.glob(".model.user.json.*")), [])

    def test_concurrent_row_saves_only_accept_one_revision(self):
        catalog = discover_catalog(self.root, initialize_missing=False)
        revision = settings_revision(catalog)
        from image_generator.catalog.persistence import SettingsConflict

        def save(field, value):
            try:
                save_model_settings(catalog, "fixture-sdxl", {field: value}, scope=[field], expected_revision=revision)
                return "saved"
            except SettingsConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(save, "width", 512), executor.submit(save, "negative_prompt", "blurry")]
            self.assertCountEqual([future.result() for future in futures], ["saved", "conflict"])


if __name__ == "__main__":
    unittest.main()

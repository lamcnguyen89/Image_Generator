"""Copied example media updates manifests without changing generation settings."""

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
from image_generator.catalog import CatalogValidationError, discover_catalog
from image_generator.catalog.media import example_media
from image_generator.catalog.persistence import settings_revision, sync_example_media, update_examples
from test_generation_ui import FormPage


class MediaTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.root)
        self.model = self.root / "FLUX/checkpoints/demo/model.defaults.json"
        self.lora = self.root / "FLUX/loras/style/lora.defaults.json"
        self.app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": self.root})
        self.client = self.app.test_client()

    def read(self, path):
        return json.loads(path.read_text())

    def catalog(self, **kwargs):
        return discover_catalog(self.root, **kwargs)

    def test_refresh_registers_arbitrary_filenames_for_checkpoint_and_lora(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        for path in (self.model, self.lora):
            (path.parent / "Example.PNG").write_bytes(b"first")
            (path.parent / "random download.jpg").write_bytes(b"second")
        for url, asset_id in (("/", "fixture-flux"), ("/generate/fixture-flux", "fixture-flux-style")):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn(f'data-media-src="/assets/{asset_id}/preview/1"', response.text)
            media = self.client.get(f"/assets/{asset_id}/preview/1")
            self.assertEqual(media.data, b"second")
            media.close()
        for path, content in before.items():
            if path in (self.model, self.lora):
                self.assertEqual(self.read(path), {
                    **json.loads(content), "example_media": ["Example.PNG", "random download.jpg"],
                })
            else:
                self.assertEqual(path.read_bytes(), content)

    def test_added_renamed_and_deleted_files_update_list_and_carousel(self):
        first = self.model.parent / "photo.png"
        first.write_bytes(b"photo")
        self.catalog()
        self.assertEqual(self.read(self.model)["example_media"], ["photo.png"])
        second = self.model.parent / "second.webp"
        second.write_bytes(b"second")
        first.rename(first.with_name("renamed.png"))
        self.assertIn("data-asset-carousel", self.client.get("/").text)
        self.assertEqual(self.read(self.model)["example_media"], ["renamed.png", "second.webp"])
        second.unlink()
        self.assertNotIn("data-asset-carousel", self.client.get("/").text)
        self.assertEqual(self.read(self.model)["example_media"], ["renamed.png"])
        first.with_name("renamed.png").unlink()
        self.catalog()
        self.assertEqual(self.read(self.model)["example_media"], [])
        self.assertEqual(self.client.get("/assets/fixture-flux/preview").status_code, 404)

    def test_configured_preview_is_first_and_deduplicated(self):
        document = self.read(self.model)
        document["preview_image"] = "z-cover.jpg"
        self.model.write_text(json.dumps(document))
        for name in ("z-cover.jpg", "a-photo.png", "b-video.webm"):
            (self.model.parent / name).write_bytes(name.encode())
        (self.model.parent / "cover-alias.jpg").symlink_to("z-cover.jpg")
        catalog = self.catalog()
        names = ["z-cover.jpg", "a-photo.png", "b-video.webm"]
        self.assertEqual(self.read(self.model)["example_media"], names)
        self.assertEqual([p.name for p in example_media(catalog.get_model("fixture-flux"))], names)

    def test_hidden_nonmedia_nested_and_external_files_are_not_discovered(self):
        directory = self.model.parent
        (directory / "keep.png").write_bytes(b"allowed")
        (directory / ".hidden.png").write_bytes(b"hidden")
        (directory / "weights.safetensors").write_bytes(b"weights")
        (directory / "nested").mkdir()
        (directory / "nested/photo.png").write_bytes(b"nested")
        outside = self.root / "private.png"
        outside.write_bytes(b"private")
        (directory / "escape.png").symlink_to(outside)
        (directory / "broken.jpg").symlink_to("missing.jpg")
        self.catalog()
        self.assertEqual(self.read(self.model)["example_media"], ["keep.png"])
        document = self.read(self.model)
        document["preview_image"] = "nested/photo.png"
        self.model.write_text(json.dumps(document))
        self.catalog()
        self.assertEqual(self.read(self.model)["example_media"], ["nested/photo.png", "keep.png"])

    def test_read_only_catalog_does_not_sync_and_stable_loads_do_not_rewrite(self):
        (self.model.parent / "photo.png").write_bytes(b"photo")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.root.rglob("*.json")}
        self.catalog(initialize_missing=False)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
        self.catalog()
        synced = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
        with patch("image_generator.catalog.persistence._write_document", side_effect=AssertionError("Unexpected write")):
            self.catalog()
        self.assertEqual(synced, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_sync_preserves_generation_form_revision_and_unsaved_settings(self):
        form = FormPage(self.client.get("/generate/fixture-flux").text).data()
        revision = form["settings_revision"]
        (self.model.parent / "new photo.png").write_bytes(b"photo")
        self.client.get("/")
        self.assertEqual(settings_revision(self.catalog()), revision)
        form["prompt"] = "Unsaved mountain lake prompt"
        form["action"] = "save_model"
        response = self.client.post("/settings/models/fixture-flux", data=form)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.read(self.model.with_name("model.user.json"))["settings"]["prompt"], form["prompt"])
        self.assertEqual(self.read(self.model)["example_media"], ["new photo.png"])

    def test_sync_reloads_stale_defaults_before_writing(self):
        catalog = self.catalog()
        document = self.read(self.model)
        document["description"] = "Updated description"
        document["settings"]["width"] = 512
        document["example_prompts"] = ["Updated prompt example"]
        self.model.write_text(json.dumps(document))
        (self.model.parent / "photo.png").write_bytes(b"photo")
        current = sync_example_media(catalog)
        self.assertEqual(self.read(self.model), {**document, "example_media": ["photo.png"]})
        self.assertEqual(current.get_model("fixture-flux").defaults, self.read(self.model))

    def test_concurrent_prompt_edit_and_sync_preserve_both_changes(self):
        catalog = self.catalog()
        revision = settings_revision(catalog)
        (self.model.parent / "photo.png").write_bytes(b"photo")
        with ThreadPoolExecutor(max_workers=2) as executor:
            sync = executor.submit(sync_example_media, catalog)
            edit = executor.submit(update_examples, catalog, "fixture-flux", "example_prompts", "add",
                                   text="New prompt", expected_revision=revision)
            sync.result()
            edit.result()
        saved = self.read(self.model)
        self.assertEqual(saved["example_media"], ["photo.png"])
        self.assertEqual(saved["example_prompts"], ["New prompt"])

    def test_failed_atomic_write_preserves_manifest_and_removes_temporary_file(self):
        before = self.model.read_bytes()
        permissions = self.model.stat().st_mode
        (self.model.parent / "photo.png").write_bytes(b"photo")
        with patch("image_generator.catalog.persistence.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(CatalogValidationError) as context:
                self.catalog()
        self.assertIn("example_media", str(context.exception))
        self.assertEqual(self.model.read_bytes(), before)
        self.assertEqual(list(self.model.parent.glob(".model.defaults.json.*")), [])
        self.catalog()
        self.assertEqual(self.model.stat().st_mode, permissions)

    def test_invalid_example_media_is_reported_without_rewriting_manifest(self):
        for value in (None, "photo.png", [1], [""]):
            document = {**self.read(self.model), "example_media": value}
            self.model.write_text(json.dumps(document))
            before = self.model.read_bytes()
            with self.subTest(value=value), self.assertRaises(CatalogValidationError) as context:
                self.catalog()
            self.assertTrue(any(issue.field == "example_media" for issue in context.exception.issues))
            self.assertEqual(self.model.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

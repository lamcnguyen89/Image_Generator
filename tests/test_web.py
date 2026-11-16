"""Exercise read-only Flask pages with temporary catalog manifests."""

from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app


class SettingsTable(HTMLParser):
    """Read rendered values rather than asserting values in hidden source JSON."""

    def __init__(self, html):
        super().__init__()
        self.values = {}
        self.key = []
        self.value = []
        self.cell = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.key, self.value = [], []
        elif tag == "th" and dict(attrs).get("scope") == "row":
            self.cell = self.key
        elif tag == "pre" and "setting-value" in dict(attrs).get("class", "").split():
            self.cell = self.value

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "pre"):
            self.cell = None
        elif tag == "tr" and self.key:
            self.values["".join(self.key)] = json.loads("".join(self.value))


class WebTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_path = self.root / "SDXL/checkpoints/demo/model.defaults.json"
        self.model = {
            "schema_version": 1, "id": "demo-model", "display_name": "Demo checkpoint",
            "description": "A local test model.", "family": "sdxl", "workflow": "sdxl",
            "weights": None, "base_model_id": "example/pretrained",
            "supported_modes": ["txt2img"], "requirements": {"dimension_multiple": 8},
            "settings": {"width": 1024, "height": 1024, "seed": 42, "loras": [],
                         "sampling": {"sampler": "euler", "noise_schedule": "default"}},
        }
        self.write(self.model_path, self.model)
        self.app = create_app({"TESTING": True, "CATALOG_ROOT": self.root})
        self.client = self.app.test_client()

    def write(self, path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def write_user(self, settings):
        return self.write(self.model_path.with_name("model.user.json"), {"schema_version": 1, "settings": settings})

    def write_lora(self):
        return self.write(self.root / "SDXL/loras/demo/lora.defaults.json", {
            "schema_version": 1, "id": "demo-lora", "display_name": "Demo adapter",
            "family": "sdxl", "adapter_type": "lora", "weights": "missing.safetensors",
            "compatibility": {"verified_checkpoint_ids": []},
            "settings": {"strength": 0.8, "trigger_words": ["demo"], "recommended_generation": {"width": 512}},
        })

    def test_catalog_lists_models_modes_and_links(self):
        self.write_lora()
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Demo checkpoint", html)
        self.assertIn('href="/models/demo-model"', html)
        self.assertIn("Text to image", html)
        self.assertIn("1 model", html)
        self.assertIn("1 registered LoRA", html)
        self.assertNotIn("<form", html)

    def test_example_images_and_videos_render_above_card_details(self):
        (self.model_path.parent / "Example.PNG").write_bytes(b"example image")
        lora = self.write_lora()
        (lora.parent / "example.mp4").write_bytes(b"example video")
        self.write_user({"loras": [{"id": "demo-lora", "enabled": False}]})

        html = self.client.get("/").text
        self.assertIn('<img class="asset-media" src="/assets/demo-model/preview"', html)
        self.assertRegex(html, r'<a class="asset-media-link" href="/generate/demo-model"[^>]*>\s*<img ')
        self.assertIn('<h3 class="asset-preview-title"><a href="/generate/demo-model">', html)
        self.assertLess(html.index('class="asset-preview-title"'), html.index('class="asset-card-body"'))
        self.assertLess(html.index('class="asset-card-body"'), html.index("A local test model."))
        for url, body_class in (("/models/demo-model", "asset-card-body"),
                                ("/generate/demo-model", "adapter-body")):
            with self.subTest(url=url):
                html = self.client.get(url).text
                self.assertIn('<video class="asset-media" controls muted playsinline preload="metadata"', html)
                self.assertIn('<source src="/assets/demo-lora/preview" type="video/mp4">', html)
                self.assertLess(html.index('class="asset-preview-title"'), html.index(f'class="{body_class}"'))
        self.assertIn('name="lora" value="demo-lora"', html)
        self.assertIn('name="strength:demo-lora"', html)

    def test_template_edits_appear_without_restarting_the_app(self):
        from jinja2 import ChoiceLoader, FileSystemLoader

        template = self.root / "templates/controls/asset_preview.html"
        template.parent.mkdir(parents=True)
        template.write_text('{% macro asset_preview(asset, href=none) %}Original preview{% endmacro %}')
        self.app.jinja_env.loader = ChoiceLoader([
            FileSystemLoader(self.root / "templates"), self.app.jinja_env.loader,
        ])
        self.assertIn("Original preview", self.client.get("/").text)
        previous_mtime = template.stat().st_mtime
        template.write_text('{% macro asset_preview(asset, href=none) %}Updated preview{% endmacro %}')
        os.utime(template, (previous_mtime + 2, previous_mtime + 2))
        html = self.client.get("/").text
        self.assertIn("Updated preview", html)
        self.assertNotIn("Original preview", html)

    def test_preview_prefers_configured_media_and_supports_video_byte_ranges(self):
        self.write(self.model_path, {**self.model, "preview_image": "chosen.webm"})
        (self.model_path.parent / "example.png").write_bytes(b"fallback image")
        video = bytes(range(128))
        (self.model_path.parent / "chosen.webm").write_bytes(video)
        response = self.client.get("/assets/demo-model/preview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "video/webm")
        self.assertEqual(response.data, video)
        response.close()
        response = self.client.get("/assets/demo-model/preview", headers={"Range": "bytes=8-15"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, video[8:16])
        response.close()

    def test_missing_configured_preview_falls_back_to_existing_example_extension(self):
        self.write(self.model_path, {**self.model, "preview_image": "example.png"})
        (self.model_path.parent / "example.jpeg").write_bytes(b"existing jpeg")
        response = self.client.get("/assets/demo-model/preview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertEqual(response.data, b"existing jpeg")
        response.close()

    def test_assets_without_media_keep_titles_and_show_placeholder(self):
        html = self.client.get("/").text
        self.assertIn("No example media available", html)
        self.assertIn("Demo checkpoint", html)
        self.assertNotIn('<img class="asset-media"', html)
        self.assertEqual(self.client.get("/assets/demo-model/preview").status_code, 404)
        self.assertEqual(self.client.get("/assets/missing/preview").status_code, 404)

    def test_carousels_render_for_checkpoints_and_both_lora_card_locations(self):
        lora = self.write_lora()
        self.write_user({"loras": [{"id": "demo-lora", "enabled": True}]})
        for manifest in (self.model_path, lora):
            (manifest.parent / "example-01.png").write_bytes(b"first image")
            (manifest.parent / "example-02.webp").write_bytes(b"second image")
            (manifest.parent / "example-03.mp4").write_bytes(b"third video")

        for url, asset in (("/", "demo-model"), ("/models/demo-model", "demo-lora"),
                           ("/generate/demo-model", "demo-lora")):
            with self.subTest(url=url):
                html = self.client.get(url).text
                self.assertEqual(html.count('data-asset-carousel role="group"'), 1)
                self.assertEqual(html.count('aria-roledescription="slide"'), 3)
                self.assertIn(f'data-media-src="/assets/{asset}/preview"', html)
                self.assertIn(f'data-media-src="/assets/{asset}/preview/1"', html)
                self.assertIn(f'data-media-src="/assets/{asset}/preview/2"', html)
                self.assertIn('type="button" data-carousel-previous', html)
                self.assertIn('type="button" data-carousel-next', html)
                self.assertIn('aria-atomic="true">1 / 3</span>', html)
                self.assertIn('/catalog-assets/carousel.js', html)
                self.assertIn('preload="none"', html)
        response = self.client.get("/catalog-assets/carousel.js")
        self.assertEqual(response.status_code, 200)
        response.close()

    def test_carousel_markup_cannot_prefetch_images_or_videos(self):
        (self.model_path.parent / "video.webm").write_bytes(b"video")
        self.write(self.model_path, {**self.model, "preview_image": "video.webm"})
        for index in range(20):
            (self.model_path.parent / f"photo-{index:02}.png").write_bytes(b"photo")

        class MediaElements(HTMLParser):
            def __init__(self, html):
                super().__init__()
                self.elements = []
                self.feed(html)

            def handle_starttag(self, tag, attrs):
                if tag in {"img", "video", "source"}:
                    self.elements.append((tag, dict(attrs)))

        html = self.client.get("/").text
        elements = MediaElements(html).elements
        self.assertEqual(len(elements), 22)  # 20 images, one video and its source.
        for tag, attrs in elements:
            self.assertFalse({"src", "srcset", "poster"} & attrs.keys())
            if tag == "video":
                self.assertEqual(attrs["preload"], "none")
            else:
                self.assertIn("data-media-src", attrs)
        self.assertIn("Enable JavaScript to browse examples.", html)

    def test_single_preview_does_not_show_carousel_controls(self):
        (self.model_path.parent / "example.png").write_bytes(b"single image")
        html = self.client.get("/").text
        self.assertIn('src="/assets/demo-model/preview"', html)
        self.assertNotIn('data-asset-carousel', html)
        self.assertNotIn('data-carousel-next', html)

    def test_carousel_media_order_deduplication_and_video_ranges(self):
        self.write(self.model_path, {**self.model, "preview_image": "example.png"})
        (self.model_path.parent / "example.png").write_bytes(b"chosen image")
        (self.model_path.parent / "EXAMPLE-01.JPG").write_bytes(b"second image")
        video = bytes(range(128))
        (self.model_path.parent / "example-02.webm").write_bytes(video)
        # An alias of the chosen preview must not add a duplicate slide.
        (self.model_path.parent / "example-03.png").symlink_to("example.png")
        for suffix, mimetype, payload in (("", "image/png", b"chosen image"),
                                          ("/1", "image/jpeg", b"second image"),
                                          ("/2", "video/webm", video)):
            response = self.client.get(f"/assets/demo-model/preview{suffix}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, mimetype)
            self.assertEqual(response.data, payload)
            response.close()
        self.assertEqual(self.client.get("/assets/demo-model/preview/3").status_code, 404)
        response = self.client.get("/assets/demo-model/preview/2", headers={"Range": "bytes=8-15"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, video[8:16])
        response.close()

    def test_carousel_excludes_nonmedia_hidden_files_directories_and_escaping_symlinks(self):
        outside = self.root / "private.png"
        outside.write_bytes(b"private image")
        directory = self.model_path.parent
        self.write(self.model_path, {**self.model, "preview_image": str(outside)})
        (directory / "example-01.png").symlink_to(outside)
        (directory / "example-02.jpg").mkdir()
        (directory / "example-03.safetensors").write_bytes(b"weights")
        (directory / "notes.txt").write_bytes(b"unrelated text")
        (directory / ".hidden.png").write_bytes(b"hidden image")
        (directory / "example-04.png").write_bytes(b"safe image")
        response = self.client.get("/assets/demo-model/preview")
        self.assertEqual(response.data, b"safe image")
        response.close()
        for suffix in ("/1", "/-1", "/999", "/example-01.png", "/../model.defaults.json"):
            with self.subTest(suffix=suffix):
                self.assertEqual(self.client.get(f"/assets/demo-model/preview{suffix}").status_code, 404)
        self.assertNotIn("data-asset-carousel", self.client.get("/").text)

    def test_preview_does_not_serve_unrelated_files_or_symlink_targets(self):
        outside = self.root / "private.png"
        outside.write_bytes(b"private image")
        for configured in (str(outside), "../../../private.png", "model.defaults.json", "weights.safetensors"):
            with self.subTest(configured=configured):
                self.write(self.model_path, {**self.model, "preview_image": configured})
                self.assertEqual(self.client.get("/assets/demo-model/preview").status_code, 404)
        (self.model_path.parent / "example.png").symlink_to(outside)
        self.assertEqual(self.client.get("/assets/demo-model/preview").status_code, 404)

    def test_settings_page_displays_resolved_values_and_sources(self):
        self.write_user({"width": 768, "seed": None, "sampling": {"sampler": "custom"}})
        response = self.client.get("/models/demo-model")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        settings = SettingsTable(html).values
        self.assertEqual(settings["width"], 768)
        self.assertEqual(settings["height"], 1024)
        self.assertIsNone(settings["seed"])
        self.assertEqual(settings["sampling"], {"sampler": "custom", "noise_schedule": "default"})
        for text in ("Saved settings", "Saved user overrides", "Pipeline requirements", "example/pretrained", str(self.model_path)):
            self.assertIn(text, html)
        self.assertEqual(html.count('data-setting-edit aria-label="Edit '), len(settings))
        self.assertIn('/catalog-assets/settings.js', html)

    def test_linked_loras_show_saved_selection_and_checkpoint_strength(self):
        lora = self.write_lora()
        self.write(lora.with_name("lora.user.json"), {"schema_version": 1, "settings": {"strength": 0.6}})
        self.write_user({"loras": [{"id": "demo-lora", "enabled": False, "strength": 0.3}]})
        response = self.client.get("/models/demo-model")
        html = response.get_data(as_text=True)
        self.assertIn("Demo adapter", html)
        self.assertIn("Disabled", html)
        self.assertIn("Saved strength: <code>0.3</code>", html)
        self.assertIn("Compatibility has not been verified", html)
        self.assertEqual(SettingsTable(html).values["width"], 1024)
        self.write_user({"loras": [{"id": "demo-lora", "enabled": True}]})
        html = self.client.get("/models/demo-model").get_data(as_text=True)
        self.assertIn("Saved strength: <code>0.6</code>", html)
        self.assertIn("Enabled", html)

    def test_unlinked_adapters_are_not_automatically_shown_as_linked(self):
        self.write_lora()
        html = self.client.get("/models/demo-model").get_data(as_text=True)
        self.assertIn("No LoRAs are linked", html)
        self.assertNotIn("Demo adapter", html)

    def test_manifest_edits_appear_on_next_request(self):
        self.assertEqual(SettingsTable(self.client.get("/models/demo-model").get_data(as_text=True)).values["width"], 1024)
        self.write_user({"width": 640})
        self.assertEqual(SettingsTable(self.client.get("/models/demo-model").get_data(as_text=True)).values["width"], 640)

    def test_browsing_creates_missing_user_manifest(self):
        user = self.model_path.with_name("model.user.json")
        self.assertFalse(user.exists())
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(json.loads(user.read_text()), {"schema_version": 1, "settings": {}})

    def test_refresh_discovers_new_checkpoint_and_lora_without_manifests(self):
        for folder, kind in (("checkpoints", "model"), ("loras", "lora")):
            parent = self.root / "SDXL" / folder
            directory = parent / "new-upload"
            directory.mkdir(parents=True)
            (directory / "uploaded.safetensors").touch()
            for layer in ("defaults", "user"):
                name = f"{kind}.{layer}.example.json"
                shutil.copyfile(ROOT / "Models/Model_Families/SDXL" / folder / name, parent / name)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("new-upload", response.get_data(as_text=True))
        self.assertIn("2 models", response.get_data(as_text=True))
        self.assertIn("1 registered LoRA", response.get_data(as_text=True))
        model = json.loads((self.root / "SDXL/checkpoints/new-upload/model.defaults.json").read_text())
        self.assertEqual(self.client.get(f"/models/{model['id']}").status_code, 200)

    def test_incomplete_family_does_not_block_catalog_or_configured_model_pages(self):
        for folder in ("checkpoints", "loras"):
            directory = self.root / "Krea2" / folder / "new-upload"
            directory.mkdir(parents=True)
            (directory / "uploaded.safetensors").touch()
        for url in ("/", "/models/demo-model", "/generate/demo-model"):
            with self.subTest(url=url), self.assertLogs("image_generator.catalog.initialization", level="WARNING"):
                response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Demo checkpoint", response.get_data(as_text=True))
            self.assertNotIn("new-upload", response.get_data(as_text=True))
        self.assertFalse(list((self.root / "Krea2").rglob("*.json")))

    def test_unknown_model_and_lora_id_return_404(self):
        self.write_lora()
        for url in ("/models/missing", "/models/demo-lora", "/missing"):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 404)
                self.assertIn("Page not found", response.get_data(as_text=True))

    def test_pages_reject_writes_and_ignore_request_overrides(self):
        self.write_user({"width": 768})
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        for url in ("/", "/models/demo-model"):
            for method in ("POST", "PUT", "PATCH", "DELETE"):
                with self.subTest(url=url, method=method):
                    self.assertEqual(self.client.open(url, method=method, json={"width": 512}).status_code, 405)
        html = self.client.get("/models/demo-model?width=512").get_data(as_text=True)
        self.assertEqual(SettingsTable(html).values["width"], 768)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*.json")})

    def test_invalid_catalog_shows_diagnostics_and_recovers_after_fix(self):
        self.model_path.write_text("{", encoding="utf-8")
        for url in ("/", "/models/demo-model"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 503)
            self.assertIn("Catalog unavailable", response.get_data(as_text=True))
            self.assertIn(str(self.model_path), response.get_data(as_text=True))
        self.write(self.model_path, self.model)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_missing_catalog_root_shows_diagnostics(self):
        self.app.config["CATALOG_ROOT"] = self.root / "missing"
        response = self.client.get("/")
        self.assertEqual(response.status_code, 503)
        self.assertIn("existing directory", response.get_data(as_text=True))

    def test_empty_catalog_has_empty_state(self):
        self.model_path.unlink()
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("No models found", response.get_data(as_text=True))

    def test_manifest_content_is_escaped(self):
        payload = '<script>alert("test")</script>'
        self.write(self.model_path, {**self.model, "display_name": payload, "description": payload,
                                   "settings": {**self.model["settings"], "prompt": payload}})
        for url in ("/", "/models/demo-model"):
            html = self.client.get(url).get_data(as_text=True)
            self.assertNotIn(payload, html)
            self.assertIn("&lt;script&gt;", html)

    def test_static_assets_are_served_even_when_catalog_is_invalid(self):
        self.model_path.write_text("{", encoding="utf-8")
        response = self.client.get("/catalog-assets/catalog.css")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/css")
        response.close()

    def test_app_instances_use_independent_catalog_roots(self):
        other_root = self.root / "empty"
        other_root.mkdir()
        other = create_app({"TESTING": True, "CATALOG_ROOT": other_root})
        self.assertIn("No models found", other.test_client().get("/").get_data(as_text=True))
        self.assertIn("Demo checkpoint", self.client.get("/").get_data(as_text=True))

    def test_repository_catalog_and_every_model_page_render(self):
        app = create_app({"TESTING": True, "CATALOG_ROOT": ROOT / "Models/Model_Families"})
        from image_generator.catalog import discover_catalog

        catalog = discover_catalog(app.config["CATALOG_ROOT"])
        with app.test_client() as client:
            self.assertEqual(client.get("/").status_code, 200)
            for model in catalog.models.values():
                with self.subTest(model=model.id):
                    response = client.get(f"/models/{model.id}")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(SettingsTable(response.get_data(as_text=True)).values, model.resolve_settings())


if __name__ == "__main__":
    unittest.main()

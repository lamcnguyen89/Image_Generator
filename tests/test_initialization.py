"""Initialize real catalog layouts with fake weights, without GPU dependencies."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import (
    CatalogValidationError, discover_catalog, initialize_asset, initialize_catalog,
)


class InitializationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for family in ("SDXL", "FLUX", "WAN"):
            for folder, kind in (("checkpoints", "model"), ("loras", "lora")):
                target = self.root / family / folder
                target.mkdir(parents=True)
                for layer in ("defaults", "user"):
                    name = f"{kind}.{layer}.example.json"
                    shutil.copyfile(ROOT / "Models/Model_Families" / family / folder / name, target / name)

    def asset(self, family="SDXL", folder="checkpoints", name="demo"):
        directory = self.root / family / folder / name
        directory.mkdir()
        if family == "WAN" and folder == "checkpoints":
            (directory / "model_index.json").write_text("{}", encoding="utf-8")
        else:
            (directory / "actual-weights.safetensors").write_bytes(b"fake weights")
        return directory

    def test_discovery_initializes_models_and_loras_in_all_three_families(self):
        for family in ("SDXL", "FLUX", "WAN"):
            for folder in ("checkpoints", "loras"):
                self.asset(family, folder)
        before_templates = {path: path.read_bytes() for path in self.root.rglob("*.example.json")}
        catalog = discover_catalog(self.root)
        self.assertEqual(len(catalog.models), 3)
        self.assertEqual(len(catalog.loras), 3)
        for asset in (*catalog.models.values(), *catalog.loras.values()):
            with self.subTest(family=asset.family, kind=asset.kind):
                prefix, identifier = asset.id.split("_", 1)
                self.assertEqual(prefix, asset.kind)
                UUID(identifier)
                self.assertEqual(asset.display_name, "demo")
                self.assertEqual(asset.defaults["family"], asset.manifest_path.parents[2].name.lower())
                self.assertTrue(asset.asset_paths["weights"].exists())
                self.assertEqual(json.loads(asset.user_path.read_text()), {"schema_version": 1, "settings": {}})
                self.assertEqual(asset.defaults["settings"]["trigger_words"], [])
                self.assertEqual(asset.defaults["example_prompts"], [])
                self.assertEqual(asset.defaults["example_negative_prompts"], [])
                if asset.kind == "model":
                    self.assertEqual(asset.defaults["settings"]["loras"], [])
                    self.assertEqual(asset.defaults["settings"]["prompt"], "")
                else:
                    self.assertEqual(asset.defaults["settings"]["recommended_generation"], {})
                    self.assertIsNone(asset.defaults["compatibility"]["expert_target"])
                    self.assertIsNone(asset.defaults["compatibility"]["model_variant"])
                    self.assertEqual(asset.defaults["compatibility"]["verified_checkpoint_ids"], [])
        self.assertEqual(len({asset.id for asset in (*catalog.models.values(), *catalog.loras.values())}), 6)
        self.assertEqual(before_templates, {path: path.read_bytes() for path in self.root.rglob("*.example.json")})

    def test_template_example_prompts_are_cleared_unless_metadata_supplies_them(self):
        for folder, kind in (("checkpoints", "model"), ("loras", "lora")):
            with self.subTest(kind=kind):
                directory = self.asset(folder=folder)
                template_path = directory.parent / f"{kind}.defaults.example.json"
                template = json.loads(template_path.read_text())
                template["example_prompts"] = ["An illustrative template prompt"]
                template["example_negative_prompts"] = ["An illustrative negative prompt"]
                template_path.write_text(json.dumps(template), encoding="utf-8")
                initialize_asset(directory)
                defaults_path = directory / f"{kind}.defaults.json"
                self.assertEqual(json.loads(defaults_path.read_text())["example_prompts"], [])
                self.assertEqual(json.loads(defaults_path.read_text())["example_negative_prompts"], [])

                uploaded = self.asset(folder=folder, name="uploaded")
                examples = ["A prompt for this uploaded asset"]
                negative_examples = ["blurry, low quality"]
                initialize_asset(uploaded, metadata={
                    "example_prompts": examples, "example_negative_prompts": negative_examples,
                })
                uploaded_path = uploaded / f"{kind}.defaults.json"
                self.assertEqual(json.loads(uploaded_path.read_text())["example_prompts"], examples)
                self.assertEqual(json.loads(uploaded_path.read_text())["example_negative_prompts"], negative_examples)

    def test_repeated_loading_preserves_ids_and_all_existing_bytes(self):
        directory = self.asset()
        initialize_asset(directory)
        user = directory / "model.user.json"
        user.write_text('{"schema_version":1,"settings":{"seed":42}}', encoding="utf-8")
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(initialize_asset(directory, metadata={"display_name": "ignored"}), ())
        self.assertEqual(initialize_catalog(self.root), ())
        catalog = discover_catalog(self.root)
        self.assertEqual(next(iter(catalog.models.values())).resolve_settings()["seed"], 42)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_missing_user_file_is_recreated_without_a_template_or_local_weights(self):
        directory = self.asset()
        initialize_asset(directory)
        defaults = directory / "model.defaults.json"
        original = defaults.read_bytes()
        user = directory / "model.user.json"
        user.unlink()
        (directory / "actual-weights.safetensors").unlink()
        (directory.parent / "model.defaults.example.json").unlink()
        (directory.parent / "model.user.example.json").unlink()
        self.assertEqual(initialize_asset(directory), (user,))
        self.assertEqual(defaults.read_bytes(), original)
        self.assertEqual(json.loads(user.read_text())["settings"], {})

    def test_missing_defaults_preserves_existing_user_overrides(self):
        directory = self.asset()
        user = directory / "model.user.json"
        user.write_text('{"schema_version":1,"settings":{"width":768}}', encoding="utf-8")
        original = user.read_bytes()
        catalog = discover_catalog(self.root)
        self.assertEqual(next(iter(catalog.models.values())).resolve_settings()["width"], 768)
        self.assertEqual(user.read_bytes(), original)

    def test_upload_can_choose_weights_and_supply_verified_metadata(self):
        directory = self.asset("WAN", "loras")
        (directory / "second.safetensors").touch()
        created = initialize_asset(directory, weights="second.safetensors", metadata={
            "display_name": "Uploaded adapter",
            "compatibility": {"model_variant": "wan_2_2_i2v_a14b", "expert_target": "low_noise",
                              "verified_checkpoint_ids": []},
            "settings": {"strength": 0.6, "trigger_words": ["documented-trigger"]},
        })
        self.assertEqual(len(created), 2)
        asset = next(iter(discover_catalog(self.root).loras.values()))
        self.assertEqual(asset.display_name, "Uploaded adapter")
        self.assertEqual(asset.asset_paths["weights"], directory / "second.safetensors")
        self.assertEqual(asset.defaults["compatibility"]["expert_target"], "low_noise")
        self.assertEqual(asset.resolve_settings()["trigger_words"], ["documented-trigger"])

    def test_pretrained_upload_and_wan_template_do_not_require_local_weights(self):
        directory = self.root / "SDXL/checkpoints/remote"
        directory.mkdir()
        initialize_asset(directory, metadata={"weights": None, "base_model_id": "organization/pretrained"})
        wan = self.root / "WAN/checkpoints/pretrained"
        wan.mkdir()
        (wan / "model.user.json").write_text('{"schema_version":1,"settings":{}}', encoding="utf-8")
        catalog = discover_catalog(self.root)
        self.assertEqual(len(catalog.models), 2)
        self.assertTrue(all(asset.asset_paths["weights"] is None for asset in catalog.models.values()))

    def test_wan_pipeline_nested_inside_asset_is_detected(self):
        directory = self.asset("WAN")
        pipeline = directory / "pipeline"
        pipeline.mkdir()
        (directory / "model_index.json").rename(pipeline / "model_index.json")
        asset = next(iter(discover_catalog(self.root).models.values()))
        self.assertEqual(asset.asset_paths["weights"], pipeline)

    def test_ambiguous_missing_and_incomplete_weights_report_errors_without_creating_json(self):
        ambiguous = self.asset()
        (ambiguous / "second.safetensors").touch()
        missing = self.root / "FLUX/checkpoints/missing"
        missing.mkdir()
        wan = self.asset("WAN")
        (wan / "model_index.json").unlink()
        (wan / "transformer.SAFETENSORS").touch()
        for directory in (ambiguous, missing, wan):
            with self.subTest(directory=directory), self.assertRaises(CatalogValidationError) as context:
                initialize_asset(directory)
            self.assertEqual(context.exception.issues[0].field, "weights")
            self.assertFalse((directory / "model.defaults.json").exists())
            self.assertFalse((directory / "model.user.json").exists())

    def test_missing_or_invalid_templates_report_the_template_path(self):
        directory = self.asset()
        template = directory.parent / "model.defaults.example.json"
        template.unlink()
        with self.assertRaises(CatalogValidationError) as context:
            initialize_asset(directory)
        self.assertEqual(context.exception.issues[0].path, template)
        template.write_text('{"schema_version":1}', encoding="utf-8")
        with self.assertRaises(CatalogValidationError):
            discover_catalog(self.root)
        self.assertFalse((directory / "model.defaults.json").exists())

    def test_missing_family_templates_skip_assets_and_recover_when_added(self):
        ready = self.asset()
        incomplete = []
        for folder, kind in (("checkpoints", "model"), ("loras", "lora")):
            directory = self.root / "Krea2" / folder / "new-asset"
            directory.mkdir(parents=True)
            (directory / "weights.safetensors").touch()
            # Preserve overrides even if defaults cannot yet be created.
            user = directory / f"{kind}.user.json"
            user.write_text('{"schema_version":1,"settings":{"width":768}}', encoding="utf-8")
            incomplete.append((directory, kind, user.read_bytes()))

        with self.assertLogs("image_generator.catalog.initialization", level="WARNING") as logs:
            catalog = discover_catalog(self.root)
        self.assertEqual(len(catalog.models), 1)
        self.assertEqual(catalog.loras, {})
        self.assertEqual(next(iter(catalog.models.values())).manifest_path.parent, ready)
        self.assertEqual(len(logs.output), 2)
        for directory, kind, original_user in incomplete:
            self.assertTrue(any(str(directory) in message and f"{kind}.defaults.example.json" in message
                                for message in logs.output))
            self.assertFalse((directory / f"{kind}.defaults.json").exists())
            self.assertEqual((directory / f"{kind}.user.json").read_bytes(), original_user)
            shutil.copyfile(
                self.root / "SDXL" / directory.parent.name / f"{kind}.defaults.example.json",
                directory.parent / f"{kind}.defaults.example.json",
            )

        catalog = discover_catalog(self.root)
        self.assertEqual(len(catalog.models), 2)
        self.assertEqual(len(catalog.loras), 1)
        for asset in (*catalog.models.values(), *catalog.loras.values()):
            if asset.family == "krea2":
                self.assertEqual(asset.resolve_settings()["width"], 768)

    def test_incomplete_weights_skip_only_the_unconfigured_assets(self):
        ready = self.asset(name="ready")
        ambiguous = self.asset(name="ambiguous")
        (ambiguous / "second.safetensors").touch()
        wan = self.asset("WAN")
        (wan / "model_index.json").unlink()
        (wan / "transformer.safetensors").touch()
        with self.assertLogs("image_generator.catalog.initialization", level="WARNING") as logs:
            catalog = discover_catalog(self.root)
        self.assertEqual(len(catalog.models), 1)
        self.assertEqual(next(iter(catalog.models.values())).manifest_path.parent, ready)
        self.assertEqual(len(logs.output), 2)
        for directory in (ambiguous, wan):
            self.assertFalse((directory / "model.defaults.json").exists())
            self.assertFalse((directory / "model.user.json").exists())

    def test_unreadable_template_is_an_error_rather_than_an_incomplete_asset(self):
        directory = self.asset()
        template = directory.parent / "model.defaults.example.json"
        template.unlink()
        template.mkdir()
        with self.assertRaises(CatalogValidationError) as context:
            discover_catalog(self.root)
        self.assertEqual(context.exception.issues[0].path, template)

    def test_invalid_existing_json_and_broken_symlinks_are_never_replaced(self):
        directory = self.asset()
        defaults = directory / "model.defaults.json"
        defaults.write_text("{broken", encoding="utf-8")
        user = directory / "model.user.json"
        user.symlink_to(directory / "nonexistent.json")
        self.assertEqual(initialize_asset(directory), ())
        with self.assertRaises(CatalogValidationError):
            discover_catalog(self.root)
        self.assertEqual(defaults.read_text(), "{broken")
        self.assertTrue(user.is_symlink())
        self.assertFalse(user.exists())

    def test_concurrent_initializers_publish_one_complete_pair_with_one_id(self):
        directory = self.asset()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: initialize_asset(directory), range(8)))
        self.assertEqual(sum(len(result) for result in results), 2)
        self.assertEqual(len(discover_catalog(self.root).models), 1)
        self.assertFalse(list(directory.glob(".manifest-*")))

    def test_write_failure_reports_error_without_publishing_partial_json(self):
        directory = self.asset()
        with patch("image_generator.catalog.initialization.os.link", side_effect=PermissionError("read-only directory")):
            with self.assertRaisesRegex(CatalogValidationError, "read-only directory"):
                discover_catalog(self.root)
        self.assertEqual(list(directory.iterdir()), [directory / "actual-weights.safetensors"])

    def test_explicit_read_only_discovery_does_not_initialize_files(self):
        directory = self.asset()
        self.assertEqual(discover_catalog(self.root, initialize_missing=False).models, {})
        self.assertFalse((directory / "model.defaults.json").exists())

    def test_empty_hidden_and_unrelated_folders_are_ignored(self):
        (self.root / "SDXL/checkpoints/empty").mkdir()
        self.asset(name=".hidden")
        directory = self.root / "SDXL/loras/notes"
        directory.mkdir()
        (directory / "README.md").write_text("notes", encoding="utf-8")
        self.assertEqual(initialize_catalog(self.root), ())
        self.assertEqual(discover_catalog(self.root).models, {})
        self.assertEqual(discover_catalog(self.root).loras, {})


if __name__ == "__main__":
    unittest.main()

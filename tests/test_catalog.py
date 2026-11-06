"""Catalog behavior using temporary manifests, without GPU dependencies."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import (
    CatalogValidationError,
    discover_catalog,
    merge_settings,
    resolve_settings,
)


class SettingsTests(unittest.TestCase):
    def test_precedence_and_recursive_partial_override(self):
        result = resolve_settings(
            {"width": 1024, "sampling": {"sampler": "euler", "noise_schedule": "default"}},
            {"width": 768, "sampling": {"sampler": "euler_ancestral"}},
            workflow_defaults={"width": 512, "height": 512, "seed": 7},
            request_overrides={"width": 640, "seed": None},
        )
        self.assertEqual(result, {
            "width": 640, "height": 512, "seed": None,
            "sampling": {"sampler": "euler_ancestral", "noise_schedule": "default"},
        })

    def test_lists_replace_including_empty_lists(self):
        self.assertEqual(
            merge_settings({"loras": [{"id": "a"}]}, {"loras": [{"id": "b"}]}),
            {"loras": [{"id": "b"}]},
        )
        self.assertEqual(merge_settings({"trigger_words": ["a"]}, {"trigger_words": []}), {"trigger_words": []})

    def test_explicit_null_replaces_object_and_can_be_replaced_again(self):
        self.assertEqual(merge_settings({"custom": {"value": 1}}, {"custom": None}), {"custom": None})
        self.assertEqual(merge_settings({"custom": None}, {"custom": {"value": 2}}), {"custom": {"value": 2}})

    def test_result_is_independent_and_removing_override_restores_default(self):
        defaults = {"sampling": {"sampler": "euler"}, "items": [{"nested": [1]}]}
        overrides = {"sampling": {"sampler": "custom"}, "custom": {"nested": [2]}}
        original_defaults, original_overrides = deepcopy(defaults), deepcopy(overrides)
        result = resolve_settings(defaults, overrides)
        result["items"][0]["nested"].append(3)
        result["custom"]["nested"].append(4)
        result["sampling"]["sampler"] = "changed"
        self.assertEqual(defaults, original_defaults)
        self.assertEqual(overrides, original_overrides)
        del overrides["sampling"]
        self.assertEqual(resolve_settings(defaults, overrides)["sampling"]["sampler"], "euler")

    def test_invalid_layer_is_not_silently_ignored(self):
        for invalid in ([], False, "", 0):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                resolve_settings({}, invalid)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = {
            "schema_version": 1, "id": "model-test", "display_name": "Test model",
            "family": "sdxl", "workflow": "sdxl", "weights": "weights.safetensors",
            "supported_modes": ["txt2img"], "requirements": {},
            "settings": {
                "width": 1024, "height": 1024, "seed": 7, "loras": [],
                "sampling": {"sampler": "euler", "noise_schedule": "default"},
            },
        }
        self.lora = {
            "schema_version": 1, "id": "lora-test", "display_name": "Test adapter",
            "family": "sdxl", "adapter_type": "lora", "weights": "adapter.safetensors",
            "compatibility": {"verified_checkpoint_ids": [], "expert_target": None},
            "settings": {"strength": 0.8, "trigger_words": [], "recommended_generation": {"width": 512}},
        }

    def write(self, relative, document):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def write_model(self, **changes):
        return self.write("SDXL/checkpoints/test/model.defaults.json", {**self.model, **changes})

    def write_lora(self, **changes):
        return self.write("SDXL/loras/test/lora.defaults.json", {**self.lora, **changes})

    def assert_invalid(self, path, field):
        with self.assertRaises(CatalogValidationError) as context:
            discover_catalog(self.root)
        self.assertTrue(
            any(issue.path == path and issue.field == field for issue in context.exception.issues),
            str(context.exception),
        )
        self.assertIn(str(path), str(context.exception))
        return context.exception

    def test_discovery_ignores_examples_and_unrelated_files(self):
        self.write_model()
        self.write_lora()
        for relative in (
            "SDXL/checkpoints/model.defaults.example.json",
            "SDXL/checkpoints/test/model.defaults.example.json",
            "SDXL/loras/lora.defaults.example.json",
            "SDXL/checkpoints/model.defaults.json",
            "unrelated/model.defaults.json",
        ):
            self.write(relative, {"invalid": True})
        catalog = discover_catalog(self.root)
        self.assertEqual(list(catalog.models), ["model-test"])
        self.assertEqual(list(catalog.loras), ["lora-test"])
        self.assertIs(catalog.get_asset("model-test"), catalog.get_model("model-test"))
        self.assertIs(catalog.get_asset("lora-test"), catalog.get_lora("lora-test"))
        with self.assertRaises(KeyError):
            catalog.get_model("missing")

    def test_paths_resolve_relative_to_manifest_and_absolute_paths_are_preserved(self):
        absolute = self.root / "shared/preview.png"
        manifest = self.write_model(notes_file="../notes.md", preview_image=str(absolute))
        asset = discover_catalog(self.root).get_model("model-test")
        self.assertEqual(asset.asset_paths, {
            "weights": manifest.parent / "weights.safetensors",
            "notes_file": manifest.parent.parent / "notes.md", "preview_image": absolute,
        })
        self.assertEqual(asset.defaults["weights"], "weights.safetensors")
        self.assertFalse(asset.asset_paths["weights"].exists())

    def test_pretrained_models_and_unknown_compatibility_are_valid(self):
        self.write_model(weights=None, base_model_id="organization/pretrained")
        self.write_lora(compatibility={"model_variant": None, "expert_target": None, "verified_checkpoint_ids": []})
        catalog = discover_catalog(self.root)
        self.assertIsNone(catalog.get_model("model-test").asset_paths["weights"])
        self.assertIsNone(catalog.get_lora("lora-test").defaults["compatibility"]["expert_target"])
        path = self.write_model(weights=None)
        self.assert_invalid(path, "base_model_id")

    def test_missing_user_file_is_initialized_with_no_overrides(self):
        self.write_model()
        before = set(self.root.rglob("*"))
        asset = discover_catalog(self.root).get_model("model-test")
        self.assertEqual(asset.resolve_settings(), self.model["settings"])
        self.assertTrue(asset.user_path.is_file())
        self.assertEqual(json.loads(asset.user_path.read_text()), {"schema_version": 1, "settings": {}})
        self.assertEqual(set(self.root.rglob("*")), before | {asset.user_path})

    def test_read_only_discovery_keeps_missing_user_files_in_memory(self):
        self.write_model()
        asset = discover_catalog(self.root, initialize_missing=False).get_model("model-test")
        self.assertEqual(asset.user_overrides, {"schema_version": 1, "settings": {}})
        self.assertFalse(asset.user_path.exists())

    def test_user_settings_are_resolved_without_changing_sources(self):
        manifest = self.write_model()
        user = self.write("SDXL/checkpoints/test/model.user.json", {
            "schema_version": 1, "settings": {"seed": None, "sampling": {"sampler": "custom"}},
        })
        original_manifest, original_user = manifest.read_bytes(), user.read_bytes()
        asset = discover_catalog(self.root).get_model("model-test")
        result = asset.resolve_settings(workflow_defaults={"custom": "value"}, request_overrides={"width": 512})
        self.assertEqual(result["sampling"], {"sampler": "custom", "noise_schedule": "default"})
        self.assertEqual(result["width"], 512)
        self.assertEqual(result["custom"], "value")
        self.assertIsNone(result["seed"])
        result["sampling"]["sampler"] = "changed"
        self.assertEqual(asset.resolve_settings()["sampling"]["sampler"], "custom")
        self.assertEqual(manifest.read_bytes(), original_manifest)
        self.assertEqual(user.read_bytes(), original_user)

    def test_lora_overrides_are_independent_and_recommendations_are_not_applied(self):
        self.write_model()
        self.write_lora()
        self.write("SDXL/loras/test/lora.user.json", {"schema_version": 1, "settings": {"strength": 0.5}})
        self.write("SDXL/checkpoints/test/model.user.json", {
            "schema_version": 1, "settings": {"loras": [{"id": "lora-test", "enabled": True, "strength": 0.3}]},
        })
        catalog = discover_catalog(self.root)
        self.assertEqual(catalog.get_lora("lora-test").resolve_settings()["strength"], 0.5)
        settings = catalog.get_model("model-test").resolve_settings()
        self.assertEqual(settings["width"], 1024)
        self.assertEqual(settings["loras"][0]["strength"], 0.3)

    def test_unknown_lora_link_reports_source_field_even_when_disabled(self):
        self.write_model()
        user = self.write("SDXL/checkpoints/test/model.user.json", {
            "schema_version": 1, "settings": {"loras": [{"id": "missing", "enabled": False}]},
        })
        self.assert_invalid(user, "settings.loras[0].id")

    def test_duplicate_ids_report_both_paths_without_overwriting(self):
        first = self.write_model()
        second = self.write("SDXL/checkpoints/second/model.defaults.json", self.model)
        with self.assertRaises(CatalogValidationError) as context:
            discover_catalog(self.root)
        self.assertTrue(any(issue.field == "id" for issue in context.exception.issues))
        self.assertIn(str(first), str(context.exception))
        self.assertIn(str(second), str(context.exception))

    def test_ids_are_unique_across_asset_kinds(self):
        self.write_model()
        lora = self.write_lora(id="model-test")
        self.assert_invalid(lora, "id")

    def test_errors_are_aggregated_across_files(self):
        model = self.write_model(schema_version=2, workflow=None)
        lora = self.write_lora(compatibility=[])
        exc = self.assert_invalid(model, "schema_version")
        self.assertIn(str(lora), str(exc))
        self.assertTrue(any(issue.field == "workflow" for issue in exc.issues))

    def test_schema_version_is_required_and_boolean_is_not_version_one(self):
        for version in (None, True, "1", 2):
            with self.subTest(version=version):
                path = self.write_model(schema_version=version)
                self.assert_invalid(path, "schema_version")

    def test_user_files_cannot_override_metadata(self):
        self.write_model()
        for field in (
            "id", "weights", "workflow", "requirements", "supported_modes",
            "compatibility", "example_prompts", "example_negative_prompts",
        ):
            with self.subTest(field=field):
                path = self.write("SDXL/checkpoints/test/model.user.json", {
                    "schema_version": 1, "settings": {}, field: "override",
                })
                self.assert_invalid(path, field)

    def test_example_prompts_are_metadata_for_models_and_loras(self):
        for examples in ([], ["A mountain lake at sunrise", "A quiet village in autumn"]):
            with self.subTest(examples=examples):
                metadata = {"example_prompts": examples, "example_negative_prompts": examples}
                self.write_model(**metadata)
                self.write_lora(**metadata)
                catalog = discover_catalog(self.root)
                for asset in (catalog.get_model("model-test"), catalog.get_lora("lora-test")):
                    for field in metadata:
                        self.assertEqual(asset.defaults[field], examples)
                        self.assertNotIn(field, asset.resolve_settings())

    def test_example_prompts_reject_invalid_arrays_and_entries(self):
        for field in ("example_prompts", "example_negative_prompts"):
            for write in (self.write_model, self.write_lora):
                self.write_model()
                self.write_lora()
                for examples in (None, "A mountain lake", {}, [42], [None], [""], ["   "]):
                    with self.subTest(field=field, kind=write.__name__, examples=examples):
                        path = write(**{field: examples})
                        self.assert_invalid(path, field)

    def test_invalid_user_documents_do_not_fall_back_to_defaults(self):
        self.write_model()
        for invalid in (None, [], "string", {"schema_version": 1, "settings": []}):
            with self.subTest(invalid=invalid):
                user = self.write("SDXL/checkpoints/test/model.user.json", invalid)
                self.assert_invalid(user, "settings" if isinstance(invalid, dict) else "$")

    def test_strict_json_errors_include_filename(self):
        path = self.write_model()
        for invalid in ('{"broken":', '{"id":"a","id":"b"}', '{"x": NaN}', '{"x": 1e999}'):
            with self.subTest(invalid=invalid):
                path.write_text(invalid, encoding="utf-8")
                self.assert_invalid(path, "$")
        path.write_bytes(b"\xff")
        self.assert_invalid(path, "$")

    def test_invalid_user_json_is_reported(self):
        self.write_model()
        user = self.write("SDXL/checkpoints/test/model.user.json", {})
        user.write_text("{", encoding="utf-8")
        self.assert_invalid(user, "$")

    def test_unreadable_user_file_does_not_fall_back_to_defaults(self):
        self.write_model()
        # A directory reliably produces a read error even when tests run as root.
        user = self.root / "SDXL/checkpoints/test/model.user.json"
        user.mkdir()
        self.assert_invalid(user, "$")

    def test_default_manifest_must_be_an_object_with_required_fields(self):
        for document, field in (
            (None, "$"), ([], "$"),
            ({**self.model, "id": ""}, "id"),
            ({**self.model, "settings": None}, "settings"),
            ({**self.model, "supported_modes": []}, "supported_modes"),
            ({**self.model, "requirements": []}, "requirements"),
            ({**self.model, "weights": 42}, "weights"),
        ):
            with self.subTest(document=document):
                path = self.write("SDXL/checkpoints/test/model.defaults.json", document)
                self.assert_invalid(path, field)

    def test_duplicate_lora_links_are_rejected(self):
        link = {"id": "lora-test", "enabled": False}
        path = self.write_model(settings={"loras": [link, link]})
        self.write_lora()
        self.assert_invalid(path, "settings.loras[1].id")

    def test_invalid_effective_request_settings_are_rejected(self):
        self.write_model()
        asset = discover_catalog(self.root).get_model("model-test")
        with self.assertRaises(CatalogValidationError) as context:
            asset.resolve_settings(request_overrides={"width": None})
        self.assertEqual(context.exception.issues[0].field, "effective_settings.width")

    def test_settings_structure_rejects_invalid_shared_types(self):
        self.write_model()
        for settings, field in (
            ({"sampling": "euler"}, "settings.sampling"),
            ({"sampling": {"sampler": None}}, "settings.sampling.sampler"),
            ({"width": True}, "settings.width"),
            ({"seed": "random"}, "settings.seed"),
            ({"guidance_scale": None}, "settings.guidance_scale"),
            ({"trigger_words": "trigger"}, "settings.trigger_words"),
            ({"loras": {}}, "settings.loras"),
            ({"loras": [{"id": "a", "enabled": 1}]}, "settings.loras[0].enabled"),
            ({"loras": [{"id": "a", "enabled": False, "strength": True}]}, "settings.loras[0].strength"),
            ({"recommended_generation": []}, "settings.recommended_generation"),
        ):
            with self.subTest(settings=settings):
                user = self.write("SDXL/checkpoints/test/model.user.json", {"schema_version": 1, "settings": settings})
                self.assert_invalid(user, field)

    def test_new_workflow_fields_are_preserved(self):
        self.write_model(workflow="future-workflow", requirements={"new_requirement": True})
        self.write("SDXL/checkpoints/test/model.user.json", {
            "schema_version": 1, "settings": {"new_control": {"enabled": True, "optional": None}},
        })
        asset = discover_catalog(self.root).get_model("model-test")
        self.assertEqual(asset.resolve_settings()["new_control"], {"enabled": True, "optional": None})

    def test_missing_root_fails_and_empty_root_is_valid(self):
        catalog = discover_catalog(self.root)
        self.assertEqual(catalog.models, {})
        self.assertEqual(catalog.loras, {})
        with self.assertRaises(CatalogValidationError):
            discover_catalog(self.root / "missing")

    def test_project_manifests_are_discoverable_without_loading_weights(self):
        catalog = discover_catalog(ROOT / "Models/Model_Families")
        self.assertTrue(catalog.models)
        self.assertTrue(catalog.loras)
        for asset in (*catalog.models.values(), *catalog.loras.values()):
            self.assertEqual(asset.id, asset.defaults["id"])
            self.assertIsInstance(asset.resolve_settings(), dict)


if __name__ == "__main__":
    unittest.main()

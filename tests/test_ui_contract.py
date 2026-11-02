"""Workflow UI definitions are typed, constrained, and usable without model libraries."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import discover_catalog
from image_generator.workflows.registry import create_workflow
from image_generator.workflows.ui import FormValidationError

FIXTURES = ROOT / "tests/fixtures/model_families"


class UIContractTests(unittest.TestCase):
    def setUp(self):
        self.catalog = discover_catalog(FIXTURES, initialize_missing=False)

    def definition(self, family):
        model = deepcopy(self.catalog.get_model(f"fixture-{family}"))
        workflow = create_workflow(family)
        return workflow, model, workflow.get_ui_definition(model, workflow.ui_modes[0])

    def test_every_workflow_setting_has_a_control_or_lora_selector(self):
        for family in ("sdxl", "flux", "qwen_2_1", "krea2", "wan", "minimax"):
            with self.subTest(family=family):
                workflow, model, ui = self.definition(family)
                names = [field.name for field in ui.fields]
                self.assertEqual(len(names), len(set(names)))
                settings = {field.name.split(".")[0] for field in ui.fields if field.scope == "settings"}
                self.assertEqual(settings, workflow.defaults.keys() - {"loras"})
                self.assertEqual(ui.supported_modes, workflow.ui_modes)
                self.assertNotIn("values", vars(ui))
                resolved = model.resolve_settings(workflow_defaults=workflow.defaults,
                                                  request_overrides={"prompt": "A mountain lake"})
                workflow.validate_form_settings(model, ui.mode, resolved)
                self.assertIsNone(workflow._pipeline)

    def test_sampler_choices_are_restricted_by_checkpoint_requirements(self):
        _, _, ui = self.definition("sdxl")
        fields = {field.name: field for field in ui.fields}
        self.assertEqual(dict(fields["sampling.sampler"].choices), {"euler": "Euler", "euler_ancestral": "Euler ancestral"})
        self.assertEqual(fields["width"].step, 16)
        self.assertTrue(fields["clip_skip"].nullable)

    def test_flux_sequence_limit_comes_from_manifest(self):
        _, _, ui = self.definition("flux")
        fields = {field.name: field for field in ui.fields}
        self.assertEqual(fields["max_sequence_length"].maximum, 256)
        self.assertEqual(fields["true_cfg_scale"].minimum, 1)
        self.assertNotIn("clip_skip", fields)

    def test_wan_combines_frame_constraints_and_exposes_input_separately(self):
        workflow, model, _ = self.definition("wan")
        model.defaults["requirements"]["frame_count"] = {"minimum": 10, "step": 8, "offset": 5}
        ui = workflow.get_ui_definition(model, "img2vid")
        fields = {field.name: field for field in ui.fields}
        self.assertEqual((fields["num_frames"].minimum, fields["num_frames"].step), (13, 8))
        self.assertEqual(fields["input_image"].scope, "inputs")
        self.assertTrue(fields["input_image"].required)
        self.assertEqual(ui.partial, "controls/wan.html")
        for count in range(1, 100):
            accepted = fields["num_frames"].validate(count) is None
            self.assertEqual(accepted, count >= 10 and (count - 5) % 8 == 0 and (count - 1) % 4 == 0)

    def test_invalid_manifest_constraints_make_the_form_unavailable(self):
        for family, requirements in (
            ("sdxl", {"dimension_multiple": 0}),
            ("sdxl", {"allowed_sampling": []}),
            ("flux", {"max_sequence_length_limit": 513}),
            ("wan", {"frame_count": {"minimum": 5, "step": 4, "offset": 2}}),
        ):
            with self.subTest(family=family, requirements=requirements):
                workflow, model, ui = self.definition(family)
                model.defaults["requirements"].update(requirements)
                with self.assertRaises(ValueError):
                    workflow.get_ui_definition(model, ui.mode)

    def test_strict_types_and_cross_field_validation(self):
        workflow, model, ui = self.definition("wan")
        original = model.resolve_settings(workflow_defaults=workflow.defaults, request_overrides={"prompt": "A lake"})
        for override, field in (
            ({"num_frames": True}, "num_frames"),
            ({"seed": 2**63}, "seed"),
            ({"fps": float("nan")}, "fps"),
            ({"fps": 0}, "fps"),
            ({"preserve_aspect_ratio": 1}, "preserve_aspect_ratio"),
            ({"guidance_scale_2": 0.5}, "guidance_scale_2"),
        ):
            with self.subTest(override=override), self.assertRaises(FormValidationError) as context:
                workflow.validate_form_settings(model, ui.mode, {**deepcopy(original), **override})
            self.assertIn(field, context.exception.errors)

    def test_only_implemented_and_declared_modes_are_offered(self):
        workflow, model, _ = self.definition("sdxl")
        model.defaults["supported_modes"] = ["img2img", "txt2img"]
        self.assertEqual(workflow.get_ui_definition(model, "txt2img").supported_modes, ("txt2img",))
        with self.assertRaises(ValueError):
            workflow.get_ui_definition(model, "img2img")

    def test_contract_imports_and_resolution_work_without_site_packages(self):
        code = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
from image_generator.catalog import discover_catalog
from image_generator.workflows.registry import create_workflow
catalog = discover_catalog({str(FIXTURES)!r}, initialize_missing=False)
for model in catalog.models.values():
    workflow = create_workflow(model.defaults['workflow'])
    ui = workflow.get_ui_definition(model, workflow.ui_modes[0])
    assert ui.fields and workflow._pipeline is None
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'flask', 'transformers'))
"""
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd="/tmp", capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

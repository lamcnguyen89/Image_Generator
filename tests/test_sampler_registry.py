"""Sampler capabilities and checkpoint policy stay available without GPU imports."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.workflows.shared.samplers import (
    available_samplers, create_scheduler, get_sampler, validate_sampling,
)
from image_generator.workflows.shared.samplers.ui import sampling_fields


class SamplerRegistryTests(unittest.TestCase):
    def test_capabilities_and_ui_honor_exact_checkpoint_pairs(self):
        requirements = {"allowed_sampling": [{"sampler": "er_sde", "noise_schedule": "default"}]}
        model = SimpleNamespace(defaults={"workflow": "krea2", "requirements": requirements})
        self.assertEqual([spec.name for spec in available_samplers("krea2", requirements)], ["er_sde"])
        fields = sampling_fields(model)
        self.assertEqual(fields[0].choices, (("er_sde", "ER SDE"),))
        self.assertEqual(fields[1].choices, (("default", "Default"),))
        self.assertEqual(validate_sampling("krea2", requirements["allowed_sampling"][0], requirements).name, "er_sde")
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_sampling("krea2", {"sampler": "flow_match_euler", "noise_schedule": "default"}, requirements)
        requirements["allowed_sampling"][0]["noise_schedule"] = "unimplemented"
        with self.assertRaisesRegex(ValueError, "no sampling options"):
            sampling_fields(model)

    def test_workflow_compatibility_is_explicit(self):
        self.assertEqual([spec.name for spec in available_samplers("krea2", {})],
                         ["flow_match_euler", "er_sde", "model_default"])
        self.assertEqual([spec.name for spec in available_samplers("sdxl", {})],
                         ["model_default", "euler", "euler_ancestral", "dpmpp_2m_sde"])
        with self.assertRaisesRegex(ValueError, "not implemented"):
            get_sampler("sdxl", "er_sde")
        for workflow in ("flux", "qwen_2_1"):
            self.assertEqual([spec.name for spec in available_samplers(workflow, {})],
                             ["flow_match_euler", "model_default"])
            with self.assertRaisesRegex(ValueError, "not implemented"):
                get_sampler(workflow, "er_sde")
        self.assertEqual([spec.name for spec in available_samplers("wan", {})], ["model_default"])
        for workflow in ("minimax", "unknown"):
            self.assertEqual(available_samplers(workflow, {}), ())
            with self.assertRaisesRegex(ValueError, "not implemented"):
                get_sampler(workflow, "er_sde")

    def test_invalid_settings_fail_before_factory_import(self):
        for sampling in ({}, [], {"sampler": "er_sde"},
                         {"sampler": [], "noise_schedule": "default"},
                         {"sampler": "er_sde", "noise_schedule": []},
                         {"sampler": "er_sde", "noise_schedule": "unknown_schedule"},
                         {"sampler": "os.system", "noise_schedule": "default"}):
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                validate_sampling("krea2", sampling, {})
        sampling = {"sampler": "er_sde", "noise_schedule": "default"}
        for requirements in ({"allowed_sampling": {}}, {"allowed_sampling": ["er_sde"]},
                             {"allowed_sampling": [{"sampler": "er_sde"}]},
                             {"scheduler_class": "EulerDiscreteScheduler"}, {"scheduler_config": []},
                             {"scheduler_config": {"invert_sigmas": True}},
                             {"scheduler_config": {"stochastic_sampling": True}}):
            with self.subTest(requirements=requirements), self.assertRaises(ValueError):
                validate_sampling("krea2", sampling, requirements)

    def test_euler_and_legacy_alias_build_fresh_schedulers_without_mutating_config(self):
        class Scheduler:
            @classmethod
            def from_config(cls, config, **options):
                return SimpleNamespace(config={**deepcopy(config), **options})

        diffusers = SimpleNamespace(FlowMatchEulerDiscreteScheduler=Scheduler)
        original = {"shift": 1, "use_dynamic_shifting": True}
        requirements = {"scheduler_config": {"base_shift": 0.7}}
        before = deepcopy((original, requirements))
        for workflow in ("krea2", "flux", "qwen_2_1"):
            with self.subTest(workflow=workflow):
                first, alias = [create_scheduler(workflow, {"sampler": name, "noise_schedule": "default"},
                                                requirements, diffusers=diffusers, original_config=original, generators=[])
                                for name in ("flow_match_euler", "model_default")]
                self.assertIsNot(first, alias)
                self.assertEqual(first.config, {**original, "base_shift": 0.7})
                self.assertEqual(first.config, alias.config)
                self.assertEqual((original, requirements), before)

    def test_flux_and_qwen_checkpoint_pairs_and_schedule_class_are_enforced(self):
        for workflow in ("flux", "qwen_2_1"):
            with self.subTest(workflow=workflow):
                sampling = {"sampler": "model_default", "noise_schedule": "default"}
                requirements = {"allowed_sampling": [sampling]}
                model = SimpleNamespace(defaults={"workflow": workflow, "requirements": requirements})
                self.assertEqual(sampling_fields(model)[0].choices, (("model_default", "Model default"),))
                validate_sampling(workflow, sampling, requirements)
                with self.assertRaisesRegex(ValueError, "prohibited"):
                    validate_sampling(workflow, {**sampling, "sampler": "flow_match_euler"}, requirements)
                with self.assertRaisesRegex(ValueError, "requires a FlowMatchEulerDiscreteScheduler"):
                    validate_sampling(workflow, sampling, {"scheduler_class": "EulerDiscreteScheduler"})
                with self.assertRaisesRegex(ValueError, "Unsupported noise schedule"):
                    validate_sampling(workflow, {**sampling, "noise_schedule": "unknown_schedule"}, {})
                for sampler in ("euler_ancestral", "dpmpp_2m_sde"):
                    with self.assertRaisesRegex(ValueError, "not implemented"):
                        validate_sampling(workflow, {**sampling, "sampler": sampler}, {})

    def test_registry_and_forms_import_without_runtime_libraries(self):
        code = """
import json, sys
sys.path.insert(0, sys.argv[1])
from image_generator.workflows.shared.samplers import available_samplers
from image_generator.workflows.shared.samplers.ui import sampling_fields
from image_generator.workflows.krea2.definition import DEFINITION
from image_generator.workflows.sdxl.definition import DEFINITION as SDXL
from image_generator.workflows.flux.definition import DEFINITION as FLUX
from image_generator.workflows.qwen21.definition import DEFINITION as QWEN
print(json.dumps([spec.name for spec in available_samplers('krea2', {})]))
assert 'torch' not in sys.modules and 'diffusers' not in sys.modules
assert 'image_generator.workflows.shared.samplers.er_sde' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT / "src")],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("er_sde", json.loads(result.stdout))

    def test_form_validation_preserves_extensions_but_execution_remains_strict(self):
        pair = {"sampler": "flow_match_euler", "noise_schedule": "default"}
        sampling = {**pair, "custom": {"x": 1}}
        before = deepcopy(sampling)
        requirements = {"allowed_sampling": [pair]}
        validate_sampling("flux", sampling, requirements, allow_extensions=True)
        self.assertEqual(sampling, before)
        with self.assertRaisesRegex(ValueError, "sampling must contain"):
            validate_sampling("flux", sampling, requirements)
        with self.assertRaisesRegex(ValueError, "not implemented"):
            validate_sampling("flux", {**sampling, "sampler": "euler"}, requirements, allow_extensions=True)

    def test_derived_form_can_explicitly_use_its_base_sampler_bindings(self):
        model = SimpleNamespace(defaults={"workflow": "custom_flux", "requirements": {}})
        with self.assertRaisesRegex(ValueError, "no sampling options"):
            sampling_fields(model)
        self.assertEqual(dict(sampling_fields(model, workflow="flux")[0].choices),
                         {"flow_match_euler": "Flow match Euler", "model_default": "Model default"})

    def test_sdxl_sampling_pairs_filter_forms_and_backend_consistently(self):
        requirements = {"allowed_sampling": [
            {"sampler": "euler_ancestral", "noise_schedule": "default"},
            {"sampler": "dpmpp_2m_sde", "noise_schedule": "default"},
        ]}
        model = SimpleNamespace(defaults={"workflow": "sdxl", "requirements": requirements})
        self.assertEqual(dict(sampling_fields(model)[0].choices),
                         {"euler_ancestral": "Euler ancestral", "dpmpp_2m_sde": "DPM++ 2M SDE"})
        for sampling in requirements["allowed_sampling"]:
            validate_sampling("sdxl", sampling, requirements)
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_sampling("sdxl", {"sampler": "euler", "noise_schedule": "default"}, requirements)
        with self.assertRaisesRegex(ValueError, "not implemented"):
            validate_sampling("krea2", requirements["allowed_sampling"][0], {})

    def test_sdxl_public_sampler_export_preserves_json_names(self):
        from image_generator.workflows.sdxl import SAMPLERS

        self.assertEqual(SAMPLERS, {
            "euler": "EulerDiscreteScheduler", "euler_ancestral": "EulerAncestralDiscreteScheduler",
            "dpmpp_2m_sde": "DPMSolverMultistepScheduler",
        })

    def test_karras_honors_native_scheduler_support_and_checkpoint_pairs(self):
        sampling = {"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"}
        validate_sampling("sdxl", sampling, {})
        requirements = {"allowed_sampling": [sampling, {"sampler": "euler", "noise_schedule": "default"}]}
        model = SimpleNamespace(defaults={"workflow": "sdxl", "requirements": requirements})
        noise = sampling_fields(model)[1]
        self.assertEqual(noise.choices, (("default", "Default"), ("karras", "Karras")))
        self.assertEqual(dict(noise.choices_by_source), {"euler": ("default",), "dpmpp_2m_sde": ("karras",)})
        validate_sampling("sdxl", {**sampling, "sampler": "euler"}, {})
        for sampler in ("euler_ancestral", "model_default"):
            with self.subTest(sampler=sampler), self.assertRaisesRegex(ValueError, "Unsupported noise schedule"):
                validate_sampling("sdxl", {**sampling, "sampler": sampler}, {})
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_sampling("sdxl", {**sampling, "noise_schedule": "default"}, requirements)
        for option, value in (("use_karras_sigmas", False), ("use_exponential_sigmas", True),
                              ("use_beta_sigmas", True), ("use_lu_lambdas", True), ("use_flow_sigmas", True)):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "requires scheduler_config"):
                validate_sampling("sdxl", sampling, {"scheduler_config": {option: value}})

    def test_lustify_default_selects_registered_karras_pair(self):
        path = ROOT / "Models/Model_Families/SDXL/checkpoints/lustifyNSFWCheckpoint_ggwpV7/model.defaults.json"
        model = json.loads(path.read_text())
        self.assertEqual(model["settings"]["sampling"], {"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"})
        validate_sampling("sdxl", model["settings"]["sampling"], model["requirements"])

    def test_big_love_default_is_supported_by_its_manifest(self):
        path = ROOT / "Models/Model_Families/Krea2/checkpoints/Big_Love_Krea2/model.defaults.json"
        model = json.loads(path.read_text())
        self.assertEqual(model["settings"]["sampling"]["sampler"], "er_sde")
        validate_sampling("krea2", model["settings"]["sampling"], model["requirements"])


if __name__ == "__main__":
    unittest.main()

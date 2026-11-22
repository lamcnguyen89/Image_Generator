"""Shared schedule discovery, compatibility and configuration without GPU imports."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.workflows.shared.schedules import available_schedules, get_schedule, scheduler_options, validate_schedule
from image_generator.workflows.shared.samplers import validate_sampling
from image_generator.workflows.shared.samplers.ui import sampling_fields


class ScheduleRegistryTests(unittest.TestCase):
    def test_profile_capabilities_and_shared_labels(self):
        for profile in ("flow_match", "euler_discrete", "dpm_multistep"):
            self.assertEqual([(spec.name, spec.label) for spec in available_schedules(profile)],
                             [("default", "Default"), ("normal", "Normal"), ("karras", "Karras"),
                              ("exponential", "Exponential"), ("beta", "Beta")])
        for profile in ("model_default",):
            self.assertEqual([spec.name for spec in available_schedules(profile)], ["default"])
            with self.assertRaisesRegex(ValueError, "Unsupported noise schedule"):
                get_schedule(profile, "karras")
        self.assertEqual([spec.name for spec in available_schedules("euler_ancestral")], ["default", "normal"])
        self.assertEqual(get_schedule("comfyui_minimax_h3", "simple").native_name, "simple")
        self.assertEqual(available_schedules("unknown"), ())
        for name in (None, [], "unknown", "os.system"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                get_schedule("flow_match", name)

    def test_curve_options_are_independent_and_reject_checkpoint_conflicts(self):
        for name in ("karras", "exponential", "beta"):
            requirements = {"scheduler_config": {"prediction_type": "v_prediction"}}
            before = deepcopy(requirements)
            options = scheduler_options("dpm_multistep", name, requirements)
            for curve in ("karras", "exponential", "beta"):
                self.assertEqual(options[f"use_{curve}_sigmas"], curve == name)
            self.assertFalse(options["use_lu_lambdas"])
            self.assertFalse(options["use_flow_sigmas"])
            with self.assertRaisesRegex(ValueError, "requires scheduler_config"):
                validate_schedule("dpm_multistep", name, {"scheduler_config": {f"use_{name}_sigmas": False}})
            options.clear()
            self.assertTrue(scheduler_options("dpm_multistep", name, {})[f"use_{name}_sigmas"])
            self.assertEqual(requirements, before)
        # Default inherits the source configuration, including an existing curve.
        self.assertEqual(scheduler_options("dpm_multistep", "default", {"scheduler_config": {"use_karras_sigmas": True}}), {})
        with self.assertRaisesRegex(ValueError, "must be an object"):
            validate_schedule("flow_match", "beta", {"scheduler_config": []})

    def test_ui_and_validation_use_exact_registered_checkpoint_pairs(self):
        pairs = [{"sampler": "euler", "noise_schedule": "exponential"},
                 {"sampler": "dpmpp_2m_sde", "noise_schedule": "beta"}]
        requirements = {"allowed_sampling": pairs}
        model = SimpleNamespace(defaults={"workflow": "sdxl", "requirements": requirements})
        sampler, schedule = sampling_fields(model)
        self.assertEqual([name for name, _ in sampler.choices], ["euler", "dpmpp_2m_sde"])
        self.assertEqual(schedule.choices, (("exponential", "Exponential"), ("beta", "Beta")))
        self.assertEqual(dict(schedule.choices_by_source), {"euler": ("exponential",), "dpmpp_2m_sde": ("beta",)})
        for pair in pairs:
            validate_sampling("sdxl", pair, requirements)
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_sampling("sdxl", {"sampler": "euler", "noise_schedule": "beta"}, requirements)

    def test_flow_samplers_share_schedule_definitions_and_video_defaults_stay_pinned(self):
        for workflow, samplers in (("flux", ("flow_match_euler", "model_default")),
                                   ("qwen_2_1", ("flow_match_euler", "model_default")),
                                   ("krea2", ("flow_match_euler", "er_sde", "model_default"))):
            for sampler in samplers:
                for schedule in ("default", "karras", "exponential", "beta"):
                    validate_sampling(workflow, {"sampler": sampler, "noise_schedule": schedule}, {})
        for schedule in ("karras", "exponential", "beta"):
            with self.assertRaisesRegex(ValueError, "Unsupported noise schedule"):
                validate_sampling("wan", {"sampler": "model_default", "noise_schedule": schedule}, {})
            with self.assertRaisesRegex(ValueError, "Unsupported noise schedule"):
                get_schedule("comfyui_minimax_h3", schedule)

    def test_normal_has_its_own_policy_and_clears_curve_overrides(self):
        for backend in ("flow_match", "euler_discrete", "dpm_multistep", "euler_ancestral"):
            binding = get_schedule(backend, "normal")
            self.assertEqual(binding.label, "Normal")
            self.assertIsNotNone(binding.adapter)
            options = scheduler_options(backend, "normal", {})
            if backend != "euler_ancestral":
                for curve in ("karras", "exponential", "beta"):
                    self.assertFalse(options[f"use_{curve}_sigmas"])
            if backend != "flow_match":
                self.assertEqual(options["timestep_spacing"], "linspace")
                with self.assertRaisesRegex(ValueError, "requires scheduler_config"):
                    validate_schedule(backend, "normal", {"scheduler_config": {"timestep_spacing": "trailing"}})
        with self.assertRaisesRegex(ValueError, "Unsupported noise schedule"):
            get_schedule("model_default", "normal")

    def test_discovery_and_forms_work_without_runtime_dependencies(self):
        code = """
import json, sys
sys.path.insert(0, sys.argv[1])
from image_generator.catalog import discover_catalog
from image_generator.workflows.registry import get_definition
from image_generator.workflows.shared.schedules import available_schedules
catalog = discover_catalog(sys.argv[2], initialize_missing=False)
for model in catalog.models.values():
    workflow = get_definition(model.defaults['workflow'])
    workflow.get_ui_definition(model, model.defaults['supported_modes'][0])
print(json.dumps([s.name for s in available_schedules('flow_match')]))
assert 'torch' not in sys.modules and 'diffusers' not in sys.modules
assert 'comfy.samplers' not in sys.modules
assert 'image_generator.workflows.shared.schedules.normal' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT / "src"), str(ROOT / "tests/fixtures/model_families")],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), ["default", "normal", "karras", "exponential", "beta"])


if __name__ == "__main__":
    unittest.main()

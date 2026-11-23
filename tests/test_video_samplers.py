"""WAN solver reference grids and video sampler discovery without model weights."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.workflows.shared.metadata import scheduler_metadata
from image_generator.workflows.shared.samplers import create_scheduler, validate_sampling
from image_generator.workflows.shared.samplers.ui import sampling_fields


class VideoSamplerRegistryTests(unittest.TestCase):
    def test_exact_pairs_filter_both_video_workflows(self):
        for workflow, sampler, schedule in (("wan", "dpmpp_2m", "default"), ("minimax", "euler", "simple")):
            pair = {"sampler": sampler, "noise_schedule": schedule}
            requirements = {"allowed_sampling": [pair]}
            model = SimpleNamespace(defaults={"workflow": workflow, "requirements": requirements})
            fields = sampling_fields(model)
            self.assertEqual([name for name, _ in fields[0].choices], [sampler])
            self.assertEqual([name for name, _ in fields[1].choices], [schedule])
            validate_sampling(workflow, pair, requirements)
            for invalid in (("karras", "beta", "exponential") if workflow == "wan" else ("default", "sgm_uniform")):
                with self.assertRaisesRegex(ValueError, "Unsupported noise schedule"):
                    validate_sampling(workflow, {**pair, "noise_schedule": invalid}, {})
            with self.assertRaisesRegex(ValueError, "prohibited"):
                default = "model_default" if workflow == "wan" else "res_multistep"
                validate_sampling(workflow, {**pair, "sampler": default}, requirements)

    def test_comfy_sampler_construction_uses_registered_native_name(self):
        nodes = SimpleNamespace(KSamplerSelect=SimpleNamespace(execute=Mock(return_value=("native sampler",))))
        for name in ("res_multistep", "euler"):
            self.assertEqual(create_scheduler("minimax", {"sampler": name, "noise_schedule": "simple"}, {}, nodes=nodes),
                             "native sampler")
            nodes.KSamplerSelect.execute.assert_called_with(sampler_name=name)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"),
                     "WAN scheduler checks require Torch and Diffusers")
class WanSamplerNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, diffusers
        import torch
        import diffusers

    def source_config(self):
        return dict(diffusers.UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True,
                                                     flow_shift=3.0, solver_order=2, solver_type="bh2").config)

    def build(self, sampler, config, requirements=None, original_class=None, *, schedule="default"):
        return create_scheduler("wan", {"sampler": sampler, "noise_schedule": schedule}, requirements or {},
                                diffusers=diffusers, original_class=original_class or diffusers.UniPCMultistepScheduler,
                                original_config=config, generators=[torch.Generator().manual_seed(42)])

    def test_explicit_unipc_matches_default_and_dpm_preserves_expert_grid(self):
        config = self.source_config()
        before = deepcopy(config)
        for shift in (3.0, 5.0):
            requirements = {"scheduler_config": {"flow_shift": shift}}
            for steps in (1, 8, 40):
                default = self.build("model_default", config, requirements)
                default.set_timesteps(steps)
                for name in ("uni_pc", "dpmpp_2m"):
                    with self.subTest(sampler=name, shift=shift, steps=steps):
                        scheduler = self.build(name, config, requirements)
                        scheduler.set_timesteps(steps)
                        torch.testing.assert_close(scheduler.sigmas, default.sigmas, rtol=0, atol=0)
                        torch.testing.assert_close(scheduler.timesteps, default.timesteps, rtol=0, atol=0)
                        self.assertEqual((scheduler.timesteps >= 875).tolist(), (default.timesteps >= 875).tolist())
                        self.assertEqual(scheduler.config.flow_shift, shift)
                        if name == "uni_pc":
                            self.assertIsInstance(scheduler, diffusers.UniPCMultistepScheduler)
        self.assertEqual(config, before)

    def test_dpm_steps_match_native_reference_across_expert_boundary_and_reset(self):
        config = self.source_config()
        scheduler = self.build("dpmpp_2m", config)
        scheduler.set_timesteps(8)
        reference = diffusers.DPMSolverMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True,
                                                        algorithm_type="dpmsolver++", solver_order=2)
        reference.set_timesteps(8)
        reference.sigmas, reference.timesteps = scheduler.sigmas.clone(), scheduler.timesteps.clone()
        sample, expected = torch.full((1, 4, 2, 2, 2), 0.3), torch.full((1, 4, 2, 2, 2), 0.3)
        experts = set()
        for t in scheduler.timesteps:
            high_noise = bool(t >= 875)
            experts.add(high_noise)
            scale = 0.2 if high_noise else 0.1
            sample = scheduler.step(sample.tanh() * scale, t, sample, return_dict=False)[0]
            expected = reference.step(expected.tanh() * scale, t, expected).prev_sample
            torch.testing.assert_close(sample, expected, rtol=0, atol=0)
        self.assertEqual(experts, {True, False})
        self.assertTrue(torch.isfinite(sample).all())
        metadata = scheduler_metadata(scheduler)
        self.assertEqual(metadata["config"]["algorithm_type"], "dpmsolver++")
        self.assertEqual(metadata["config"]["solver_order"], 2)
        self.assertEqual(metadata["config"]["source_scheduler_class"], "UniPCMultistepScheduler")
        json.dumps(metadata, allow_nan=False)
        scheduler.set_begin_index(2)
        scheduler.set_timesteps(1)
        self.assertIsNone(scheduler.step_index)
        self.assertIsNone(scheduler.begin_index)
        self.assertEqual(scheduler.lower_order_nums, 0)
        self.assertEqual(scheduler.model_outputs, [None, None])

    def test_wrong_source_and_conflicting_flow_settings_fail(self):
        config = self.source_config()
        for sampler in ("uni_pc", "dpmpp_2m"):
            with self.assertRaisesRegex(ValueError, "UniPCMultistepScheduler source"):
                self.build(sampler, config, original_class=diffusers.DDIMScheduler)
            for overrides in ({"prediction_type": "epsilon"}, {"use_flow_sigmas": False}):
                with self.assertRaisesRegex(ValueError, "requires scheduler_config"):
                    self.build(sampler, config, {"scheduler_config": overrides})
            invalid = {**config, "prediction_type": "epsilon"}
            with self.assertRaisesRegex(ValueError, "flow_prediction"):
                self.build(sampler, invalid)
        with self.assertRaisesRegex(ValueError, "requires scheduler_config"):
            self.build("dpmpp_2m", config, {"scheduler_config": {"solver_order": 3}})

    def test_named_wan_schedules_match_training_grid_reference_and_keep_shift(self):
        import numpy as np

        config = self.source_config()
        before = deepcopy(config)
        for schedule in ("normal", "simple"):
            for steps in (1, 7, 20, 1000):
                for shift in (1.0, 3.0, 5.0):
                    if schedule == "normal":
                        inputs = np.linspace(1.0, 1.0 / 1000, steps)
                    else:
                        # Scalar ComfyUI Simple reference: walk backwards
                        # through the full unshifted flow training table.
                        table = [(i + 1) / 1000 for i in range(1000)]
                        inputs = np.array([table[-(1 + int(i * 1000 / steps))] for i in range(steps)])
                    expected = shift * inputs / (1 + (shift - 1) * inputs)
                    expected[0] -= 1e-6
                    expected_times = torch.tensor(expected * 1000, dtype=torch.int64)
                    expected_sigmas = torch.tensor([*expected, 0], dtype=torch.float32)
                    for sampler in ("model_default", "uni_pc", "dpmpp_2m"):
                        with self.subTest(schedule=schedule, steps=steps, shift=shift, sampler=sampler):
                            scheduler = self.build(sampler, config, {"scheduler_config": {"flow_shift": shift}}, schedule=schedule)
                            scheduler.set_timesteps(steps, device="cpu")
                            torch.testing.assert_close(scheduler.sigmas, expected_sigmas, rtol=0, atol=0)
                            torch.testing.assert_close(scheduler.timesteps, expected_times, rtol=0, atol=0)
                            self.assertEqual((scheduler.timesteps >= 875).tolist(), (expected_times >= 875).tolist())
                            if sampler == "dpmpp_2m":
                                self.assertEqual(scheduler.config.source_scheduler_config["schedule_adapter"], schedule)
                            else:
                                self.assertEqual(scheduler.config.schedule_adapter, schedule)
        self.assertEqual(config, before)

    def test_named_wan_schedules_run_on_video_latents_and_reset(self):
        for sampler in ("model_default", "uni_pc", "dpmpp_2m"):
            for schedule in ("normal", "simple"):
                with self.subTest(sampler=sampler, schedule=schedule):
                    scheduler = self.build(sampler, self.source_config(), schedule=schedule)
                    results = []
                    for _ in range(2):
                        scheduler.set_timesteps(8)
                        self.assertIsNone(scheduler.step_index)
                        sample = torch.full((1, 4, 2, 2, 2), 0.3)
                        for t in scheduler.timesteps:
                            scale = 0.2 if t >= 875 else 0.1
                            sample = scheduler.step(sample.tanh() * scale, t, sample).prev_sample
                        self.assertTrue(torch.isfinite(sample).all())
                        results.append(sample)
                    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
                    json.dumps(scheduler_metadata(scheduler), allow_nan=False)
                    for invalid in (0, -1, 1001, True, 2.5):
                        with self.assertRaisesRegex(ValueError, "inference steps"):
                            scheduler.set_timesteps(invalid)
        for schedule in ("normal", "simple"):
            for config in ({"use_karras_sigmas": True}, {"use_exponential_sigmas": True}, {"use_beta_sigmas": True},
                           {"use_flow_sigmas": False}, {"prediction_type": "epsilon"}, {"final_sigmas_type": "sigma_min"}):
                with self.assertRaisesRegex(ValueError, "requires scheduler_config"):
                    validate_sampling("wan", {"sampler": "uni_pc", "noise_schedule": schedule}, {"scheduler_config": config})


if __name__ == "__main__":
    unittest.main()

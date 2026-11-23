"""Numerical and pipeline-interface checks for the additional image samplers."""

from copy import deepcopy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.workflows.shared.metadata import scheduler_metadata
from image_generator.workflows.shared.samplers import available_samplers, create_scheduler, validate_sampling
from image_generator.workflows.shared.schedules import available_schedules


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"),
                     "Image sampler checks require Torch and Diffusers")
class ImageSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, diffusers
        import torch
        import diffusers

    def build(self, workflow, sampler, schedule="default", *, seeds=(42,), config=None, requirements=None):
        if config is None:
            config = dict((diffusers.DDIMScheduler() if workflow == "sdxl" else
                           diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)).config)
        return create_scheduler(workflow, {"sampler": sampler, "noise_schedule": schedule}, requirements or {},
                                diffusers=diffusers, original_class=diffusers.DDIMScheduler,
                                original_config=config,
                                generators=[torch.Generator().manual_seed(seed) for seed in seeds])

    def run_solver(self, scheduler, *, batch=1, native=False, steps=8):
        if native:
            scheduler.set_timesteps(steps, device="cpu")
        else:
            scheduler.set_timesteps(sigmas=torch.linspace(1, 1 / steps, steps).tolist(), mu=0.7, device="cpu")
            scheduler.set_begin_index(0)
        sample = torch.full((batch, 4, 2, 2), 0.3)
        options = {"generator": [torch.Generator().manual_seed(42 + i) for i in range(batch)]} if native and (
            "generator" in inspect.signature(scheduler.step).parameters) else {}
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(sample, timestep) if hasattr(scheduler, "scale_model_input") else sample
            sample = scheduler.step(model_input.tanh() * 0.1, timestep, sample, **options).prev_sample
        self.assertTrue(torch.isfinite(sample).all())
        json.dumps(scheduler_metadata(scheduler), allow_nan=False)
        return sample

    def test_all_sdxl_samplers_and_registered_schedules_run_repeatably(self):
        config = dict(diffusers.DDIMScheduler().config)
        before = deepcopy(config)
        for spec in available_samplers("sdxl", {}):
            for schedule in available_schedules(spec.schedule_backend):
                with self.subTest(sampler=spec.name, schedule=schedule.name):
                    first = self.build("sdxl", spec.name, schedule.name, config=config)
                    second = self.build("sdxl", spec.name, schedule.name, config=config)
                    torch.testing.assert_close(self.run_solver(first, native=True),
                                               self.run_solver(second, native=True), rtol=0, atol=0)
                    if spec.name.startswith("dpmpp_"):
                        self.assertEqual(first.config.solver_order, 3 if "3m" in spec.name else 2)
                        self.assertEqual(first.config.algorithm_type,
                                         "sde-dpmsolver++" if "sde" in spec.name else "dpmsolver++")
        self.assertEqual(config, before)

    def test_all_flow_samplers_keep_shifted_grids_and_finite_outputs(self):
        config = dict(diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True,
                                                               time_shift_type="linear").config)
        before = deepcopy(config)
        for workflow in ("flux", "qwen_2_1", "krea2"):
            for spec in available_samplers(workflow, {}):
                for schedule in available_schedules(spec.schedule_backend):
                    with self.subTest(workflow=workflow, sampler=spec.name, schedule=schedule.name):
                        scheduler = self.build(workflow, spec.name, schedule.name, config=config)
                        self.run_solver(scheduler)
                        reference = self.build(workflow, "flow_match_euler", schedule.name, config=config)
                        reference.set_timesteps(sigmas=torch.linspace(1, 1 / 8, 8).tolist(), mu=0.7, device="cpu")
                        expected_sigmas, expected_times = reference.sigmas.clone(), reference.timesteps.clone()
                        if spec.name == "flow_match_heun":
                            expected_sigmas = torch.cat([expected_sigmas[:1], expected_sigmas[1:-1].repeat_interleave(2), expected_sigmas[-1:]])
                            expected_times = torch.cat([expected_times[:1], expected_times[1:].repeat_interleave(2)])
                        elif spec.name in ("er_sde", "uni_pc", "dpmpp_2m", "dpmpp_2m_sde") and expected_sigmas[0] == 1:
                            expected_sigmas[0] = 1 - scheduler.initial_sigma_epsilon
                            expected_times[0] = expected_sigmas[0] * config["num_train_timesteps"]
                        torch.testing.assert_close(scheduler.sigmas, expected_sigmas, rtol=0, atol=0)
                        torch.testing.assert_close(scheduler.timesteps, expected_times, rtol=0, atol=0)
        self.assertEqual(config, before)

    def test_flow_stochastic_solvers_own_generators_and_preserve_batch_independence(self):
        for workflow in ("flux", "qwen_2_1", "krea2"):
            for sampler in ("er_sde", "dpmpp_2m_sde"):
                with self.subTest(workflow=workflow, sampler=sampler):
                    batch = self.run_solver(self.build(workflow, sampler, seeds=(42, 43)), batch=2)
                    singles = torch.cat([self.run_solver(self.build(workflow, sampler, seeds=(seed,)))
                                         for seed in (42, 43)])
                    torch.testing.assert_close(batch, singles, rtol=0, atol=0)
                    self.assertFalse(torch.equal(batch[0], batch[1]))
                    repeat = self.run_solver(self.build(workflow, sampler, seeds=(42, 43)), batch=2)
                    torch.testing.assert_close(batch, repeat, rtol=0, atol=0)

    def test_flow_heun_performs_predictor_corrector_evaluations(self):
        scheduler = self.build("flux", "flow_match_heun")
        scheduler.set_timesteps(sigmas=[1, 0.7, 0.3], mu=0.7, device="cpu")
        self.assertEqual(len(scheduler.timesteps), 5)
        self.assertEqual(scheduler.order, 1)  # Pipeline progress counts evaluations.
        grid = scheduler.sigmas[[0, 1, 3, 5]].double().tolist()
        expected = 0.3
        for index, (source, target) in enumerate(zip(grid, grid[1:])):
            delta = target - source
            predicted = expected + delta * 0.1 * expected
            expected = predicted if index == 2 else expected + delta * 0.05 * (expected + predicted)
        sample = torch.full((1, 2, 3), 0.3)
        for t in scheduler.timesteps:
            sample = scheduler.step(sample * 0.1, t, sample, return_dict=False)[0]
        torch.testing.assert_close(sample, torch.full_like(sample, expected), rtol=1e-6, atol=1e-7)
        scheduler.set_begin_index(2)
        scheduler.set_timesteps(sigmas=[1.0], mu=0.7)
        self.assertIsNone(scheduler.begin_index)
        self.assertIsNone(scheduler.step_index)
        self.assertIsNone(scheduler.dt)
        self.assertIsNone(scheduler.prev_derivative)
        self.assertIsNone(scheduler.sample)

    def test_native_flow_solvers_match_reference_steps_and_reset_history(self):
        for sampler, cls, options in (
            ("uni_pc", diffusers.UniPCMultistepScheduler, {"solver_order": 2}),
            ("dpmpp_2m", diffusers.DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++"}),
            ("dpmpp_2m_sde", diffusers.DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
        ):
            with self.subTest(sampler=sampler):
                scheduler = self.build("qwen_2_1", sampler)
                scheduler.set_timesteps(sigmas=[1, 0.8, 0.4, 0.2], mu=0.6, device="cpu")
                reference = cls(prediction_type="flow_prediction", use_flow_sigmas=True, **options)
                reference.set_timesteps(4)
                reference.sigmas, reference.timesteps = scheduler.sigmas.clone(), scheduler.timesteps.clone()
                sample, expected = torch.full((1, 2, 3), 0.3), torch.full((1, 2, 3), 0.3)
                generator = torch.Generator().manual_seed(42)
                for t in scheduler.timesteps:
                    sample = scheduler.step(sample.tanh() * 0.1, t, sample).prev_sample
                    kwargs = {"generator": generator} if sampler != "uni_pc" else {}
                    expected = reference.step(expected.tanh() * 0.1, t, expected, **kwargs).prev_sample
                    torch.testing.assert_close(sample, expected, rtol=0, atol=0)
                scheduler.set_timesteps(sigmas=[1, 0.5], mu=0.6)
                self.assertIsNone(scheduler.step_index)
                self.assertIsNone(scheduler.begin_index)
                self.assertEqual(scheduler.lower_order_nums, 0)
                self.assertTrue(all(value is None for value in scheduler.model_outputs))

    def test_flow_invalid_source_configs_and_grids_fail(self):
        for sampler in ("flow_match_heun", "uni_pc", "dpmpp_2m", "dpmpp_2m_sde"):
            for option in ("invert_sigmas", "stochastic_sampling"):
                with self.subTest(sampler=sampler, option=option), self.assertRaisesRegex(ValueError, "require"):
                    config = dict(diffusers.FlowMatchEulerDiscreteScheduler(**{option: True}).config)
                    self.build("flux", sampler, config=config)
            for sigmas in ([1, 0.5, 0.7], [1, 0.5, 0.5], [1, 0.0], [1, float("nan")]):
                with self.subTest(sampler=sampler, sigmas=sigmas), self.assertRaises(ValueError):
                    self.build("flux", sampler).set_timesteps(sigmas=sigmas, mu=0.7)

    def test_fixed_solver_settings_cannot_be_overridden_by_checkpoint(self):
        for workflow, sampler, overrides in (
            ("sdxl", "dpmpp_2m", {"algorithm_type": "sde-dpmsolver++"}),
            ("sdxl", "dpmpp_3m", {"solver_order": 2}),
            ("flux", "dpmpp_2m_sde", {"prediction_type": "epsilon"}),
            ("qwen_2_1", "uni_pc", {"use_flow_sigmas": False}),
        ):
            with self.subTest(workflow=workflow, sampler=sampler), self.assertRaisesRegex(ValueError, "requires scheduler_config"):
                validate_sampling(workflow, {"sampler": sampler, "noise_schedule": "default"},
                                  {"scheduler_config": overrides})
        for sampler in ("heun", "lms", "dpm_2", "dpm_2_ancestral", "deis", "pndm"):
            with self.subTest(sampler=sampler), self.assertRaisesRegex(ValueError, "rescale_betas_zero_snr"):
                validate_sampling("sdxl", {"sampler": sampler, "noise_schedule": "default"},
                                  {"scheduler_config": {"rescale_betas_zero_snr": True}})
            with self.subTest(sampler=sampler), self.assertRaisesRegex(ValueError, "loader's rescale_betas_zero_snr"):
                self.build("sdxl", sampler, config=dict(diffusers.DDIMScheduler(rescale_betas_zero_snr=True).config))

    def test_new_samplers_support_single_step_and_half_precision_flow_latents(self):
        for spec in available_samplers("sdxl", {}):
            with self.subTest(sampler=spec.name):
                self.run_solver(self.build("sdxl", spec.name), native=True, steps=1)
        for sampler in ("flow_match_heun", "uni_pc", "dpmpp_2m", "dpmpp_2m_sde"):
            scheduler = self.build("flux", sampler)
            scheduler.set_timesteps(sigmas=[1.0], mu=0.7)
            sample = torch.full((1, 2, 3), 0.3, dtype=torch.bfloat16)
            for t in scheduler.timesteps:
                sample = scheduler.step(sample * 0.1, t, sample).prev_sample
            self.assertEqual(sample.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(sample).all())

    def test_flow_sde_does_not_fall_back_to_global_random_state(self):
        scheduler = self.build("flux", "dpmpp_2m_sde", seeds=())
        scheduler.set_timesteps(sigmas=[1, 0.5], mu=0.7)
        sample = torch.full((1, 2, 3), 0.3)
        with self.assertRaisesRegex(ValueError, "seeded generators"):
            scheduler.step(sample * 0.1, scheduler.timesteps[0], sample)


if __name__ == "__main__":
    unittest.main()

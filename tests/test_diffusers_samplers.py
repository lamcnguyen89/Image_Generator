"""Exercise registered native SDXL schedulers on CPU without model weights."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.workflows.shared.samplers import create_scheduler


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"),
                     "Native scheduler smoke tests require Torch and Diffusers")
class NativeSDXLSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, diffusers
        import torch
        import diffusers

    def build(self, sampler, config, requirements=None, *, noise_schedule="default"):
        return create_scheduler("sdxl", {"sampler": sampler, "noise_schedule": noise_schedule}, requirements or {},
                                diffusers=diffusers, original_class=diffusers.DDIMScheduler,
                                original_config=config, generators=[])

    def test_real_samplers_produce_finite_seeded_steps(self):
        original = dict(diffusers.DDIMScheduler().config)
        before = deepcopy(original)
        for sampler, schedule, scheduler_class in (
            ("euler_ancestral", "default", diffusers.EulerAncestralDiscreteScheduler),
            ("dpmpp_2m_sde", "default", diffusers.DPMSolverMultistepScheduler),
            ("dpmpp_2m_sde", "karras", diffusers.DPMSolverMultistepScheduler),
        ):
            results = []
            for seed in (42, 42, 43):
                scheduler = self.build(sampler, original, noise_schedule=schedule)
                self.assertIsInstance(scheduler, scheduler_class)
                scheduler.set_timesteps(4, device="cpu")
                if schedule == "karras":
                    self.assertTrue(scheduler.config.use_karras_sigmas)
                    # Karras uses uniform spacing in sigma**(1/rho), rho=7.
                    gaps = scheduler.sigmas[:-1].double().pow(1 / 7).diff()
                    torch.testing.assert_close(gaps, gaps[0].expand_as(gaps), rtol=1e-5, atol=1e-7)
                generator = torch.Generator().manual_seed(seed)
                sample = torch.full((1, 4, 2, 2), 0.3)
                for timestep in scheduler.timesteps:
                    model_input = scheduler.scale_model_input(sample, timestep)
                    prediction = model_input.tanh() * 0.1
                    sample = scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
                self.assertTrue(torch.isfinite(sample).all())
                results.append(sample)
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
            self.assertFalse(torch.equal(results[0], results[2]))
            if sampler == "dpmpp_2m_sde":
                self.assertEqual(scheduler.config.algorithm_type, "sde-dpmsolver++")
                self.assertEqual(scheduler.config.solver_order, 2)
        self.assertEqual(original, before)

    def test_karras_switching_uses_original_schedule_and_resets_alternative_flags(self):
        original = dict(diffusers.DPMSolverMultistepScheduler(use_exponential_sigmas=True).config)
        before = deepcopy(original)
        karras = self.build("dpmpp_2m_sde", original, noise_schedule="karras")
        self.assertTrue(karras.config.use_karras_sigmas)
        self.assertFalse(karras.config.use_exponential_sigmas)
        default = self.build("dpmpp_2m_sde", original)
        self.assertFalse(default.config.use_karras_sigmas)
        self.assertTrue(default.config.use_exponential_sigmas)
        self.assertIsNot(karras, default)
        self.assertEqual(original, before)

    def test_original_scheduler_and_checkpoint_options_survive_switching(self):
        original = dict(diffusers.DDIMScheduler().config)
        requirements = {"scheduler_config": {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True}}
        for sampler in ("dpmpp_2m_sde", "euler_ancestral", "model_default"):
            scheduler = self.build(sampler, original, requirements)
            self.assertEqual(scheduler.config.prediction_type, "v_prediction")
            self.assertTrue(scheduler.config.rescale_betas_zero_snr)
        self.assertIsInstance(scheduler, diffusers.DDIMScheduler)
        self.assertNotIn("algorithm_type", scheduler.config)
        self.assertNotIn("solver_order", scheduler.config)

    def test_flow_bindings_preserve_dynamic_schedule_and_alias_results(self):
        original = dict(diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True).config)
        before = deepcopy(original)
        requirements = {"scheduler_config": {"base_shift": 0.7}}
        for workflow in ("flux", "qwen_2_1"):
            results = []
            for sampler in ("flow_match_euler", "model_default"):
                scheduler = create_scheduler(
                    workflow, {"sampler": sampler, "noise_schedule": "default"}, requirements,
                    diffusers=diffusers, original_config=original, generators=[],
                )
                self.assertIsInstance(scheduler, diffusers.FlowMatchEulerDiscreteScheduler)
                self.assertTrue(scheduler.config.use_dynamic_shifting)
                self.assertEqual(scheduler.config.base_shift, 0.7)
                scheduler.set_timesteps(sigmas=[1, 0.7, 0.3], mu=0.6, device="cpu")
                sample = torch.full((1, 2, 3), 0.3)
                for timestep in scheduler.timesteps:
                    sample = scheduler.step(sample.tanh() * 0.1, timestep, sample).prev_sample
                self.assertTrue(torch.isfinite(sample).all())
                results.append(sample)
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        self.assertEqual(original, before)

    def test_registered_curves_run_and_record_actual_sigmas_across_native_and_flow_backends(self):
        from image_generator.workflows.shared.metadata import scheduler_metadata
        import json

        native_config = dict(diffusers.DDIMScheduler().config)
        flow_config = dict(diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True).config)
        cases = (("sdxl", "euler", native_config), ("sdxl", "dpmpp_2m_sde", native_config),
                 ("flux", "flow_match_euler", flow_config), ("qwen_2_1", "flow_match_euler", flow_config),
                 ("krea2", "er_sde", flow_config))
        for workflow, sampler, original in cases:
            for schedule in ("default", "normal", "karras", "exponential", "beta"):
                with self.subTest(workflow=workflow, sampler=sampler, schedule=schedule):
                    generator = torch.Generator().manual_seed(42)
                    scheduler = create_scheduler(
                        workflow, {"sampler": sampler, "noise_schedule": schedule}, {}, diffusers=diffusers,
                        original_class=diffusers.DDIMScheduler, original_config=original, generators=[generator],
                    )
                    if workflow == "sdxl":
                        scheduler.set_timesteps(8, device="cpu")
                    else:
                        scheduler.set_timesteps(sigmas=[1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125], mu=0.7, device="cpu")
                    self.assertTrue(torch.isfinite(scheduler.sigmas).all())
                    self.assertTrue((scheduler.sigmas[:-1] >= scheduler.sigmas[1:]).all())
                    sample = torch.full((1, 4, 2, 2), 0.3)
                    for timestep in scheduler.timesteps:
                        model_input = scheduler.scale_model_input(sample, timestep) if hasattr(scheduler, "scale_model_input") else sample
                        sample = scheduler.step(model_input.tanh() * 0.1, timestep, sample, generator=generator).prev_sample
                    self.assertTrue(torch.isfinite(sample).all())
                    metadata = scheduler_metadata(scheduler)
                    self.assertEqual(metadata["sigmas"], scheduler.sigmas.tolist())
                    self.assertEqual(metadata["timesteps"], scheduler.timesteps.tolist())
                    json.dumps(metadata, allow_nan=False)
                    if schedule != "default":
                        for curve in ("karras", "exponential", "beta"):
                            self.assertEqual(scheduler.config[f"use_{curve}_sigmas"], curve == schedule)
                    if schedule == "exponential" and sampler != "er_sde":
                        gaps = scheduler.sigmas[:-1].double().log().diff()
                        torch.testing.assert_close(gaps, gaps[0].expand_as(gaps), rtol=1e-5, atol=1e-6)

    def test_normal_discrete_grid_matches_reference_and_differs_from_dpm_default(self):
        import inspect
        original = dict(diffusers.DDIMScheduler().config)
        for sampler in ("euler", "euler_ancestral", "dpmpp_2m_sde"):
            with self.subTest(sampler=sampler):
                scheduler = self.build(sampler, original, noise_schedule="normal")
                scheduler.set_timesteps(8, device="cpu")
                training = ((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod).sqrt().cpu()
                # Scalar reference for ComfyUI normal_scheduler with this table:
                # uniform training coordinates, log-sigma interpolation, zero.
                expected = []
                for position in torch.linspace(len(training) - 1, 0, 8):
                    low, high = int(position.floor()), int(position.ceil())
                    fraction = float(position - low)
                    expected.append(((1 - fraction) * training[low].log() + fraction * training[high].log()).exp())
                expected.append(torch.tensor(0.))
                torch.testing.assert_close(scheduler.sigmas, torch.stack(expected))
                self.assertEqual(len(scheduler.timesteps), 8)
                self.assertEqual(int(scheduler.timesteps[-1]), 0)
                self.assertIn("generator", inspect.signature(scheduler.step).parameters)
                if sampler == "dpmpp_2m_sde":
                    default = self.build(sampler, original)
                    default.set_timesteps(8)
                    self.assertFalse(torch.equal(default.sigmas, scheduler.sigmas))
                sample = torch.full((1, 4, 2, 2), 0.3)
                for timestep in scheduler.timesteps:
                    model_input = scheduler.scale_model_input(sample, timestep)
                    sample = scheduler.step(model_input * 0.1, timestep, sample,
                                            generator=torch.Generator().manual_seed(42)).prev_sample
                self.assertTrue(torch.isfinite(sample).all())
                # A new run must clear history and restore the complete grid.
                scheduler.set_timesteps(1)
                self.assertEqual(len(scheduler.sigmas), 2)
                self.assertIsNone(scheduler.step_index)

    def test_flow_normal_spans_training_range_and_keeps_model_shift(self):
        original = dict(diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True).config)
        scheduler = create_scheduler("flux", {"sampler": "flow_match_euler", "noise_schedule": "normal"}, {},
                                     diffusers=diffusers, original_config=original,
                                     generators=[torch.Generator().manual_seed(42)])
        scheduler.set_timesteps(sigmas=[1, 0.8, 0.6, 0.4], mu=0.7)
        reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(original)
        reference.set_timesteps(4, mu=0.7)
        torch.testing.assert_close(scheduler.sigmas, reference.sigmas)
        torch.testing.assert_close(scheduler.timesteps, reference.timesteps)
        pipeline_default = diffusers.FlowMatchEulerDiscreteScheduler.from_config(original)
        pipeline_default.set_timesteps(sigmas=[1, 0.75, 0.5, 0.25], mu=0.7)
        self.assertFalse(torch.equal(scheduler.sigmas, pipeline_default.sigmas))
        with self.assertRaisesRegex(ValueError, "positive inference"):
            scheduler.set_timesteps(0)
        with self.assertRaisesRegex(ValueError, "custom timesteps"):
            scheduler.set_timesteps(timesteps=[1000, 500])


if __name__ == "__main__":
    unittest.main()

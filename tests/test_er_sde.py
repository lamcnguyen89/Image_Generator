"""CPU numerical checks for ER-SDE; no checkpoint weights or network needed."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"),
                     "ER-SDE numerical tests require Torch and Diffusers")
class ERSDENumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, FlowMatchEulerDiscreteScheduler, FlowMatchERSDEScheduler
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from image_generator.workflows.shared.samplers.er_sde import FlowMatchERSDEScheduler

    def scheduler(self, seeds=(42,), **config):
        return FlowMatchERSDEScheduler(FlowMatchEulerDiscreteScheduler(**config),
                                      generators=[torch.Generator().manual_seed(seed) for seed in seeds])

    def test_matches_comfyui_reference_at_every_order_and_terminal_step(self):
        # Frozen outputs from ComfyUI sample_er_sde at commit
        # fd7639f477050e5c25b68fc38a9c2a6b6d988211, flow prediction mode.
        # Model: x0=tanh(x/4)+sigma/20; fixed injected noise below. This exercises
        # first, second, third order, alpha scaling and the zero-noise endpoint.
        expected = torch.tensor([
            [[0.18993321, -0.31999296, 0.52947533], [0.22022109, 0.41499811, -0.05502860]],
            [[0.20371331, -0.33123285, 0.55988365], [0.25235009, 0.44897458, -0.05636784]],
            [[0.15737137, -0.22720551, 0.41309947], [0.19796854, 0.33660644, -0.03066922]],
            [[0.08867349, -0.11044984, 0.22096126], [0.11100245, 0.18212907, -0.00893463]],
            [[0.02716474, -0.02260544, 0.06018419], [0.03274349, 0.05050083, 0.00276635]],
        ])
        scheduler = self.scheduler((42, 43))
        scheduler.set_timesteps(sigmas=[0.9999, 0.8, 0.6, 0.3, 0.1])
        sample = torch.tensor([[0.4, -0.2, 0.8], [-0.6, 0.1, 0.3]])
        noise = torch.tensor([[0.2, -0.4, 0.6], [0.3, 0.5, -0.1]])
        with patch("image_generator.workflows.shared.samplers.er_sde.randn_tensor", return_value=noise) as draw:
            for index, timestep in enumerate(scheduler.timesteps):
                sigma = scheduler.sigmas[index]
                clean = (sample / 4).tanh() + sigma / 20
                velocity = (sample - clean) / sigma
                sample = scheduler.step(velocity, timestep, sample).prev_sample
                torch.testing.assert_close(sample, expected[index], rtol=2e-5, atol=2e-7)
        self.assertEqual(draw.call_count, 4)
        self.assertEqual(scheduler.step_index, 5)

    def run_seeded(self, seeds, *, dtype=None, scheduler=None):
        dtype = dtype or torch.float32
        scheduler = scheduler or self.scheduler(seeds)
        scheduler.set_timesteps(sigmas=[1, 0.8, 0.55, 0.25, 0.1])
        sample = torch.full((len(seeds), 4, 8), 0.3, dtype=dtype)
        for timestep in scheduler.timesteps:
            velocity = (sample * 0.2).tanh()
            sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
        return sample

    def test_seed_reproducibility_independence_and_batch_order(self):
        together = self.run_seeded((42, 43))
        torch.testing.assert_close(together, self.run_seeded((42, 43)), rtol=0, atol=0)
        for index, seed in enumerate((42, 43)):
            torch.testing.assert_close(together[index:index + 1], self.run_seeded((seed,)), rtol=0, atol=0)
        torch.testing.assert_close(together.flip(0), self.run_seeded((43, 42)), rtol=0, atol=0)
        self.assertFalse(torch.equal(together[0], together[1]))

    def test_timesteps_reset_history_and_preserve_resolution_shift(self):
        scheduler = self.scheduler(use_dynamic_shifting=True)
        euler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        sigmas = [1, 0.8, 0.55, 0.25, 0.1]
        euler.set_timesteps(sigmas=sigmas, mu=0.7)
        scheduler.set_timesteps(sigmas=sigmas, mu=0.7)
        torch.testing.assert_close(scheduler.sigmas[1:], euler.sigmas[1:])
        self.assertLess(float(scheduler.sigmas[0]), 1)
        torch.testing.assert_close(scheduler.timesteps, scheduler.sigmas[:-1] * scheduler.config.num_train_timesteps)
        sample = torch.ones((1, 4, 8))
        scheduler.step(sample * 0.2, scheduler.timesteps[0], sample)
        self.assertIsNotNone(scheduler._previous_denoised)
        scheduler.set_timesteps(sigmas=sigmas, mu=0.7)
        self.assertIsNone(scheduler._previous_denoised)
        self.assertIsNone(scheduler._previous_derivative)
        self.assertIsNone(scheduler.step_index)

    def test_bfloat16_and_one_step_remain_finite_and_keep_dtype(self):
        result = self.run_seeded((42,), dtype=torch.bfloat16)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.dtype, torch.bfloat16)
        scheduler = self.scheduler()
        scheduler.set_timesteps(sigmas=[1])
        sample = torch.ones((1, 4), dtype=torch.bfloat16)
        velocity = sample / 4
        result = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample
        expected = (sample.float() - scheduler.sigmas[0] * velocity.float()).to(sample.dtype)
        torch.testing.assert_close(result, expected)

    def test_invalid_schedules_fail_clearly(self):
        for sigmas in ([0.8, 0.9], [0.8, 0.8], [0.8, 0], [1.2, 0.5], [0.8, -0.1], [float("nan"), 0.5]):
            with self.subTest(sigmas=sigmas), self.assertRaisesRegex(ValueError, "ER SDE requires"):
                self.scheduler().set_timesteps(sigmas=sigmas)
        scheduler = self.scheduler()
        with self.assertRaisesRegex(ValueError, "set_timesteps"):
            scheduler.step(torch.ones(1, 4), torch.tensor(1000.), torch.ones(1, 4))

    def test_registry_builds_real_adapter_and_rejects_incompatible_loader_config(self):
        import diffusers
        from image_generator.workflows.shared.samplers import create_scheduler

        config = {"use_dynamic_shifting": True, "base_shift": 0.7}
        generators = [torch.Generator().manual_seed(42)]
        scheduler = create_scheduler("krea2", {"sampler": "er_sde", "noise_schedule": "default"}, {},
                                     diffusers=diffusers, original_config=config, generators=generators)
        self.assertIsInstance(scheduler, FlowMatchERSDEScheduler)
        self.assertIs(scheduler._generators, generators)
        self.assertEqual(scheduler.config.base_shift, 0.7)
        self.assertEqual(scheduler.config.solver_order, 3)
        with self.assertRaisesRegex(ValueError, "descending flow schedule"):
            create_scheduler("krea2", {"sampler": "er_sde", "noise_schedule": "default"}, {},
                             diffusers=diffusers, original_config={"invert_sigmas": True}, generators=generators)

    def test_real_krea2_pipeline_runs_seeded_er_sde_without_weights(self):
        from types import SimpleNamespace
        import diffusers
        from diffusers import Krea2Pipeline
        from image_generator.workflows.shared.samplers import create_scheduler

        class TinyTransformer(torch.nn.Module):
            config = SimpleNamespace(in_channels=4)

            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            @property
            def device(self):
                return self.anchor.device

            def forward(self, hidden_states, **options):
                return (hidden_states * 0.1,)

        pipeline = Krea2Pipeline(scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
                                 vae=None, text_encoder=None, tokenizer=None, transformer=TinyTransformer())
        pipeline.set_progress_bar_config(disable=True)
        for schedule in ("default", "normal"):
            results = []
            for _ in range(2):
                generators = [torch.Generator().manual_seed(42)]
                pipeline.scheduler = create_scheduler(
                    "krea2", {"sampler": "er_sde", "noise_schedule": schedule}, {}, diffusers=diffusers,
                    original_config={"use_dynamic_shifting": True}, generators=generators,
                )
                results.append(pipeline(
                    prompt_embeds=torch.zeros((1, 2, 1, 2)), prompt_embeds_mask=torch.ones((1, 2), dtype=torch.bool),
                    height=16, width=16, guidance_scale=0, num_inference_steps=4,
                    generator=generators, output_type="latent",
                ).images)
                self.assertEqual(pipeline.scheduler.step_index, 4)
            self.assertTrue(torch.isfinite(results[0]).all())
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

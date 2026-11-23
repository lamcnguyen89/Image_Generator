"""Native H3 schedule grids and CPU solver steps, without model weights."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.workflows.shared.schedules import get_schedule


@unittest.skipUnless((ROOT / "build/ComfyUI/comfy/samplers.py").is_file()
                     and importlib.util.find_spec("torch") and importlib.util.find_spec("scipy"),
                     "Requires the optional native ComfyUI environment")
class NativeMiniMaxScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch
        import torch

        sys.path.insert(0, str(ROOT / "build/ComfyUI"))
        from comfy.cli_args import args
        previous_cpu = args.cpu
        try:
            args.cpu = True
            import comfy.model_sampling
            import comfy.samplers
            import comfy.k_diffusion.sampling
        finally:
            args.cpu = previous_cpu
        cls.model_sampling = comfy.model_sampling.ModelSamplingAV(
            SimpleNamespace(sampling_settings={"shift": 12.0, "audio_shift": 3.0}))
        cls.calculate_sigmas = staticmethod(comfy.samplers.calculate_sigmas)
        cls.sampling = comfy.k_diffusion.sampling

    def test_native_curves_produce_valid_h3_grids_and_finite_seeded_solver_steps(self):
        model_sampling = self.model_sampling

        class Model:
            inner_model = SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda name: model_sampling))

            def __call__(self, sample, sigma, **options):
                sigma = sigma.reshape(-1, *([1] * (sample.ndim - 1)))
                return sample - sigma * sample.tanh() * 0.1

        for name in ("simple", "normal", "karras", "exponential", "beta"):
            for steps in (1, 20):
                with self.subTest(schedule=name, steps=steps):
                    native = get_schedule("comfyui_minimax_h3", name).native_name
                    sigmas = self.calculate_sigmas(self.model_sampling, native, steps)
                    self.assertTrue(torch.isfinite(sigmas).all())
                    self.assertTrue(((sigmas[:-1] > 0) & (sigmas[:-1] <= 1)).all())
                    self.assertTrue((sigmas[:-1] > sigmas[1:]).all())
                    self.assertEqual(sigmas[-1], 0)
                    if steps > 1 and name in ("karras", "exponential"):
                        values = sigmas[:-1].double()
                        gaps = (values.pow(1 / 7) if name == "karras" else values.log()).diff()
                        torch.testing.assert_close(gaps, gaps[0].expand_as(gaps), rtol=1e-5, atol=1e-6)
                    for sampler in (self.sampling.sample_euler, self.sampling.sample_res_multistep):
                        outputs = [sampler(Model(), torch.full((1, 4, 2, 2, 2), 0.3), sigmas,
                                           extra_args={"seed": 42}, disable=True) for _ in range(2)]
                        self.assertTrue(torch.isfinite(outputs[0]).all())
                        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_native_beta_can_reduce_effective_steps(self):
        sigmas = self.calculate_sigmas(self.model_sampling, "beta", 1000)
        self.assertLess(len(sigmas) - 1, 1000)
        self.assertTrue((sigmas[:-1] > sigmas[1:]).all())


if __name__ == "__main__":
    unittest.main()

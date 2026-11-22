"""Third-order ER-SDE updates for flow predictions, behind a Diffusers interface.

The VP ER-SDE equations use alpha = 1 - sigma and lambda = sigma / alpha.
Flow velocity predicts clean data as x0 = sample - sigma * velocity. The
Taylor solver and noise scaling follow Cui et al., arXiv:2309.06169, using
phi(lambda) = lambda * (exp(lambda**0.3) + 10).

Reference: https://github.com/QinpengCui/ER-SDE-Solver (MIT).
This module is loaded only when its runtime factory is selected.
"""

# ER-SDE reference algorithm notice:
# MIT License
# Copyright (c) 2023 Qinpeng Cui
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
import math

import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass(frozen=True)
class _Coefficients:
    sample: float
    denoised: float
    first: float = 0.0
    second: float = 0.0
    inverse_delta: float = 0.0
    inverse_span: float = 0.0
    noise: float = 0.0


class FlowMatchERSDEScheduler:
    """Use Euler's schedule preparation with a distinct, stateful ER-SDE solver.

    ``order=1`` counts model evaluations per pipeline iteration, not the
    numerical order: the solver ramps from first to third order using history.
    Bind generators because Krea2Pipeline does not forward them to ``step``.
    """

    order = 1
    solver_order = 3
    integration_points = 200
    initial_sigma_epsilon = 1e-4

    def __init__(self, schedule, *, generators=None):
        if schedule.config.get("invert_sigmas") or schedule.config.get("stochastic_sampling"):
            raise ValueError("ER SDE requires a descending flow schedule without Euler stochastic_sampling")
        self._schedule = schedule
        self._generators = generators
        self.config = FrozenDict({
            **dict(schedule.config),
            "_class_name": type(self).__name__,
            "solver_order": self.solver_order,
            "integration_points": self.integration_points,
            "initial_sigma_epsilon": self.initial_sigma_epsilon,
            "algorithm_type": "er_sde",
            "s_noise": 1.0,
        })
        self._previous_denoised = self._previous_derivative = None
        self._coefficients = None

    def __getattr__(self, name):
        # timesteps, sigmas, step_index, set_begin_index and schedule utilities
        # retain the interface expected by the existing Krea2 pipeline.
        return getattr(self._schedule, name)

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        self._schedule.set_timesteps(num_inference_steps=num_inference_steps, device=device,
                                     sigmas=sigmas, mu=mu, timesteps=timesteps)
        self._previous_denoised = self._previous_derivative = None
        self._coefficients = None
        values = self._schedule.sigmas.detach().cpu().double()
        if len(values) < 2 or not torch.isfinite(values).all() or values[-1] != 0:
            raise ValueError("ER SDE requires a finite flow schedule ending at zero")
        if values[0] == 1:
            # Pure noise gives alpha=0 and infinite lambda. Move both the model
            # timestep and solver sigma slightly inward before model evaluation.
            self._schedule.sigmas = self._schedule.sigmas.clone()
            self._schedule.timesteps = self._schedule.timesteps.clone()
            self._schedule.sigmas[0] = 1 - self.initial_sigma_epsilon
            self._schedule.timesteps[0] = self._schedule.sigmas[0] * self.config.num_train_timesteps
            values = self._schedule.sigmas.detach().cpu().double()
        if not ((values[:-1] > values[1:]).all() and (values >= 0).all() and (values < 1).all()):
            raise ValueError("ER SDE requires strictly descending sigmas in [0, 1)")
        self._coefficients = self._prepare_coefficients(values)

    def _prepare_coefficients(self, sigmas):
        # Scalar coefficients are independent of latents. Compute once in CPU
        # float64, avoiding per-step GPU synchronization and low-precision sums.
        lambdas = sigmas / (1 - sigmas)
        grid = torch.arange(self.integration_points, dtype=torch.float64) / self.integration_points

        def phi(value):
            return value * (torch.exp(value ** 0.3) + 10)

        coefficients = []
        for index in range(len(sigmas) - 1):
            if sigmas[index + 1] == 0:
                coefficients.append(_Coefficients(0.0, 1.0))
                continue
            source, target = lambdas[index], lambdas[index + 1]
            alpha_source, alpha_target = 1 - sigmas[index], 1 - sigmas[index + 1]
            phi_target = phi(target)
            ratio = phi_target / phi(source)
            first = second = inverse_delta = inverse_span = 0.0
            if index >= 1:
                delta = target - source
                positions = target + grid * (source - target)
                weights = (source - target) / self.integration_points / phi(positions)
                first = alpha_target * (delta + weights.sum() * phi_target)
                inverse_delta = 1 / (source - lambdas[index - 1])
                if index >= 2:
                    second = alpha_target * (delta.square() / 2 + ((positions - source) * weights).sum() * phi_target)
                    inverse_span = 2 / (source - lambdas[index - 2])
            variance = target.square() - source.square() * ratio.square()
            coefficients.append(_Coefficients(
                sample=float(alpha_target / alpha_source * ratio),
                denoised=float(alpha_target * (1 - ratio)),
                first=float(first), second=float(second), inverse_delta=float(inverse_delta),
                inverse_span=float(inverse_span), noise=float(alpha_target * variance.clamp_min(0).sqrt()),
            ))
        if any(not math.isfinite(value) for coefficient in coefficients for value in vars(coefficient).values()):
            raise ValueError("ER SDE produced nonfinite coefficients for this schedule")
        return coefficients

    @torch.no_grad()
    def step(self, model_output, timestep, sample, *, generator=None, return_dict=True):
        if self._coefficients is None:
            raise ValueError("Call set_timesteps before ER SDE sampling")
        if self._schedule.step_index is None:
            self._schedule._init_step_index(timestep)
        index = self._schedule.step_index
        if index >= len(self._coefficients):
            raise ValueError("ER SDE sampling has already finished")
        coefficient = self._coefficients[index]
        sample = sample.float()
        velocity = model_output.float()
        sigma = self._schedule.sigmas[index].to(device=sample.device, dtype=torch.float32)
        denoised = sample - sigma * velocity
        previous = coefficient.sample * sample + coefficient.denoised * denoised
        derivative = None
        if coefficient.inverse_delta:
            if self._previous_denoised is None:
                raise ValueError("ER SDE requires sequential steps starting at the first timestep")
            derivative = (denoised - self._previous_denoised) * coefficient.inverse_delta
            previous = previous + coefficient.first * derivative
            if coefficient.inverse_span:
                second_derivative = (derivative - self._previous_derivative) * coefficient.inverse_span
                previous = previous + coefficient.second * second_derivative
        if coefficient.noise:
            generators = generator if generator is not None else self._generators
            if generators is None:
                raise ValueError("ER SDE requires seeded generators for sampling noise")
            noise = randn_tensor(sample.shape, generator=generators, device=sample.device, dtype=sample.dtype)
            previous = previous + coefficient.noise * noise
        self._previous_denoised, self._previous_derivative = denoised, derivative
        self._schedule._step_index += 1
        previous = previous.to(model_output.dtype)
        if not return_dict:
            return (previous,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=previous)

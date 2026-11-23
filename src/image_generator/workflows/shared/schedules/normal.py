"""Uniform training-timestep grids behind the existing Diffusers step interface.

For discrete diffusion, interpolate the training log-sigma table on an inclusive
uniform timestep grid, then append zero. This is ComfyUI Normal's sigma rule
when supplied the same training table. The native solver still owns its update,
prediction conversion, history and generator handling.

For flow matching, use the native scheduler's full training-range grid and
model-specific shift, instead of the pipeline's step-count-dependent input grid.
"""

import torch
from diffusers.configuration_utils import FrozenDict


class NormalScheduleAdapter:
    def __init__(self, scheduler, *, backend):
        self._scheduler = scheduler
        self._backend = backend
        self.config = FrozenDict({**dict(scheduler.config), "schedule_adapter": "normal",
                                  "base_scheduler_class": type(scheduler).__name__})

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    # ER_SDE adjusts the first sigma/timestep and advances the underlying index.
    # Delegate writes as well as reads so both adapters share exactly one state.
    @property
    def sigmas(self):
        return self._scheduler.sigmas

    @sigmas.setter
    def sigmas(self, value):
        self._scheduler.sigmas = value

    @property
    def timesteps(self):
        return self._scheduler.timesteps

    @timesteps.setter
    def timesteps(self, value):
        self._scheduler.timesteps = value

    @property
    def _step_index(self):
        return self._scheduler._step_index

    @_step_index.setter
    def _step_index(self, value):
        self._scheduler._step_index = value

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        if timesteps is not None:
            raise ValueError("Normal generates its own timestep grid; use Default for custom timesteps")
        steps = num_inference_steps if num_inference_steps is not None else (len(sigmas) if sigmas is not None else None)
        if type(steps) is not int or steps < 1:
            raise ValueError("Normal requires a positive inference step count")
        if sigmas is not None and len(sigmas) != steps:
            raise ValueError("Normal sigma input length must match the inference step count")
        if self._backend == "flow_match":
            # Native defaults span the training range. Pipelines usually pass
            # a grid ending at 1/steps; Normal deliberately replaces that grid.
            self._scheduler.set_timesteps(num_inference_steps=steps, device=device, mu=mu)
            return
        training_alpha = self._scheduler.alphas_cumprod.detach().cpu().float()
        if steps > len(training_alpha):
            raise ValueError("Normal cannot use more inference steps than training timesteps")
        log_sigmas = ((1 - training_alpha) / training_alpha).sqrt().log()
        positions = torch.linspace(len(log_sigmas) - 1, 0, steps)
        low, high = positions.floor().long(), positions.ceil().long()
        fraction = positions - low
        interpolated = (1 - fraction) * log_sigmas[low] + fraction * log_sigmas[high]
        sigmas = torch.cat([interpolated.exp(), torch.zeros(1)])
        if not torch.isfinite(sigmas).all():
            raise ValueError("Normal requires finite training sigmas")

        # Match the discrete model's nearest training timestep for each sigma.
        # Comparing adjacent entries avoids building an inference*training matrix.
        nearer_high = (log_sigmas[high] - interpolated).abs() < (interpolated - log_sigmas[low]).abs()
        model_timesteps = torch.where(nearer_high, high, low)
        self._scheduler.set_timesteps(steps, device=device)
        dtype = torch.int64 if self._backend in ("dpm_multistep", "unipc_multistep") else torch.float32
        self._scheduler.timesteps = model_timesteps.to(device=device, dtype=dtype)
        self._scheduler.sigmas = sigmas

"""WAN training-time grids, shifted once by the native UniPC flow scheduler."""

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.configuration_utils import FrozenDict


class WanFlowScheduleAdapter:
    def __init__(self, scheduler, *, backend, schedule):
        if (not isinstance(scheduler, UniPCMultistepScheduler)
                or scheduler.config.prediction_type != "flow_prediction" or not scheduler.config.use_flow_sigmas):
            raise ValueError("WAN named schedules require a UniPC flow_prediction source with use_flow_sigmas=True")
        self._scheduler, self._schedule = scheduler, schedule
        self.config = FrozenDict({**dict(scheduler.config), "schedule_adapter": schedule,
                                  "base_scheduler_class": type(scheduler).__name__})

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def set_timesteps(self, num_inference_steps, device=None):
        training_steps = self.config.num_train_timesteps
        if type(num_inference_steps) is not int or not 1 <= num_inference_steps <= training_steps:
            raise ValueError(f"WAN named schedules require 1 to {training_steps} inference steps")
        if self._schedule == "normal":
            # Inclusive full training range; Default uses a step-dependent end.
            grid = np.linspace(1.0, 1.0 / training_steps, num_inference_steps)
        else:
            # ComfyUI Simple's reverse-table indices, applied to the flow model's
            # training times (not UniPC's discrete-diffusion alpha table).
            indices = np.floor(np.arange(num_inference_steps) * training_steps / num_inference_steps)
            grid = (training_steps - indices) / training_steps
        self._scheduler.set_timesteps(sigmas=grid, device=device)
        sigmas = self._scheduler.sigmas
        if (not torch.isfinite(sigmas).all() or sigmas[-1] != 0
                or not ((sigmas[:-1] > 0) & (sigmas[:-1] < 1)).all()
                or not (sigmas[:-1] > sigmas[1:]).all()):
            raise ValueError("WAN named schedule produced an invalid descending flow grid")

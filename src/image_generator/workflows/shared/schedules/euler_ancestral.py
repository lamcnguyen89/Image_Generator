"""Native Euler sigma curves with the Euler ancestral numerical update."""

from diffusers import EulerDiscreteScheduler
from diffusers.configuration_utils import FrozenDict


class EulerAncestralScheduleAdapter:
    def __init__(self, scheduler, *, backend, **curve_options):
        self._scheduler = scheduler
        # Diffusers' Euler scheduler supplies both curve generation and the
        # matching fractional model timesteps. The ancestral scheduler retains
        # input scaling, prediction conversion and generator-owned step noise.
        self._grid = EulerDiscreteScheduler.from_config(scheduler.config, **curve_options)
        curve = next(name for name in ("karras", "exponential", "beta")
                     if curve_options[f"use_{name}_sigmas"])
        self.config = FrozenDict({**dict(scheduler.config), **curve_options,
                                  "schedule_adapter": curve,
                                  "base_scheduler_class": type(scheduler).__name__})

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def set_timesteps(self, num_inference_steps=None, device=None, *, timesteps=None, sigmas=None):
        if timesteps is not None or sigmas is not None:
            raise ValueError("Named Euler ancestral schedules generate their own grid and do not accept custom grids")
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("Euler ancestral schedules require a positive inference step count")
        self._grid.set_timesteps(num_inference_steps, device=device)
        # Reset the solver's begin/step indices before installing the new grid.
        self._scheduler.set_timesteps(num_inference_steps, device=device)
        self._scheduler.sigmas = self._grid.sigmas.clone()
        self._scheduler.timesteps = self._grid.timesteps.clone()

"""Native flow solvers driven by the model's original shifted sigma grid."""

import inspect

import torch
from diffusers.configuration_utils import FrozenDict


class FlowSolverAdapter:
    # Flow pipelines supply custom sigmas and count the expanded timestep list
    # as model evaluations, including Heun predictor/corrector evaluations.
    order = 1
    initial_sigma_epsilon = 1e-6

    def __init__(self, schedule, solver, *, generators, heun=False):
        if schedule.config.get("invert_sigmas") or schedule.config.get("stochastic_sampling"):
            raise ValueError("Flow solvers require descending schedules without Euler stochastic_sampling")
        self._schedule, self._solver = schedule, solver
        self._generators, self._heun = generators, heun
        self._accepts_generator = "generator" in inspect.signature(solver.step).parameters
        self._stochastic = solver.config.get("algorithm_type", "").startswith("sde-")
        self.config = FrozenDict({**dict(solver.config), **dict(schedule.config),
                                  "_class_name": type(self).__name__,
                                  "base_scheduler_class": type(solver).__name__,
                                  "prediction_type": "flow_prediction",
                                  "initial_sigma_epsilon": 0 if heun else self.initial_sigma_epsilon,
                                  "solver_config": dict(solver.config)})

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        self._schedule.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu, timesteps=timesteps)
        grid = self._schedule.sigmas.detach().cpu().float().clone()
        times = self._schedule.timesteps.clone()
        if (len(grid) != len(times) + 1 or not torch.isfinite(grid).all()
                or grid[-1] != 0 or not ((grid[:-1] > 0) & (grid[:-1] <= 1)).all()
                or not (grid[:-1] > grid[1:]).all()):
            raise ValueError("Flow solvers require strictly decreasing finite sigmas in (0, 1] ending at zero")
        if not self._heun and grid[0] == 1:
            # UniPC and DPM++ use log((1-sigma)/sigma), singular at sigma=1.
            grid[0] = 1 - self.initial_sigma_epsilon
            times[0] = grid[0] * self.config.num_train_timesteps
            if not (grid[:-1] > grid[1:]).all():
                raise ValueError("Flow sigma grid is too close to the singular sigma=1 endpoint")
        self._solver.set_timesteps(len(times), device=device)
        if self._heun:
            times = torch.cat([times[:1], times[1:].repeat_interleave(2)])
            grid = torch.cat([grid[:1], grid[1:-1].repeat_interleave(2), grid[-1:]])
            self._solver.sample = None
        self._solver.sigmas = grid
        self._solver.timesteps = times.to(device=device)

    def step(self, model_output, timestep, sample, *, generator=None, return_dict=True):
        generators = generator if generator is not None else self._generators
        if self._stochastic and not generators:
            raise ValueError("Flow SDE samplers require seeded generators for sampling noise")
        options = {"generator": generators} if self._accepts_generator else {}
        return self._solver.step(model_output, timestep, sample, return_dict=return_dict, **options)

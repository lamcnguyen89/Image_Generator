"""WAN solvers preserving the loader's UniPC flow grid and expert timesteps."""

import inspect

import torch
from diffusers.configuration_utils import FrozenDict

from ..sampling import build_scheduler


def build_wan(*, spec, sampling, requirements, diffusers, original_class, original_config, generators):
    if original_class is not diffusers.UniPCMultistepScheduler:
        raise ValueError("Explicit WAN samplers require a UniPCMultistepScheduler source")
    source = build_scheduler(diffusers.UniPCMultistepScheduler, original_config, requirements=requirements)
    if source.config.prediction_type != "flow_prediction" or not source.config.use_flow_sigmas:
        raise ValueError("Explicit WAN samplers require UniPC flow_prediction and use_flow_sigmas=True")
    if spec.name == "uni_pc":
        return source
    solver_class = getattr(diffusers, spec.scheduler_class)
    parameters = inspect.signature(solver_class.__init__).parameters
    config = {name: value for name, value in source.config.items() if name in parameters}
    solver = solver_class.from_config(config, **dict(spec.scheduler_options))
    return WanDPMSolverAdapter(source, solver)


class WanDPMSolverAdapter:
    def __init__(self, source, solver):
        self._source, self._solver = source, solver
        self.config = FrozenDict({**dict(solver.config), "_class_name": type(self).__name__,
                                  "base_scheduler_class": type(solver).__name__,
                                  "source_scheduler_class": type(source).__name__,
                                  "source_scheduler_config": dict(source.config)})

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def set_timesteps(self, num_inference_steps, device=None):
        self._source.set_timesteps(num_inference_steps, device=device)
        grid = self._source.sigmas
        if (len(grid) != len(self._source.timesteps) + 1 or not torch.isfinite(grid).all()
                or grid[-1] != 0 or not ((grid[:-1] > 0) & (grid[:-1] < 1)).all()
                or not (grid[:-1] > grid[1:]).all()):
            raise ValueError("WAN DPM++ requires a descending UniPC flow grid in (0, 1) ending at zero")
        self._solver.set_timesteps(num_inference_steps, device=device)
        # The pipeline selects high/low-noise experts using these exact source
        # timesteps. Native DPM++ otherwise prepares a different flow grid.
        self._solver.sigmas = grid.clone()
        self._solver.timesteps = self._source.timesteps.clone()

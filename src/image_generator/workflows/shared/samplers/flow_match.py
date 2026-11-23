"""Factories for flow-prediction pipelines; invoked only during generation."""

import inspect

from ..sampling import build_scheduler
from ..schedules import apply_schedule, scheduler_options


def build_euler(*, spec, diffusers, original_config, requirements, sampling, generators, original_class=None):
    scheduler = build_scheduler(getattr(diffusers, spec.schedule_class), original_config,
                                requirements=requirements,
                                **scheduler_options(spec.schedule_backend, sampling["noise_schedule"], requirements))
    return apply_schedule(spec.schedule_backend, sampling["noise_schedule"], requirements, scheduler)


def build_er_sde(*, spec, diffusers, original_config, requirements, sampling, generators, original_class=None):
    from .er_sde import FlowMatchERSDEScheduler

    # Keep each model's resolution-aware schedule; change only the solver.
    schedule = build_euler(spec=spec, diffusers=diffusers, original_config=original_config,
                          requirements=requirements, sampling=sampling, generators=generators)
    return FlowMatchERSDEScheduler(schedule, generators=generators)


def build_solver(*, spec, diffusers, original_config, requirements, sampling, generators, original_class=None):
    from .flow_solver import FlowSolverAdapter

    schedule = build_euler(spec=spec, diffusers=diffusers, original_config=original_config,
                          requirements=requirements, sampling=sampling, generators=generators)
    options = dict(spec.scheduler_options)
    if spec.scheduler_class != "FlowMatchHeunDiscreteScheduler":
        # Only the source scheduler prepares and shifts the grid. Native solver
        # flags must not transform that grid a second time during reset.
        options.update(use_dynamic_shifting=False, use_karras_sigmas=False,
                       use_exponential_sigmas=False, use_beta_sigmas=False, final_sigmas_type="zero")
    solver_class = getattr(diffusers, spec.scheduler_class)
    parameters = inspect.signature(solver_class.__init__).parameters
    config = {name: value for name, value in schedule.config.items() if name in parameters}
    solver = solver_class.from_config(config, **options)
    return FlowSolverAdapter(schedule, solver, generators=generators,
                             heun=spec.scheduler_class == "FlowMatchHeunDiscreteScheduler")

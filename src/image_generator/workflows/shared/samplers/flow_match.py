"""Factories for flow-prediction pipelines; invoked only during generation."""

from ..sampling import build_scheduler
from ..schedules import apply_schedule, scheduler_options


def build_euler(*, spec, diffusers, original_config, requirements, sampling, generators, original_class=None):
    scheduler = build_scheduler(getattr(diffusers, spec.schedule_class), original_config,
                                requirements=requirements,
                                **scheduler_options(spec.schedule_backend, sampling["noise_schedule"], requirements))
    return apply_schedule(spec.schedule_backend, sampling["noise_schedule"], requirements, scheduler)


def build_er_sde(*, spec, diffusers, original_config, requirements, sampling, generators, original_class=None):
    from .er_sde import FlowMatchERSDEScheduler

    # Keep Krea2's resolution-aware schedule; change only the numerical solver.
    schedule = build_euler(spec=spec, diffusers=diffusers, original_config=original_config,
                          requirements=requirements, sampling=sampling, generators=generators)
    return FlowMatchERSDEScheduler(schedule, generators=generators)

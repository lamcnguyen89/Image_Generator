"""CPU-only form controls derived from registered sampler capabilities."""

from ...parameters import Parameter
from .registry import available_samplers
from ..schedules import available_schedules


def sampling_fields(model, *, workflow=None):
    requirements = model.defaults["requirements"]
    specs = available_samplers(workflow or model.defaults["workflow"], requirements)
    if not specs:
        raise ValueError("This checkpoint has no sampling options supported by its workflow")
    allowed = requirements.get("allowed_sampling")
    schedules = dict(
        (schedule.name, schedule.label) for spec in specs for schedule in available_schedules(spec.schedule_backend)
        if allowed is None or {"sampler": spec.name, "noise_schedule": schedule.name} in allowed
    )
    schedules_by_sampler = tuple(
        (spec.name, tuple(schedule.name for schedule in available_schedules(spec.schedule_backend)
                          if allowed is None or {"sampler": spec.name, "noise_schedule": schedule.name} in allowed))
        for spec in specs
    )
    return (
        Parameter("sampling.sampler", "Sampler", "select", "Sampling",
                  choices=tuple((spec.name, spec.label) for spec in specs), cli_flags=("--sampler",)),
        Parameter("sampling.noise_schedule", "Noise schedule", "select", "Advanced",
                  choices=tuple(schedules.items()), choice_source="sampling.sampler",
                  choices_by_source=schedules_by_sampler,
                  validation_message="Choose a noise schedule supported by the selected sampler."),
    )

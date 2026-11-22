"""Native Diffusers scheduler bindings, constructed lazily from registry metadata."""

from ..sampling import build_scheduler
from ..schedules import apply_schedule, scheduler_options


def build_native(*, spec, sampling, requirements, diffusers, original_class, original_config, generators):
    scheduler_class = getattr(diffusers, spec.scheduler_class) if spec.scheduler_class else original_class
    options = dict(spec.scheduler_options)
    options.update(scheduler_options(spec.schedule_backend, sampling["noise_schedule"], requirements))
    scheduler = build_scheduler(scheduler_class, original_config, requirements=requirements, **options)
    return apply_schedule(spec.schedule_backend, sampling["noise_schedule"], requirements, scheduler)

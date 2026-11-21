"""Construct schedulers from the original loader configuration for each request."""


def build_scheduler(scheduler_class, original_config, *, requirements, **options):
    options.update(requirements.get("scheduler_config", {}))
    return scheduler_class.from_config(original_config, **options)

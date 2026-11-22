"""Shared metadata helpers."""

import math


def scheduler_config_snapshot(value):
    """Represent library infinity/NaN sentinels as strings in strict JSON."""
    if isinstance(value, dict):
        return {key: scheduler_config_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [scheduler_config_snapshot(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def scheduler_metadata(scheduler):
    result = {"class_name": type(scheduler).__name__,
              "config": scheduler_config_snapshot(dict(scheduler.config))}
    for name in ("sigmas", "timesteps"):
        values = getattr(scheduler, name, None)
        if values is None:
            continue
        if hasattr(values, "detach"):
            values = values.detach().cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, (list, tuple)):
            result[name] = scheduler_config_snapshot(values)
    return result

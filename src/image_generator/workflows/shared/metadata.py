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

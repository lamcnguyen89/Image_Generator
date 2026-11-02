"""Resolve settings without changing source dictionaries or writing files."""

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


def lora_strength_minimum(family: str) -> int | None:
    """Qwen 2.1 supports signed adapter ratios; other workflows require >= 0."""
    return None if family == "qwen_2_1" else 0


def merge_settings(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """Merge objects recursively; lists and scalars (including null) replace.

    The returned tree shares no mutable values with the input trees. Removing
    an override from a user file restores the lower-priority value on the next
    resolution. This function merges settings, never whole manifests.
    """
    result: dict[str, Any] = {}
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise TypeError("each settings layer must be an object")
        for key, value in layer.items():
            if isinstance(value, Mapping):
                previous = result.get(key)
                result[key] = merge_settings(previous if isinstance(previous, Mapping) else {}, value)
            else:
                result[key] = deepcopy(value)
    return result


def resolve_settings(
    defaults: Mapping[str, Any],
    user_overrides: Mapping[str, Any] | None = None,
    *,
    workflow_defaults: Mapping[str, Any] | None = None,
    request_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply workflow → asset → saved user → current request precedence.

    Arguments are the contents of each settings object, not the manifests.
    This is a pure merge; callers validate the resulting settings separately.
    LoRA recommendations are never applied to model settings automatically.
    """
    return merge_settings(
        {} if workflow_defaults is None else workflow_defaults,
        defaults,
        {} if user_overrides is None else user_overrides,
        {} if request_overrides is None else request_overrides,
    )

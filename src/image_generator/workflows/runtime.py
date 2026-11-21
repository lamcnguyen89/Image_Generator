"""Runtime configuration declarations; never part of a saved generation request."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeOption:
    name: str
    type: type = str
    help: str = ""


def resolve_runtime_options(definition, options):
    """Project process options onto the backend's declared configuration."""
    names = set(definition.runtime_options)
    names.update(option.name for option in definition.extra_runtime_options)
    return {name: value for name, value in options.items() if name in names}

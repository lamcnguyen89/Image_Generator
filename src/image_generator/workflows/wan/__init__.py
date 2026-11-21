"""wan integration with lazy compatibility exports."""

from importlib import import_module

__all__ = ["WanWorkflow"]
_EXPORTS = {"WanWorkflow": "runtime"}


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    return getattr(import_module(f"{__name__}.{module}"), name)

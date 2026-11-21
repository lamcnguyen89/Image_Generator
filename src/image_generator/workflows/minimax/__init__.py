"""minimax integration with lazy compatibility exports."""

from importlib import import_module

__all__ = ["MiniMaxWorkflow", "protect_quantized_transfers", "safetensors_header"]
_EXPORTS = {
    "MiniMaxWorkflow": "runtime",
    "protect_quantized_transfers": "comfy_runtime",
    "safetensors_header": "validation",
}


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    return getattr(import_module(f"{__name__}.{module}"), name)

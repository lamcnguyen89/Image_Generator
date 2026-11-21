"""qwen21 integration with lazy compatibility exports."""

from importlib import import_module

__all__ = ["Qwen21Workflow", "split_fused_lora", "load_qwen21_lora", "validate_sigmas"]
_EXPORTS = {
    "Qwen21Workflow": "runtime",
    "split_fused_lora": "adapters",
    "load_qwen21_lora": "adapters",
    "validate_sigmas": "validation",
}


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    return getattr(import_module(f"{__name__}.{module}"), name)

"""MiniMax's explicit H3 adapter compatibility policy."""

from dataclasses import dataclass

from ..adapters import AdapterPolicy


@dataclass(frozen=True)
class MiniMaxAdapterPolicy(AdapterPolicy):
    retain_unavailable: bool = True

    def unavailable(self, model, asset, **options):
        reason = super().unavailable(model, asset, **options)
        if reason:
            return reason
        if asset.defaults["compatibility"].get("model_variant") != "minimax_h3":
            return "Record explicit MiniMax H3 compatibility before using this adapter."
        return ""

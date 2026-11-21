"""WAN expert targeting and variant compatibility."""

from dataclasses import dataclass

from image_generator.catalog.compatibility import lora_model_variants

from ..adapters import AdapterPolicy
from ..parameters import Parameter


@dataclass(frozen=True)
class WanAdapterPolicy(AdapterPolicy):
    retain_unavailable: bool = True
    link_fields: tuple = (
        Parameter(
            "expert_target",
            "Noise expert",
            "select",
            "Adapters",
            required=True,
            choices=(("high_noise", "High noise"), ("low_noise", "Low noise")),
            help="Choose which expert loads this adapter. Saved with this model's settings.",
            validation_message="Choose the high-noise or low-noise expert.",
        ),
    )
    group_field: str = "expert_target"
    groups: tuple = (
        ("high_noise", "High-noise adapters"),
        ("low_noise", "Low-noise adapters"),
        ("unknown", "Unavailable or unconfigured adapters"),
    )

    def unavailable(self, model, asset, *, link=None, require_complete=True):
        if asset.family != model.family:
            return "This adapter belongs to a different family."
        compatibility = asset.defaults["compatibility"]
        if model.defaults.get("model_variant") not in lora_model_variants(compatibility):
            return "This adapter is not declared compatible with the selected WAN model variant."
        target = self.field_value(asset, link or {}, "expert_target")
        if require_complete and target not in {"high_noise", "low_noise"}:
            return "Choose a high-noise or low-noise expert for this WAN adapter."
        return ""


def configure_expert_loras(self, request):
    self.report_progress("resetting_loras")
    if self._has_adapters:
        options = {}
        self._pipeline.unload_lora_weights(**options)
        self._has_adapters = False
    self.report_progress("loading_loras", 0, len(request.loras))
    for index, selection in enumerate(request.loras):
        self._has_adapters = True
        options = {}
        options["load_into_transformer_2"] = selection.expert_target == "low_noise"
        try:
            self._load_lora(selection, **options)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load LoRA {selection.asset.id!r}; check model and loader compatibility: {exc}"
            ) from exc
        self.report_progress("loading_loras", index + 1, len(request.loras))
    for target, component in (("high_noise", "transformer"), ("low_noise", "transformer_2")):
        selections = [item for item in request.loras if item.expert_target == target]
        if selections:
            getattr(self._pipeline, component).set_adapters(
                [item.adapter_name for item in selections],
                weights=[float(item.strength) for item in selections],
            )
    self.report_progress("loras_ready", len(request.loras), len(request.loras))

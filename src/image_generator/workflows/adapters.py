"""Adapter policy contract shared by forms, persistence, and request preparation."""

from dataclasses import dataclass

from image_generator.catalog.compatibility import lora_model_variants


@dataclass(frozen=True)
class AdapterPolicy:
    minimum_strength: float | None = 0
    retain_unavailable: bool = False
    link_fields: tuple = ()
    group_field: str | None = None
    groups: tuple[tuple[str, str], ...] = ()

    def unavailable(self, model, asset, *, link=None, require_complete=True):
        if asset.family != model.family:
            return "This adapter belongs to a different family."
        variants = lora_model_variants(asset.defaults["compatibility"])
        if variants and model.defaults.get("model_variant") and model.defaults["model_variant"] not in variants:
            return "This adapter targets a different model variant."
        return ""

    def validate_link(self, asset, settings, *, require_complete=True):
        from .parameters import Parameter

        fields = (
            Parameter("strength", "Strength", "number", "Adapters", minimum=self.minimum_strength),
            *self.link_fields,
        )
        for parameter in fields:
            value = self.field_value(asset, settings, parameter.name)
            if parameter.name != "strength" and not require_complete and value in (None, ""):
                continue
            error = parameter.validate(value)
            if error:
                raise ValueError(f"LoRA {asset.id} {parameter.name}: {parameter.validation_message or error}")

    def field_value(self, asset, link, name):
        return link.get(name, asset.resolve_settings().get(name, asset.defaults["compatibility"].get(name)))

    def request_settings(self, asset, link):
        settings = asset.resolve_settings()
        for field in self.link_fields:
            if field.name in link:
                settings[field.name] = link[field.name]
        return settings


def policy_for_model(model):
    from .registry import get_definition

    return get_definition(model.defaults["workflow"]).adapter_policy


def policy_for_family(family):
    from .registry import WORKFLOWS, get_definition

    # A family's standard integration defines its shared adapter defaults.
    if family in WORKFLOWS:
        return get_definition(family).adapter_policy
    for name in WORKFLOWS:
        spec = get_definition(name)
        if spec.family == family:
            return spec.adapter_policy
    return AdapterPolicy()

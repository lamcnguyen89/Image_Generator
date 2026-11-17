"""Shared declarations for adapters that support multiple WAN variants."""


def lora_model_variants(compatibility):
    """An explicit list takes precedence; legacy single-variant entries still work."""
    if "model_variants" in compatibility:
        return tuple(compatibility["model_variants"])
    variant = compatibility.get("model_variant")
    return (variant,) if variant else ()

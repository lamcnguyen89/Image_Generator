"""Cross-field and asset validation without loading model runtimes."""

import math
import re

from ..base import GenerationRequest
from ..shared.validation import _nonnegative_number
from ..shared.samplers import validate_sampling
from .parameters import get_ui_definition


def validate_form_settings(model, mode, settings):
    get_ui_definition(model, mode).validate_settings(settings)
    validate_sampling("sdxl", settings["sampling"], model.defaults["requirements"], allow_extensions=True)


def validate(request: GenerationRequest) -> None:
    model, settings = request.model, request.settings
    if request.inputs:
        raise ValueError("SDXL txt2img does not accept image inputs")
    if model.kind != "model" or model.family != "sdxl":
        raise ValueError("The SDXL workflow requires an SDXL catalog model")
    if request.mode != "txt2img" or request.mode not in model.defaults["supported_modes"]:
        raise ValueError("This workflow currently supports only declared txt2img mode")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if not isinstance(request.negative_prompt, str):
        raise ValueError("negative_prompt must be a string")
    requirements = model.defaults["requirements"]
    multiple = requirements.get("dimension_multiple", 8)
    if type(multiple) is not int or multiple <= 0:
        raise ValueError("requirements.dimension_multiple must be a positive integer")
    multiple = math.lcm(8, multiple)
    for name in ("width", "height", "num_inference_steps", "num_images_per_prompt"):
        value = settings.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if name in {"width", "height"} and value % multiple:
            raise ValueError(f"{name} must be divisible by {multiple}")
    seed = settings.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("seed must be null or an integer between 0 and 2**63 - 1")
    if not _nonnegative_number(settings.get("guidance_scale")):
        raise ValueError("guidance_scale must be a finite number >= 0")
    clip_skip = settings.get("clip_skip")
    if clip_skip is not None and (type(clip_skip) is not int or clip_skip < 0):
        raise ValueError("clip_skip must be null or a nonnegative integer")
    validate_sampling("sdxl", settings.get("sampling"), requirements)
    validate_weights(model.asset_paths["weights"], "Checkpoint", optional=True)
    if model.asset_paths["weights"] is None and not model.defaults.get("base_model_id"):
        raise ValueError("A pretrained SDXL model requires base_model_id")
    names, identifiers = set(), set()
    for selection in request.loras:
        asset = selection.asset
        if asset.family != "sdxl" or asset.kind != "lora":
            raise ValueError(f"LoRA {asset.id!r} does not declare SDXL compatibility")
        variant = asset.defaults["compatibility"].get("model_variant")
        model_variant = model.defaults.get("model_variant")
        if variant and model_variant and variant != model_variant:
            raise ValueError(f"LoRA {asset.id!r} targets a different model variant")
        if asset.id in identifiers or selection.adapter_name in names:
            raise ValueError("Duplicate LoRA IDs or adapter names")
        identifiers.add(asset.id)
        names.add(selection.adapter_name)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", selection.adapter_name):
            raise ValueError("Adapter names may contain only letters, digits, underscores, and hyphens")
        if not _nonnegative_number(selection.strength):
            raise ValueError(f"LoRA {asset.id!r} strength must be a finite number >= 0")
        validate_weights(asset.asset_paths["weights"], f"LoRA {asset.id!r}")


def validate_weights(path, label, *, optional=False):
    if path is None and optional:
        return
    from ..shared.validation import validate_weights as validate_local

    validate_local(path, label)

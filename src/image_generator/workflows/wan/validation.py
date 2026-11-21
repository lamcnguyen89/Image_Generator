"""Cross-field and asset validation without loading model runtimes."""

import json

from image_generator.catalog.compatibility import lora_model_variants

from ..shared.validation import _nonnegative_number, validate_common
from .parameters import DEFAULTS, VARIANTS, get_ui_definition


def variant_definition(model):
    variant = model.defaults.get("model_variant")
    if variant not in VARIANTS:
        raise ValueError("WAN requires model_variant wan_2_2_i2v_a14b or wan_2_2_t2v_a14b")
    return VARIANTS[variant]


def validate_form_settings(model, mode, settings):
    from ..ui import FormValidationError

    get_ui_definition(model, mode).validate_settings(settings)
    errors = {}
    if settings["fps"] <= 0:
        errors["fps"] = "Playback FPS must be greater than zero."
    if (settings["guidance_scale"] > 1) != (settings["guidance_scale_2"] > 1):
        errors["guidance_scale_2"] = "Both guidance values must be above 1 for CFG, or both at most 1."
    if type(settings["preserve_aspect_ratio"]) is not bool:
        errors["preserve_aspect_ratio"] = "Choose a boolean value."
    if errors:
        raise FormValidationError(errors)


def validate(request):
    mode, pipeline_class = variant_definition(request.model)
    validate_common(
        request, DEFAULTS, family="wan", mode=mode, count_field="num_videos_per_prompt", samplers={"model_default"}
    )
    model, settings = request.model, request.settings
    if mode == "img2vid":
        if set(request.inputs) != {"input_image"} or not request.inputs["input_image"].is_file():
            raise ValueError("WAN img2vid requires a local input_image file")
    elif request.inputs:
        raise ValueError("WAN txt2vid does not accept image inputs")
    path = model.asset_paths["weights"]
    if path is not None:
        if (
            model.defaults.get("weights_component") != "pipeline"
            or not path.is_dir()
            or not (path / "model_index.json").is_file()
        ):
            raise ValueError("Local WAN weights must be a complete Diffusers pipeline directory with model_index.json")
        index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            raise ValueError("Local WAN model_index.json must be an object")
        declared = index.get("_class_name")
        if declared != pipeline_class and (declared is not None or mode == "txt2vid"):
            raise ValueError(f"This WAN variant requires a {pipeline_class} pipeline directory")
    elif not model.defaults.get("base_model_id"):
        raise ValueError("WAN requires base_model_id or a local pipeline directory")
    guidance = settings["guidance_scale_2"]
    if not _nonnegative_number(guidance):
        raise ValueError("guidance_scale_2 must be a finite number >= 0")
    if (settings["guidance_scale"] > 1) != (guidance > 1):
        raise ValueError("Both WAN guidance scales must be > 1 for CFG, or both <= 1")
    if not _nonnegative_number(settings["fps"]) or settings["fps"] == 0:
        raise ValueError("fps must be a finite number > 0")
    for name in ("preserve_aspect_ratio", "enable_vae_tiling"):
        if type(settings[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    rule = model.defaults["requirements"].get("frame_count", {"minimum": 5, "step": 4, "offset": 1})
    if (
        not isinstance(rule, dict)
        or any(type(rule.get(key)) is not int for key in ("minimum", "step", "offset"))
        or rule["step"] <= 0
        or rule["minimum"] < 1
    ):
        raise ValueError("requirements.frame_count needs integer minimum, positive step, and offset")
    count = settings["num_frames"]
    if (
        type(count) is not int
        or count < max(5, rule["minimum"])
        or (count - 1) % 4
        or (count - rule["offset"]) % rule["step"]
    ):
        raise ValueError("num_frames must satisfy the checkpoint frame_count rule and be 4*n + 1, at least 5")
    for selection in request.loras:
        compatibility = selection.asset.defaults["compatibility"]
        if model.defaults["model_variant"] not in lora_model_variants(compatibility):
            raise ValueError(f"WAN LoRA {selection.asset.id!r} needs an explicit compatible model_variant")
        if selection.expert_target not in {"high_noise", "low_noise"}:
            raise ValueError(f"WAN LoRA {selection.asset.id!r} needs an explicit high_noise or low_noise expert_target")

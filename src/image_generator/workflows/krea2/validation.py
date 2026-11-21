"""Cross-field and asset validation without loading model runtimes."""

from ..shared.validation import validate_common, validate_weights
from .parameters import DEFAULTS, get_ui_definition


def validate_form_settings(model, mode, settings):
    get_ui_definition(model, mode).validate_settings(settings)


def validate(request):
    validate_common(
        request,
        DEFAULTS,
        family="krea2",
        mode="txt2img",
        count_field="num_images_per_prompt",
        samplers={"flow_match_euler", "model_default"},
    )
    model, settings = request.model, request.settings
    if request.inputs:
        raise ValueError("Krea2 txt2img does not accept image inputs")
    if model.defaults.get("model_variant") != "krea_2_raw":
        raise ValueError("This Krea2 workflow supports only model_variant krea_2_raw")
    if not model.defaults.get("base_model_id"):
        raise ValueError("Krea2 requires base_model_id for its pipeline components")
    path = model.asset_paths["weights"]
    if path is not None:
        if model.defaults.get("weights_component") != "transformer":
            raise ValueError("A local Krea2 checkpoint must declare weights_component transformer")
        validate_weights(path, "Krea2 transformer")
    requirements = model.defaults["requirements"]
    if requirements.get("scheduler_class", "FlowMatchEulerDiscreteScheduler") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("Krea 2 RAW requires FlowMatchEulerDiscreteScheduler")
    limit = requirements.get("max_sequence_length_limit", 512)
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ValueError("requirements.max_sequence_length_limit must be between 1 and 512")
    length = settings["max_sequence_length"]
    if type(length) is not int or not 1 <= length <= limit:
        raise ValueError(f"max_sequence_length must be between 1 and {limit}")

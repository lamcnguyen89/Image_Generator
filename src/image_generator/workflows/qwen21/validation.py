"""Cross-field and asset validation without loading model runtimes."""

from ..shared.validation import _finite_number, validate_common, validate_weights
from ..shared.samplers import validate_sampling
from ..ui import FormValidationError
from .parameters import DEFAULTS, get_ui_definition


def validate_sigmas(sigmas):
    if sigmas is None:
        return
    if (
        not isinstance(sigmas, list)
        or not sigmas
        or any(not _finite_number(value) or not 0 < value <= 1 for value in sigmas)
    ):
        raise ValueError("Sigmas must be a nonempty list of finite numbers in (0, 1]; omit the terminal zero.")
    if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError("Sigmas must be strictly descending.")


def validate_form_settings(model, mode, settings):
    get_ui_definition(model, mode).validate_settings(settings)
    validate_sampling("qwen_2_1", settings["sampling"], model.defaults["requirements"], allow_extensions=True)
    if settings["sigmas"] is not None and settings["sampling"]["noise_schedule"] != "default":
        raise FormValidationError({"sigmas": "Choose the Default noise schedule when supplying custom sigmas."})
    try:
        validate_sigmas(settings["sigmas"])
    except ValueError as exc:
        raise FormValidationError({"sigmas": str(exc)}) from exc


def validate(request):
    validate_common(
        request,
        DEFAULTS,
        family="qwen_2_1",
        mode="txt2img",
        count_field="num_images_per_prompt",
        sampling_validator=validate_sampling,
        guidance_field="true_cfg_scale",
        dimension_base=32,
    )
    model = request.model
    if request.inputs:
        raise ValueError("Qwen 2.1 txt2img does not accept image inputs")
    if model.defaults.get("model_variant") != "qwen_image_2_1":
        raise ValueError("This Qwen workflow supports only model_variant qwen_image_2_1")
    if not model.defaults.get("base_model_id"):
        raise ValueError("Qwen 2.1 requires base_model_id for its pipeline components")
    path = model.asset_paths["weights"]
    if path is not None:
        if model.defaults.get("weights_component") != "transformer":
            raise ValueError("A local Qwen 2.1 checkpoint must declare weights_component transformer")
        validate_weights(path, "Qwen 2.1 transformer")
    validate_form_settings(model, request.mode, request.settings)

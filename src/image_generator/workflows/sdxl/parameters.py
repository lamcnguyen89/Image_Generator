"""Parameter defaults and controls; safe to inspect without GPU dependencies."""

from ..shared.samplers import available_samplers

# Retain the public SDXL compatibility export, deriving it from the registry.
SAMPLERS = {spec.name: spec.scheduler_class for spec in available_samplers("sdxl", {}) if spec.scheduler_class}

DEFAULTS = {
    "prompt": "",
    "negative_prompt": "",
    "positive_prefix": "",
    "positive_suffix": "",
    "trigger_words": [],
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "num_images_per_prompt": 1,
    "seed": None,
    "clip_skip": None,
    "sampling": {"sampler": "model_default", "noise_schedule": "default"},
    "loras": [],
}

MODES = ("txt2img",)


def get_ui_definition(model, mode):
    from ..ui import UIField, common_fields, definition
    from ..shared.samplers.ui import sampling_fields

    return definition(
        model,
        mode,
        workflow=model.defaults["workflow"],
        family="sdxl",
        supported_modes=MODES,
        fields=(
            *common_fields(model, dimension_base=8),
            *sampling_fields(model, workflow="sdxl"),
            UIField(
                "clip_skip",
                "Clip skip",
                "integer",
                "Advanced",
                minimum=0,
                nullable=True,
                help="Leave blank to use the pipeline default.",
            ),
        ),
    )

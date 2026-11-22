"""Parameter defaults and controls; safe to inspect without GPU dependencies."""

from dataclasses import replace

DEFAULTS = {
    "prompt": "",
    "negative_prompt": "",
    "positive_prefix": "",
    "positive_suffix": "",
    "trigger_words": [],
    "width": 1024,
    "height": 1024,
    "num_images_per_prompt": 1,
    "num_inference_steps": 52,
    "guidance_scale": 3.5,
    "max_sequence_length": 512,
    "seed": None,
    "sampling": {"sampler": "flow_match_euler", "noise_schedule": "default"},
    "loras": [],
}

MODES = ("txt2img",)


def get_ui_definition(model, mode):
    from ..ui import UIField, common_fields, definition
    from ..shared.samplers.ui import sampling_fields

    limit = model.defaults["requirements"].get("max_sequence_length_limit", 512)
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ValueError("requirements.max_sequence_length_limit must be between 1 and 512")
    fields = tuple(
        replace(field, help="Krea 2 CFG: 0 disables guidance; negative prompting applies above 0.")
        if field.name == "guidance_scale"
        else field
        for field in common_fields(model)
    )
    return definition(
        model,
        mode,
        workflow=model.defaults["workflow"],
        family="krea2",
        supported_modes=MODES,
        variant="krea_2_raw",
        fields=(
            *fields,
            *sampling_fields(model, workflow="krea2"),
            UIField(
                "max_sequence_length",
                "Maximum sequence length",
                "integer",
                "Advanced",
                minimum=1,
                maximum=limit,
                step=1,
                help="Maximum prompt token sequence length.",
            ),
        ),
    )

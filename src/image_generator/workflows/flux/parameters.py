"""Parameter defaults and controls; safe to inspect without GPU dependencies."""

DEFAULTS = {
    "prompt": "",
    "negative_prompt": "",
    "positive_prefix": "",
    "positive_suffix": "",
    "trigger_words": [],
    "width": 1024,
    "height": 1024,
    "num_images_per_prompt": 1,
    "num_inference_steps": 50,
    "guidance_scale": 3.5,
    "true_cfg_scale": 1.0,
    "max_sequence_length": 512,
    "seed": None,
    "sampling": {"sampler": "flow_match_euler", "noise_schedule": "default"},
    "loras": [],
}

MODES = ("txt2img",)


def get_ui_definition(model, mode):
    from ..ui import UIField, common_fields, definition, sampling_fields

    limit = model.defaults["requirements"].get("max_sequence_length_limit", 512)
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ValueError("requirements.max_sequence_length_limit must be between 1 and 512")
    return definition(
        model,
        mode,
        workflow=model.defaults["workflow"],
        family="flux",
        supported_modes=MODES,
        variant="flux_1_dev",
        fields=(
            *common_fields(model),
            *sampling_fields(model, ("flow_match_euler", "model_default")),
            UIField(
                "true_cfg_scale",
                "True CFG scale",
                "number",
                "Sampling",
                minimum=1,
                help="Negative prompting takes effect above 1, with a nonempty negative prompt.",
            ),
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

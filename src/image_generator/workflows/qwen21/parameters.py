"""Parameter defaults and controls; safe to inspect without GPU dependencies."""

from dataclasses import replace

DEFAULTS = {
    "prompt": "",
    "negative_prompt": "",
    "positive_prefix": "",
    "positive_suffix": "",
    "trigger_words": [],
    "width": 2048,
    "height": 2048,
    "num_images_per_prompt": 1,
    "num_inference_steps": 40,
    "true_cfg_scale": 1.0,
    "use_kv_cache": True,
    "sigmas": None,
    "seed": None,
    "sampling": {"sampler": "flow_match_euler", "noise_schedule": "default"},
    "loras": [],
}

MODES = ("txt2img",)


def get_ui_definition(model, mode):
    from ..ui import UIField, common_fields, definition, sampling_fields

    fields = tuple(
        UIField(
            "true_cfg_scale",
            "True CFG scale",
            "number",
            "Sampling",
            minimum=1,
            help="1 disables guidance. Negative prompting requires a value above 1 and a nonempty negative prompt.",
        )
        if field.name == "guidance_scale"
        else field
        for field in common_fields(model, dimension_base=32)
    )
    fields = tuple(
        replace(field, help="Used only when true CFG is above 1.") if field.name == "negative_prompt" else field
        for field in fields
    )
    return definition(
        model,
        mode,
        workflow=model.defaults["workflow"],
        family="qwen_2_1",
        supported_modes=MODES,
        variant="qwen_image_2_1",
        fields=(
            *fields,
            *sampling_fields(model, ("flow_match_euler", "model_default")),
            UIField(
                "use_kv_cache",
                "Cache prompt attention",
                "boolean",
                "Advanced",
                cli_flags=("--kv-cache",),
                help="Reuse prompt keys and values during denoising. Keep this fixed when comparing seeded images.",
            ),
            UIField(
                "sigmas",
                "Custom sigma schedule",
                "number_list",
                "Advanced",
                nullable=True,
                help="Optional descending values in (0, 1], separated by commas or newlines. Omit zero. "
                "Blank uses the model schedule; a custom list overrides the inference step count.",
            ),
        ),
    )

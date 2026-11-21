"""Parameter defaults and controls; safe to inspect without GPU dependencies."""

VARIANTS = {
    "wan_2_2_i2v_a14b": ("img2vid", "WanImageToVideoPipeline"),
    "wan_2_2_t2v_a14b": ("txt2vid", "WanPipeline"),
}

DEFAULTS = {
    "prompt": "",
    "negative_prompt": "",
    "positive_prefix": "",
    "positive_suffix": "",
    "trigger_words": [],
    "width": 832,
    "height": 480,
    "preserve_aspect_ratio": True,
    "num_frames": 161,
    "fps": 16,
    "num_videos_per_prompt": 1,
    "num_inference_steps": 40,
    "guidance_scale": 3.5,
    "guidance_scale_2": 3.5,
    "seed": None,
    "enable_vae_tiling": True,
    "sampling": {"sampler": "model_default", "noise_schedule": "default"},
    "loras": [],
}

MODES = ("img2vid", "txt2vid")


def get_ui_definition(model, mode):
    from ..ui import (
        UIField,
        common_fields,
        definition,
        frame_constraints,
        sampling_fields,
    )
    from .validation import variant_definition

    minimum, step = frame_constraints(model.defaults["requirements"])
    variant_mode, _ = variant_definition(model)
    fields = list(common_fields(model, video=True))
    if mode == "img2vid":
        fields.append(
            UIField(
                "preserve_aspect_ratio",
                "Preserve source aspect ratio",
                "boolean",
                "Output",
                inherit_when_absent=True,
                help="Width and height set the pixel budget; actual dimensions follow the source image.",
            )
        )
    return definition(
        model,
        mode,
        workflow=model.defaults["workflow"],
        family="wan",
        supported_modes=(variant_mode,),
        partial="controls/wan.html",
        fields=(
            *fields,
            UIField(
                "num_frames",
                "Frame count",
                "integer",
                "Output",
                cli_flags=("--frames",),
                minimum=minimum,
                step=step,
                help=f"Start at {minimum} and increase by {step} frames.",
            ),
            UIField(
                "fps",
                "Playback FPS",
                "number",
                "Output",
                minimum=0,
                help="Must be greater than zero. Controls exported video playback.",
            ),
            *sampling_fields(model, ("model_default",)),
            UIField(
                "guidance_scale_2",
                "Low-noise guidance",
                "number",
                "Sampling",
                minimum=0,
                cli_flags=("--guidance-2",),
                help="Both guidance values must be above 1 for CFG, or both at most 1.",
            ),
            UIField("enable_vae_tiling", "Enable VAE tiling", "boolean", "Advanced", cli_flags=("--vae-tiling",)),
        ),
    )

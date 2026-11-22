"""Parameter defaults and controls; safe to inspect without GPU dependencies."""

from dataclasses import replace

from ..ui import UIField, common_fields, definition, frame_constraints
from ..shared.schedules import get_schedule

DEFAULTS = {
    "prompt": "",
    "negative_prompt": "",
    "positive_prefix": "",
    "positive_suffix": "",
    "trigger_words": [],
    "width": 832,
    "height": 480,
    "preserve_aspect_ratio": True,
    "num_frames": 124,
    "fps": 24,
    "num_videos_per_prompt": 1,
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
    "seed": None,
    "enable_vae_tiling": True,
    "sampling": {"sampler": "res_multistep", "noise_schedule": "simple"},
    "loras": [],
}

MODES = ("img2vid", "txt2vid")


def get_ui_definition(model, mode):
    requirements = model.defaults["requirements"]
    schedule = get_schedule("comfyui_minimax_h3", DEFAULTS["sampling"]["noise_schedule"])
    if requirements.get("runtime") != "comfyui" or requirements.get("guider", "BasicGuider") != "BasicGuider":
        raise ValueError("MiniMax H3 requires runtime comfyui and BasicGuider")
    if requirements.get("native_fps", 24) != 24:
        raise ValueError("MiniMax H3 requires native_fps 24")
    minimum, step = frame_constraints(requirements, native_step=17, native_offset=5)
    if minimum > 3600:
        raise ValueError("MiniMax frame_count requirements leave no supported frame count")
    allowed = requirements.get("allowed_sampling")
    if allowed is not None and (not isinstance(allowed, list) or DEFAULTS["sampling"] not in allowed):
        raise ValueError("This MiniMax workflow requires res_multistep with the simple schedule")
    fields = []
    for field in common_fields(model, dimension_base=32, video=True):
        if field.name == "guidance_scale":
            field = replace(
                field,
                label="Guidance scale",
                minimum=1,
                maximum=1,
                read_only=True,
                help="H3 BasicGuider uses CFG 1; there is no second expert.",
            )
        elif field.name == "negative_prompt":
            field = replace(field, read_only=True, help="BasicGuider does not use a negative prompt; leave empty.")
        fields.append(field)
    if mode == "img2vid":
        fields.append(
            UIField(
                "preserve_aspect_ratio",
                "Preserve source aspect ratio",
                "boolean",
                "Output",
                inherit_when_absent=True,
                help="Width and height set the output pixel budget; the source determines aspect ratio.",
            )
        )
    return definition(
        model,
        mode,
        workflow=model.defaults["workflow"],
        family="minimax",
        supported_modes=MODES,
        variant="minimax_h3",
        partial="controls/minimax.html",
        fields=(
            *fields,
            UIField(
                "num_frames",
                "Frame count",
                "integer",
                "Output",
                cli_flags=("--frames",),
                minimum=minimum,
                maximum=3600,
                step=step,
                help="H3 uses 17*n + 5 frames, independently of WAN's frame rule.",
            ),
            UIField(
                "fps",
                "Playback FPS",
                "number",
                "Output",
                minimum=24,
                maximum=24,
                read_only=True,
                help="MiniMax H3's native rate is 24 FPS.",
            ),
            UIField(
                "sampling.sampler", "Sampler", "text", "Sampling", read_only=True, help="Fixed to res_multistep for H3."
            ),
            UIField(
                "sampling.noise_schedule",
                "Noise schedule",
                "text",
                "Sampling",
                read_only=True,
                help=f"Fixed to {schedule.label} for H3.",
            ),
            UIField("enable_vae_tiling", "Enable VAE tiling", "boolean", "Advanced", cli_flags=("--vae-tiling",)),
        ),
    )

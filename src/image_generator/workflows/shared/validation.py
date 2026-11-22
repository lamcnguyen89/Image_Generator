"""Shared validation helpers."""

import math
import re
import subprocess

from image_generator.catalog.compatibility import lora_model_variants

from ..adapters import policy_for_model


def check_video_encoder():
    from imageio_ffmpeg import get_ffmpeg_exe

    try:
        subprocess.run([get_ffmpeg_exe(), "-version"], capture_output=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"MP4 encoder is unavailable: {exc}") from exc


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _nonnegative_number(value):
    return _finite_number(value) and value >= 0


def validate_common(
    request, defaults, *, family, mode, count_field, samplers=None, guidance_field="guidance_scale", dimension_base=16,
    sampling_validator=None,
):
    model, settings = request.model, request.settings
    if model.kind != "model" or model.family != family:
        raise ValueError(f"The {family.upper()} workflow requires a {family.upper()} catalog model")
    if request.mode != mode or mode not in model.defaults["supported_modes"]:
        raise ValueError(f"This {family.upper()} workflow supports only declared {mode} mode")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if not isinstance(request.negative_prompt, str):
        raise ValueError("negative_prompt must be a string")
    requirements = model.defaults["requirements"]
    multiple = requirements.get("dimension_multiple", 16)
    if type(multiple) is not int or multiple <= 0:
        raise ValueError("requirements.dimension_multiple must be a positive integer")
    multiple = math.lcm(dimension_base, multiple)
    for name in ("width", "height", "num_inference_steps", count_field):
        value = settings.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if name in {"width", "height"} and value % multiple:
            raise ValueError(f"{name} must be divisible by {multiple}")
    seed = settings.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("seed must be null or an integer between 0 and 2**63 - 1")
    if not _nonnegative_number(settings.get(guidance_field)):
        raise ValueError(f"{guidance_field} must be a finite number >= 0")
    sampling = settings.get("sampling")
    if not isinstance(sampling, dict) or set(sampling) != {"sampler", "noise_schedule"}:
        raise ValueError("sampling must contain sampler and noise_schedule")
    if sampling_validator is not None:
        sampling_validator(family, sampling, requirements)
    else:
        if sampling["sampler"] not in samplers or sampling["noise_schedule"] != "default":
            raise ValueError(f"Unsupported {family.upper()} sampling: {sampling!r}")
        allowed = requirements.get("allowed_sampling")
        if allowed is not None and (not isinstance(allowed, list) or sampling not in allowed):
            raise ValueError("Sampling is prohibited by this checkpoint's allowed_sampling")
    if not isinstance(requirements.get("scheduler_config", {}), dict):
        raise ValueError("requirements.scheduler_config must be an object")
    identifiers, names = set(), set()
    for selection in request.loras:
        asset = selection.asset
        if asset.kind != "lora" or asset.family != family:
            raise ValueError(f"LoRA {asset.id!r} does not declare {family.upper()} compatibility")
        variants = lora_model_variants(asset.defaults["compatibility"])
        if variants and model.defaults.get("model_variant") not in variants:
            raise ValueError(f"LoRA {asset.id!r} targets a different model variant")
        if asset.id in identifiers or selection.adapter_name in names:
            raise ValueError("Duplicate LoRA IDs or adapter names")
        identifiers.add(asset.id)
        names.add(selection.adapter_name)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", selection.adapter_name):
            raise ValueError("Invalid adapter name")
        minimum = policy_for_model(model).minimum_strength
        if not _finite_number(selection.strength) or (minimum is not None and selection.strength < minimum):
            suffix = f" >= {minimum}" if minimum is not None else ""
            raise ValueError(f"LoRA {asset.id!r} strength must be a finite number{suffix}")
        validate_weights(asset.asset_paths["weights"], f"LoRA {asset.id!r}")


def validate_weights(path, label):
    if path is None or not path.is_file():
        raise ValueError(f"{label} weights not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"{label} weights must be a .safetensors file")

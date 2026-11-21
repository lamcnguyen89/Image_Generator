"""Cross-field and asset validation without loading model runtimes."""

import json
import re
import struct
from pathlib import Path

from ..shared.validation import _nonnegative_number, validate_weights
from ..ui import FormValidationError
from .parameters import DEFAULTS, get_ui_definition

ENCODERS = {
    f"text_encoders/qwen3vl_32b_minimax_h3_{kind}.safetensors" for kind in ("bf16", "int8_convrot", "nvfp4_awq")
}


def component_sources(model):
    components = model.defaults.get("components")
    if not isinstance(components, dict):
        raise ValueError("MiniMax requires components.vae and components.text_encoder")
    result = {}
    for name in ("vae", "text_encoder"):
        source = components.get(name)
        if not isinstance(source, dict):
            raise ValueError(f"MiniMax requires components.{name}")
        weights = source.get("weights")
        if isinstance(weights, str) and weights.strip():
            result[name] = (model.manifest_path.parent / weights).expanduser().resolve()
        elif (
            name == "text_encoder"
            and source.get("repo_id") == "Comfy-Org/MiniMax-H3"
            and source.get("filename") in ENCODERS
        ):
            local = model.manifest_path.parents[2] / "text_encoders" / Path(source["filename"]).name
            result[name] = local if local.is_file() else dict(source)
        else:
            raise ValueError(f"components.{name} needs local weights or a documented MiniMax H3 encoder reference")
    return result


def safetensors_header(path):
    """Read architecture metadata only, never tensor payloads or GPU libraries."""
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        if not 2 <= size <= min(path.stat().st_size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path}")
    return header


def validate_form_settings(model, mode, settings):
    get_ui_definition(model, mode).validate_settings(settings)
    if settings["sampling"] != DEFAULTS["sampling"]:
        raise FormValidationError({"sampling": "H3 requires res_multistep with the simple schedule."})
    if settings["negative_prompt"]:
        raise FormValidationError({"negative_prompt": "BasicGuider does not support negative prompting."})
    if type(settings["preserve_aspect_ratio"]) is not bool:
        raise FormValidationError({"preserve_aspect_ratio": "Enter a boolean value."})


def validate(request):
    validate_form_settings(request.model, request.mode, request.settings)
    if not isinstance(request.prompt, str) or not request.prompt.strip() or request.negative_prompt:
        raise ValueError("MiniMax requires a nonempty prompt and no negative prompt")
    if request.mode == "img2vid":
        if set(request.inputs) != {"input_image"} or not request.inputs["input_image"].is_file():
            raise ValueError("MiniMax img2vid requires a local input_image file")
    elif request.inputs:
        raise ValueError("MiniMax txt2vid does not accept image inputs")
    if request.model.defaults.get("weights_component") != "transformer":
        raise ValueError("MiniMax weights_component must be transformer")
    checkpoint = request.model.asset_paths["weights"]
    validate_weights(checkpoint, "MiniMax checkpoint")
    header = safetensors_header(checkpoint)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not isinstance(metadata.get("config", "{}"), str):
        raise ValueError("MiniMax checkpoint metadata.config must be a JSON string")
    config = json.loads(metadata.get("config", "{}"))
    transformer = config.get("transformer") if isinstance(config, dict) else None
    if not isinstance(transformer, dict) or transformer.get("image_model") != "minimax_h3":
        raise ValueError("Checkpoint must declare image_model=minimax_h3 in its ComfyUI metadata")
    sources = component_sources(request.model)
    if not sources["vae"].is_file():
        raise ValueError(f"MiniMax VAE weights not found: {sources['vae']}")
    vae_header = safetensors_header(sources["vae"])
    if "latents_mean" not in vae_header or not any(
        key.startswith(("decoder.transformer_blocks.", "decoder.blocks.")) for key in vae_header
    ):
        raise ValueError("Select the MiniMax H3 video VAE, not a WAN or audio VAE")
    if isinstance(sources["text_encoder"], Path):
        validate_weights(sources["text_encoder"], "MiniMax text encoder")
    identifiers, paths, names = set(), set(), set()
    for selection in request.loras:
        asset = selection.asset
        if (
            asset.kind != "lora"
            or asset.family != "minimax"
            or asset.defaults["compatibility"].get("model_variant") != "minimax_h3"
        ):
            raise ValueError(f"LoRA {asset.id!r} needs explicit MiniMax H3 compatibility")
        path = asset.asset_paths["weights"]
        if asset.id in identifiers or path in paths or selection.adapter_name in names:
            raise ValueError("Duplicate MiniMax LoRA IDs, weights, or adapter names")
        identifiers.add(asset.id)
        paths.add(path)
        names.add(selection.adapter_name)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", selection.adapter_name) or not _nonnegative_number(selection.strength):
            raise ValueError("MiniMax LoRAs need valid adapter names and finite nonnegative strengths")
        validate_weights(path, f"MiniMax LoRA {asset.id!r}")

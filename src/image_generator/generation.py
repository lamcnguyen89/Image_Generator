"""Prepare catalog requests and save results independently of the web interface."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from .catalog import Catalog
from .workflows.base import GeneratedBatch, GenerationRequest, LoRASelection, SavedImage, SavedVideo
from .workflows.registry import create_workflow, get_definition


def prepare_generation(
    catalog: Catalog, model_id: str, *, mode: str | None = None,
    overrides: dict | None = None, include_trigger_words: bool = False,
    inputs: dict[str, str | Path] | None = None,
) -> GenerationRequest:
    """Snapshot and validate before GPU work. Trigger insertion is opt-in.

    Model prefix/suffix settings are applied. Adapter prompt additions and
    recommended generation presets are never merged into model settings.
    """
    try:
        model = deepcopy(catalog.get_model(model_id))
    except KeyError:
        raise ValueError(f"Unknown model ID {model_id!r}") from None
    workflow = get_definition(model.defaults["workflow"])
    settings = model.resolve_settings(workflow_defaults=workflow.defaults, request_overrides=overrides)
    loras = []
    for link in settings.get("loras", []):
        if not link["enabled"]:
            continue
        try:
            asset = deepcopy(catalog.get_lora(link["id"]))
        except KeyError:
            raise ValueError(f"Unknown LoRA ID {link['id']!r}") from None
        adapter_settings = workflow.adapter_policy.request_settings(asset, link)
        strength = link.get("strength", adapter_settings.get("strength"))
        if type(strength) not in (int, float):
            raise ValueError(f"LoRA {asset.id!r} needs a numeric strength")
        loras.append(LoRASelection(asset, adapter_settings, strength, f"adapter_{len(loras)}"))
    raw_prompt = settings.get("prompt", "")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    triggers = list(settings.get("trigger_words", [])) if include_trigger_words else []
    if include_trigger_words:
        for lora in loras:
            triggers.extend(lora.settings.get("trigger_words", []))
    prompt = " ".join(part.strip() for part in (
        settings.get("positive_prefix", ""), ", ".join(dict.fromkeys(triggers)),
        raw_prompt, settings.get("positive_suffix", ""),
    ) if part.strip())
    request = GenerationRequest(
        model, model.defaults["supported_modes"][0] if mode is None else mode, settings, prompt,
        settings.get("negative_prompt", ""), tuple(loras),
        {key: Path(value).expanduser().resolve() for key, value in (inputs or {}).items()},
    )
    workflow.validate(request)
    return request


def request_snapshot(request: GenerationRequest) -> dict:
    """JSON-ready provenance with no authentication credentials."""
    model = request.model
    weights = model.asset_paths["weights"]
    return deepcopy({
        "schema_version": 1, "mode": request.mode,
        "model": {
            "id": model.id, "display_name": model.display_name,
            "family": model.family, "workflow": model.defaults["workflow"],
            "model_variant": model.defaults.get("model_variant"),
            "weights_component": model.defaults.get("weights_component"),
            "base_model_id": model.defaults.get("base_model_id"),
            "weights": str(weights) if weights else None,
            "manifest_path": str(model.manifest_path), "user_path": str(model.user_path),
            "requirements": model.defaults["requirements"],
            "components": model.defaults.get("components", {}),
        },
        "settings": request.settings, "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "inputs": {key: str(value) for key, value in request.inputs.items()},
        "loras": [{
            "id": selection.asset.id, "display_name": selection.asset.display_name,
            "weights": str(selection.asset.asset_paths["weights"]),
            "manifest_path": str(selection.asset.manifest_path),
            "strength": selection.strength, "adapter_name": selection.adapter_name,
            "settings": selection.settings,
            "compatibility": selection.asset.defaults["compatibility"],
            **({"expert_target": selection.expert_target} if selection.expert_target is not None else {}),
        } for selection in request.loras],
    })


def save_batch(batch: GeneratedBatch, output_dir: str | Path) -> tuple[SavedImage | SavedVideo, ...]:
    """Save image or video outputs with metadata and actual per-output seeds."""
    if batch.videos:
        return save_video_batch(batch, output_dir)
    seeds = batch.metadata["seeds"]
    if not batch.images or len(batch.images) != len(seeds):
        raise ValueError("The generated image count does not match the recorded seeds")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, (image, seed) in enumerate(zip(batch.images, seeds, strict=True)):
        family = batch.metadata["model"]["workflow"]
        image_path = output_dir / f"{family}-seed-{seed}-{uuid4().hex}.png"
        metadata_path = image_path.with_suffix(".json")
        metadata = {
            **deepcopy(batch.metadata), "seed": seed, "image_index": index,
            "image_file": image_path.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
        try:
            image.save(image_path, format="PNG")
            metadata_path.write_text(payload + "\n", encoding="utf-8")
        except Exception:
            image_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
        saved.append(SavedImage(image_path, metadata_path, seed))
    return tuple(saved)


def export_video(frames, path: Path, fps):
    """Stream NumPy, PIL, or ComfyUI tensor frames without importing Diffusers."""
    import imageio.v2 as imageio
    import numpy as np

    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            if hasattr(frame, "detach"):
                frame = frame.detach().float().cpu().numpy()
            pixels = np.asarray(frame)
            if np.issubdtype(pixels.dtype, np.floating):
                pixels = (pixels.clip(0, 1) * 255).round().astype(np.uint8)
            writer.append_data(pixels)


def save_video_batch(batch: GeneratedBatch, output_dir: str | Path) -> tuple[SavedVideo, ...]:
    seeds = batch.metadata["seeds"]
    if batch.images or len(batch.videos) != len(seeds):
        raise ValueError("Video outputs require matching seeds")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for frames, seed in zip(batch.videos, seeds, strict=True):
        frame = frames[0]
        if hasattr(frame, "shape"):
            height, width = frame.shape[:2]
        else:
            width, height = frame.size
        fps = batch.metadata["settings"]["fps"]
        family = batch.metadata["model"]["workflow"]
        video_path = output_dir / f"{family}-seed-{seed}-{uuid4().hex}.mp4"
        metadata_path = video_path.with_suffix(".json")
        conditioning_path = video_path.with_suffix(".input.png") if batch.conditioning_image is not None else None
        metadata = {
            **deepcopy(batch.metadata), "seed": seed, "video_file": video_path.name,
            "conditioning_image": conditioning_path.name if conditioning_path else None,
            "output_video": {"width": int(width), "height": int(height), "frame_count": len(frames),
                             "fps": fps, "duration_seconds": len(frames) / fps},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
        try:
            export_video(frames, video_path, metadata["settings"]["fps"])
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise OSError("The encoder did not produce a video file")
            if conditioning_path is not None:
                batch.conditioning_image.save(conditioning_path, format="PNG")
            metadata_path.write_text(payload + "\n", encoding="utf-8")
        except Exception:
            for path in (video_path, metadata_path, conditioning_path):
                if path is not None:
                    path.unlink(missing_ok=True)
            raise
        saved.append(SavedVideo(video_path, metadata_path, seed, conditioning_path))
    return tuple(saved)


def run_generation(request: GenerationRequest, output_dir: str | Path, **runtime_options) -> tuple[SavedImage | SavedVideo, ...]:
    """Run one request and release its pipeline, suitable for a CLI invocation.

    A future worker can retain a workflow instance and call generate() directly
    to reuse its pipeline across serialized requests.
    """
    workflow = create_workflow(request.model.defaults["workflow"], **runtime_options)
    try:
        saved = []
        for batch in workflow.generate_batches(request):
            saved.extend(save_batch(batch, output_dir))
            del batch  # Release decoded video frames before requesting another variation.
        return tuple(saved)
    finally:
        workflow.close()

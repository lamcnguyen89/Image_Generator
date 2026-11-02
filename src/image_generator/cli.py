"""Shared CLI wiring for catalog-backed transformer image and video generation."""

import argparse
import json
import os
from pathlib import Path
import sys

from .catalog import discover_catalog
from .generation import prepare_generation, request_snapshot, run_generation
from .workflows.base import SavedVideo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_IDS = {
    "flux": "model_6e8d8d55-6a4b-4cf9-b91b-3a4a5beaab1f",
    "qwen_2_1": "model_73c73d8d-ba6e-462f-b3da-ac775c627979",
    "krea2": "model_43359779-6bc0-4f45-8b70-39a2edff93fa",
    "wan": "model_95f8fb96-fc3b-4cf3-8b6f-e95e8cb5209b",
    "minimax": "model_b065ec3d-cd27-4cb3-bcab-48797bee8b6b",
}


def seed_value(value):
    if value.lower() == "random":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("seed must be an integer or 'random'") from None


def lora_value(value):
    identifier, separator, strength = value.partition(":")
    if not identifier.strip():
        raise argparse.ArgumentTypeError("LoRA ID must not be empty")
    link = {"id": identifier, "enabled": True}
    if separator:
        try:
            link["strength"] = float(strength)
        except ValueError:
            raise argparse.ArgumentTypeError("LoRA must be ID or ID:STRENGTH") from None
    return link


def run_family(family, argv=None):
    video = family in {"wan", "minimax"}
    parser = argparse.ArgumentParser(description=f"Generate {family.upper()} {'videos' if video else 'images'} from catalog settings.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "Models/Model_Families")
    parser.add_argument("--model-id", default=MODEL_IDS[family])
    parser.add_argument("--prompt", required=True)
    fields = ["prompt", "negative_prompt", "width", "height", "num_inference_steps", "seed"]
    parser.add_argument("--negative-prompt", default=argparse.SUPPRESS)
    for flag, dest, kind in (("width", "width", int), ("height", "height", int),
                             ("steps", "num_inference_steps", int)):
        parser.add_argument(f"--{flag}", dest=dest, type=kind, default=argparse.SUPPRESS)
    if family != "qwen_2_1":
        parser.add_argument("--guidance", dest="guidance_scale", type=float, default=argparse.SUPPRESS)
        fields.append("guidance_scale")
    parser.add_argument("--seed", type=seed_value, default=argparse.SUPPRESS)
    if video:
        parser.add_argument("--input-image", type=Path, required=True)
        parser.add_argument("--num-videos", dest="num_videos_per_prompt", type=int, default=1)
        parser.add_argument("--frames", dest="num_frames", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--fps", type=float, default=argparse.SUPPRESS)
        if family == "wan":
            parser.add_argument("--guidance-2", dest="guidance_scale_2", type=float, default=argparse.SUPPRESS)
            fields.append("guidance_scale_2")
        parser.add_argument("--preserve-aspect-ratio", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        parser.add_argument("--vae-tiling", dest="enable_vae_tiling", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        fields.extend(("num_videos_per_prompt", "num_frames", "fps", "preserve_aspect_ratio", "enable_vae_tiling"))
    else:
        parser.add_argument("--num-images", dest="num_images_per_prompt", type=int, default=1)
        if family in {"flux", "qwen_2_1"}:
            parser.add_argument("--true-cfg-scale", type=float, default=argparse.SUPPRESS)
            fields.append("true_cfg_scale")
        if family == "qwen_2_1":
            parser.add_argument("--sigmas", type=float, nargs="+", default=argparse.SUPPRESS)
            parser.add_argument("--kv-cache", dest="use_kv_cache", action=argparse.BooleanOptionalAction,
                                default=argparse.SUPPRESS)
            fields.extend(("sigmas", "use_kv_cache"))
        else:
            parser.add_argument("--max-sequence-length", type=int, default=argparse.SUPPRESS)
            fields.append("max_sequence_length")
        fields.append("num_images_per_prompt")
    samplers = {"wan": ["model_default"], "minimax": ["res_multistep"],
                "flux": ["flow_match_euler", "model_default"], "krea2": ["flow_match_euler", "model_default"],
                "qwen_2_1": ["flow_match_euler", "model_default"]}
    parser.add_argument("--sampler", choices=samplers[family])
    if family == "minimax":
        parser.add_argument("--comfyui-dir", type=Path, help="ComfyUI checkout with native H3 support (default: build/ComfyUI)")
    adapters = parser.add_mutually_exclusive_group()
    adapters.add_argument("--lora", type=lora_value, action="append", metavar="ID[:STRENGTH]")
    adapters.add_argument("--no-loras", action="store_true")
    parser.add_argument("--include-trigger-words", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / ("Videos" if video else "Images"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and print request JSON without importing model libraries")
    args = parser.parse_args(argv)
    overrides = {key: getattr(args, key) for key in fields if hasattr(args, key)}
    if args.sampler is not None:
        overrides["sampling"] = {"sampler": args.sampler}
    if args.lora is not None:
        overrides["loras"] = args.lora
    elif args.no_loras:
        overrides["loras"] = []
    try:
        catalog = discover_catalog(args.root)
        try:
            model = catalog.get_model(args.model_id)
        except KeyError:
            raise ValueError(f"Unknown model ID {args.model_id!r}") from None
        if model.defaults["workflow"] != family:
            raise ValueError(f"This runner requires a {family.upper()} catalog model")
        request = prepare_generation(
            catalog, args.model_id, overrides=overrides, include_trigger_words=args.include_trigger_words,
            inputs={"input_image": args.input_image} if video else None,
        )
        if args.dry_run:
            print(json.dumps(request_snapshot(request), indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        cache_root = Path(os.getenv("CACHE_LOCATION") or PROJECT_ROOT / "Models").expanduser()
        if not cache_root.is_absolute():
            cache_root = PROJECT_ROOT / cache_root
        runtime_options = {"comfyui_dir": args.comfyui_dir} if family == "minimax" else {}
        outputs = run_generation(
            request, args.output_dir, device=args.device, dtype=args.dtype,
            cache_dir=args.cache_dir if args.cache_dir is not None else cache_root / "hub",
            local_files_only=args.local_files_only,
            token=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or None,
            **runtime_options,
        )
        for output in outputs:
            if isinstance(output, SavedVideo):
                print(f"Video: {output.video_path}\nConditioning image: {output.conditioning_path}")
            else:
                print(f"Image: {output.image_path}")
            print(f"Metadata: {output.metadata_path}\nSeed: {output.seed}")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1
    return 0

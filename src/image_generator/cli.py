"""Shared CLI wiring for catalog-backed transformer image and video generation."""

import argparse
import json
import os
from pathlib import Path
import sys
import logging

from .catalog import discover_catalog
from .diagnostics import configure_logging
from .generation import prepare_generation, request_snapshot, run_generation
from .workflows.base import SavedVideo


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    """Run any registered workflow using its parameter and input declarations."""
    from .workflows.registry import get_definition
    from .workflows.runtime import resolve_runtime_options
    from .workflows.ui import set_setting

    definition = get_definition(family)
    argv = sys.argv[1:] if argv is None else argv
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--root", type=Path, default=PROJECT_ROOT / "Models/Model_Families")
    bootstrap.add_argument("--model-id", default=definition.default_model_id)
    bootstrap.add_argument("--mode")
    selected, _ = bootstrap.parse_known_args(argv)
    # Read logging options without changing the environment during dry runs.
    try:
        from dotenv import dotenv_values
    except ImportError:
        logging_options = {}
    else:
        logging_options = dotenv_values(PROJECT_ROOT / ".env")
    configure_logging(
        os.getenv("IMAGE_GENERATOR_LOG_DIR") or logging_options.get("IMAGE_GENERATOR_LOG_DIR"),
        os.getenv("IMAGE_GENERATOR_LOG_LEVEL") or logging_options.get("IMAGE_GENERATOR_LOG_LEVEL"),
    )
    try:
        catalog = discover_catalog(selected.root)
        candidates = [model for model in catalog.models.values() if model.defaults["workflow"] == family]
        model_id = selected.model_id or (candidates[0].id if candidates else None)
        try:
            model = catalog.get_model(model_id)
        except KeyError:
            raise ValueError(f"Unknown model ID {model_id!r}") from None
        if model.defaults["workflow"] != family:
            raise ValueError(f"This runner requires workflow {family!r}")
        mode = selected.mode or next(
            (name for name in definition.ui_modes if name in model.defaults["supported_modes"]), None
        )
        ui = definition.get_ui_definition(model, mode)
        mode_definition = definition.mode(mode)
    except (ValueError, OSError) as error:
        logging.getLogger(__name__).exception("Generation preparation failed")
        print(f"Generation failed: {error}", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=f"Generate with the {family} workflow.", parents=[bootstrap])
    for parameter in ui.fields:
        canonical = "--" + parameter.name.replace("_", "-").replace(".", "-")
        flags = list(dict.fromkeys([*parameter.cli_flags, canonical]))
        options = {"dest": parameter.name, "default": argparse.SUPPRESS, "help": parameter.help or parameter.label}
        if parameter.scope == "inputs":
            options.update(type=Path, required=parameter.required)
        elif parameter.type == "boolean":
            options["action"] = argparse.BooleanOptionalAction
        elif parameter.type == "integer":
            options["type"] = seed_value if parameter.name == "seed" else int
        elif parameter.type == "number":
            options["type"] = float
        elif parameter.type == "number_list":
            options.update(type=float, nargs="+")
        elif parameter.type == "string_list":
            options["nargs"] = "+"
        if parameter.choices:
            options["choices"] = tuple(dict(parameter.choices))
        if parameter.required and parameter.scope == "settings":
            options["required"] = True
        parser.add_argument(*flags, **options)
    adapters = parser.add_mutually_exclusive_group()
    adapters.add_argument("--lora", type=lora_value, action="append", metavar="ID[:STRENGTH]")
    adapters.add_argument("--no-loras", action="store_true")
    parser.add_argument("--include-trigger-words", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / ("Videos" if mode_definition.output_kind == "video" else "Images"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    for option in definition.extra_runtime_options:
        parser.add_argument("--" + option.name.replace("_", "-"), type=option.type, help=option.help)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print request JSON without loading models")
    args = parser.parse_args(argv)
    values = vars(args)
    overrides, inputs = {}, {}
    for parameter in ui.fields:
        if parameter.name not in values:
            continue
        value = values[parameter.name]
        if parameter.scope == "inputs":
            inputs[parameter.name] = value
        else:
            set_setting(overrides, parameter.name, value)
    if args.lora is not None:
        overrides["loras"] = args.lora
    elif args.no_loras:
        overrides["loras"] = []
    try:
        request = prepare_generation(
            catalog,
            model.id,
            mode=mode,
            overrides=overrides,
            include_trigger_words=args.include_trigger_words,
            inputs=inputs,
        )
        if args.dry_run:
            print(json.dumps(request_snapshot(request), indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        configure_logging()
        cache_root = Path(os.getenv("CACHE_LOCATION") or PROJECT_ROOT / "Models").expanduser()
        if not cache_root.is_absolute():
            cache_root = PROJECT_ROOT / cache_root
        options = {
            "device": args.device,
            "dtype": args.dtype,
            "cache_dir": args.cache_dir if args.cache_dir is not None else cache_root / "hub",
            "local_files_only": args.local_files_only,
            "token": os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or None,
        }
        options.update({option.name: values.get(option.name) for option in definition.extra_runtime_options})
        outputs = run_generation(request, args.output_dir, **resolve_runtime_options(definition, options))
        for output in outputs:
            if isinstance(output, SavedVideo):
                print(f"Video: {output.video_path}\nConditioning image: {output.conditioning_path}")
            else:
                print(f"Image: {output.image_path}")
            print(f"Metadata: {output.metadata_path}\nSeed: {output.seed}")
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        logging.getLogger(__name__).exception("Generation failed")
        print(f"Generation failed: {error}", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    """Dispatch any registered integration without creating a family script."""
    from .workflows.registry import WORKFLOWS

    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Run a registered image/video workflow.")
    parser.add_argument("workflow", choices=tuple(WORKFLOWS))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    selected = parser.parse_args(argv[:1])
    return run_family(selected.workflow, argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

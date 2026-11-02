#!/usr/bin/env python3
"""Check dependencies and small CUDA operations without loading model weights."""

import argparse
import importlib
from pathlib import Path
import platform
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = {
    "torch": ("torch", ()),
    "torchvision": ("torchvision", ()),
    "diffusers": (
        "diffusers",
        (
            "StableDiffusionXLPipeline",
            "FluxPipeline",
            "FluxTransformer2DModel",
            "QwenImage21Pipeline",
            "QwenImage21Transformer2DModel",
            "Krea2Pipeline",
            "Krea2Transformer2DModel",
            "WanImageToVideoPipeline",
            "AutoencoderKLWan",
        ),
    ),
    "transformers": (
        "transformers",
        ("CLIPTextModel", "CLIPImageProcessor", "CLIPVisionModel", "T5EncoderModel", "UMT5EncoderModel"),
    ),
    "accelerate": ("accelerate", ()),
    "peft": ("peft", ()),
    "safetensors": ("safetensors", ()),
    "huggingface-hub": ("huggingface_hub", ()),
    "Pillow": ("PIL.Image", ()),
    "numpy": ("numpy", ()),
    "python-dotenv": ("dotenv", ("load_dotenv",)),
    "sentencepiece": ("sentencepiece", ()),
    "ftfy": ("ftfy", ()),
    "imageio": ("imageio", ()),
    "imageio-ffmpeg": ("imageio_ffmpeg", ("get_ffmpeg_exe",)),
    "Flask": ("flask", ("Flask",)),
}


def check_python():
    expected = (PROJECT_ROOT / ".python-version").read_text().strip()
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python: {platform.python_version()} | {sys.executable}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    if actual != expected:
        return [f"Expected Python {expected}; this interpreter uses {actual}."]
    return []


def check_imports():
    modules = {}
    errors = []
    for package, (module_name, attributes) in IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
            for name in attributes:
                getattr(module, name)  # Resolve lazy imports, including pipeline dependencies.
            modules[package] = module
            print(f"OK: {package}")
        except Exception as exc:
            errors.append(f"Cannot use {package}: {exc}")
    return modules, errors


def check_ffmpeg(module):
    try:
        executable = module.get_ffmpeg_exe()
        subprocess.run(
            [executable, "-version"], check=True, capture_output=True, timeout=10
        )
        print(f"FFmpeg: {executable}")
    except Exception as exc:
        return [f"FFmpeg is unavailable: {exc}"]
    return []


def check_cuda(torch, allow_cpu=False):
    try:
        print(f"PyTorch: {torch.__version__} | compiled CUDA: {torch.version.cuda}")
        if not torch.cuda.is_available():
            if allow_cpu:
                print("SKIP: CUDA unavailable; GPU compatibility has not been verified.")
                return []
            return ["CUDA is unavailable. Use the compatible GPU PyTorch build and driver stack."]
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if not torch.cuda.is_bf16_supported():
            return ["CUDA BF16 support is required by the current FLUX/Qwen 2.1/Krea2/WAN workflows."]
        with torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                matrix = torch.ones((16, 16), device="cuda", dtype=dtype)
                result = matrix @ matrix
                if result[0, 0].item() != 16:
                    return [f"CUDA matrix multiplication returned an unexpected result for {dtype}."]
                query = torch.ones((1, 1, 16, 64), device="cuda", dtype=dtype)
                attention = torch.nn.functional.scaled_dot_product_attention(query, query, query)
                torch.cuda.synchronize()
                if not torch.isfinite(attention).all().item():
                    return [f"PyTorch SDPA returned nonfinite values for {dtype}."]
                print(f"OK: CUDA matrix multiplication and native SDPA ({dtype})")
    except Exception as exc:
        return [f"CUDA check failed: {exc}"]
    return []


def check_minimax(comfyui_dir=None, allow_cpu=False):
    directory = Path(comfyui_dir or PROJECT_ROOT / "build/ComfyUI").expanduser().resolve()
    if not (directory / "comfy_extras/nodes_minimax_h3.py").is_file():
        return [f"MiniMax ComfyUI runtime missing native H3 support at {directory}. Run bash scripts/setup.sh."]

    errors = []
    added_path = str(directory) not in sys.path
    if added_path:
        sys.path.insert(0, str(directory))
    try:
        options = importlib.import_module("comfy.options")
        options.enable_args_parsing(False)
        # Select CPU before importing modules that initialize device management.
        if allow_cpu:
            importlib.import_module("comfy.cli_args").args.cpu = True
        sd = importlib.import_module("comfy.sd")
        samplers = importlib.import_module("comfy.samplers")
        importlib.import_module("comfy_extras.nodes_minimax_h3")
        importlib.import_module("comfy_extras.nodes_custom_sampler")
        if not hasattr(sd.CLIPType, "MINIMAX"):
            errors.append("MiniMax ComfyUI runtime lacks CLIPType.MINIMAX encoder support.")
        if "res_multistep" not in samplers.SAMPLER_NAMES:
            errors.append("MiniMax ComfyUI runtime lacks the res_multistep sampler.")
        if "simple" not in samplers.SCHEDULER_NAMES:
            errors.append("MiniMax ComfyUI runtime lacks the simple noise schedule.")
        if not errors:
            print(f"OK: MiniMax native H3 nodes, text encoder, and sampling support ({directory})")
    except Exception as exc:
        errors.append(f"Cannot use MiniMax ComfyUI runtime: {exc}")
    finally:
        if added_path:
            sys.path.remove(str(directory))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Allow dependency checks without CUDA; does not validate GPU generation.",
    )
    parser.add_argument(
        "--comfyui-dir", type=Path,
        help="MiniMax ComfyUI checkout (default: build/ComfyUI in this project).",
    )
    args = parser.parse_args(argv)
    errors = check_python()
    modules, import_errors = check_imports()
    errors.extend(import_errors)
    if "imageio-ffmpeg" in modules:
        errors.extend(check_ffmpeg(modules["imageio-ffmpeg"]))
    if "torch" in modules:
        errors.extend(check_cuda(modules["torch"], allow_cpu=args.allow_cpu))
    errors.extend(check_minimax(args.comfyui_dir, allow_cpu=args.allow_cpu))
    if errors:
        print("\nEnvironment checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nEnvironment checks passed. Full model generation still needs separate validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

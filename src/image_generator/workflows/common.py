"""Shared validation and pipeline ownership for transformer workflows."""

from copy import deepcopy
import gc
from importlib import import_module
import math
from pathlib import Path
import re
import secrets
import subprocess

from image_generator.catalog.settings import lora_strength_minimum

from .base import Workflow


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


def scheduler_config_snapshot(value):
    """Represent library infinity/NaN sentinels as strings in strict JSON."""
    if isinstance(value, dict):
        return {key: scheduler_config_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [scheduler_config_snapshot(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def validate_common(request, defaults, *, family, mode, count_field, samplers,
                    guidance_field="guidance_scale", dimension_base=16):
    model, settings = request.model, request.settings
    if model.kind != "model" or model.family != family or model.defaults["workflow"] != family:
        raise ValueError(f"The {family.upper()} workflow requires a {family.upper()} catalog model")
    if request.mode != mode or mode not in model.defaults["supported_modes"]:
        raise ValueError(f"This {family.upper()} workflow supports only declared {mode} mode")
    unknown = settings.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unsupported {family.upper()} settings: {', '.join(sorted(unknown))}")
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
        variant = asset.defaults["compatibility"].get("model_variant")
        if variant and variant != model.defaults.get("model_variant"):
            raise ValueError(f"LoRA {asset.id!r} targets a different model variant")
        if asset.id in identifiers or selection.adapter_name in names:
            raise ValueError("Duplicate LoRA IDs or adapter names")
        identifiers.add(asset.id)
        names.add(selection.adapter_name)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", selection.adapter_name):
            raise ValueError("Invalid adapter name")
        minimum = lora_strength_minimum(family)
        if not _finite_number(selection.strength) or (minimum is not None and selection.strength < minimum):
            suffix = f" >= {minimum}" if minimum is not None else ""
            raise ValueError(f"LoRA {asset.id!r} strength must be a finite number{suffix}")
        validate_weights(asset.asset_paths["weights"], f"LoRA {asset.id!r}")


def validate_weights(path, label):
    if path is None or not path.is_file():
        raise ValueError(f"{label} weights not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"{label} weights must be a .safetensors file")


def file_fingerprint(path):
    if path is None:
        return None
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def image_seeds(settings, count):
    seed = settings["seed"]
    return [secrets.randbelow(2**63) if seed is None else (seed + index) % (2**63) for index in range(count)]


class TransformerWorkflow(Workflow):
    """Lazy runtime imports and reusable pipeline ownership; no web dependencies."""

    def __init__(self, *, device="cuda:0", dtype=None, cache_dir=None, local_files_only=False, token=None):
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
        self.device = device
        self.dtype = dtype or ("float32" if device == "cpu" else "bfloat16")
        if self.dtype not in {"float16", "bfloat16", "float32"} or (device == "cpu" and self.dtype == "float16"):
            raise ValueError("Unsupported dtype; CPU generation requires float32 or bfloat16")
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self.local_files_only = local_files_only
        self._token = token
        self._torch = self._diffusers = self._pipeline = self._model_key = None
        self._scheduler_class = self._scheduler_config = None
        self._has_adapters = False

    def _load_runtime(self):
        self.report_progress("loading_runtime")
        if self._torch is None:
            self._torch = import_module("torch")
        if self.device.startswith("cuda"):
            if not self._torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Run on the GPU host, or explicitly use --device cpu.")
            if self.dtype == "bfloat16":
                with self._torch.cuda.device(self.device):
                    if not self._torch.cuda.is_bf16_supported():
                        raise RuntimeError("The selected CUDA device does not support bfloat16")
        if self._diffusers is None:
            self._diffusers = import_module("diffusers")

    def _load_options(self):
        options = {"dtype": getattr(self._torch, self.dtype), "local_files_only": self.local_files_only,
                   "use_safetensors": True, "low_cpu_mem_usage": True}
        if self.cache_dir is not None:
            options["cache_dir"] = str(self.cache_dir)
        if self._token:
            options["token"] = self._token
        return options

    def _ensure_pipeline(self, request, fingerprint):
        key = (request.model.id, request.model.defaults.get("base_model_id"), fingerprint)
        if self._pipeline is not None and self._model_key == key:
            self.report_progress("reusing_model")
            return
        self.report_progress("switching_model" if self._pipeline is not None else "loading_model")
        self.close()
        self._pipeline = self._build_pipeline(request)
        self._pipeline.to(self.device)
        self._scheduler_class = type(self._pipeline.scheduler)
        self._scheduler_config = deepcopy(dict(self._pipeline.scheduler.config))
        self._model_key = key

    def _configure_scheduler(self, request, scheduler_class=None):
        cls = scheduler_class or self._scheduler_class
        self._pipeline.scheduler = cls.from_config(
            self._scheduler_config, **request.model.defaults["requirements"].get("scheduler_config", {}),
        )

    def _load_lora(self, selection, **options):
        path = selection.asset.asset_paths["weights"]
        self._pipeline.load_lora_weights(
            str(path.parent), weight_name=path.name, adapter_name=selection.adapter_name,
            local_files_only=True, **options,
        )

    def _configure_loras(self, request, *, restore_overwritten=False, targeted=False):
        self.report_progress("resetting_loras")
        if self._has_adapters:
            options = {"reset_to_overwritten_params": True} if restore_overwritten else {}
            self._pipeline.unload_lora_weights(**options)
            self._has_adapters = False
        self.report_progress("loading_loras", 0, len(request.loras))
        for index, selection in enumerate(request.loras):
            self._has_adapters = True
            options = {}
            if targeted:
                options["load_into_transformer_2"] = selection.asset.defaults["compatibility"]["expert_target"] == "low_noise"
            try:
                self._load_lora(selection, **options)
            except Exception as exc:
                raise RuntimeError(f"Could not load LoRA {selection.asset.id!r}; check model and loader compatibility: {exc}") from exc
            self.report_progress("loading_loras", index + 1, len(request.loras))
        if targeted:
            for target, component in (("high_noise", "transformer"), ("low_noise", "transformer_2")):
                selections = [item for item in request.loras if item.asset.defaults["compatibility"]["expert_target"] == target]
                if selections:
                    getattr(self._pipeline, component).set_adapters(
                        [item.adapter_name for item in selections], weights=[float(item.strength) for item in selections],
                    )
        elif request.loras:
            self._pipeline.set_adapters(
                [item.adapter_name for item in request.loras], adapter_weights=[float(item.strength) for item in request.loras],
            )
        self.report_progress("loras_ready", len(request.loras), len(request.loras))

    def _synchronize(self):
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize(self.device)

    def _runtime_metadata(self, seconds):
        return {"device": self.device, "dtype": self.dtype, "torch_version": self._torch.__version__,
                "diffusers_version": self._diffusers.__version__, "generation_seconds": seconds}

    def _scheduler_metadata(self):
        return {"class_name": type(self._pipeline.scheduler).__name__,
                "config": scheduler_config_snapshot(dict(self._pipeline.scheduler.config))}

    def close(self):
        self._pipeline = self._model_key = self._scheduler_class = self._scheduler_config = None
        self._has_adapters = False
        gc.collect()
        if self._torch is not None and self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

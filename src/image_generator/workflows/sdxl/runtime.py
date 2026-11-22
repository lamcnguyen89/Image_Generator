"""Instance-owned generation implementation."""

import gc
import re
import secrets
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from time import perf_counter

from ..base import GeneratedBatch, GenerationRequest, Workflow
from ..shared.metadata import scheduler_metadata
from ..shared.samplers import create_scheduler
from .parameters import DEFAULTS, MODES


class SDXLWorkflow(Workflow):
    workflow_id = "sdxl"
    sampler_workflow_id = "sdxl"
    defaults = DEFAULTS
    ui_modes = MODES

    def __init__(
        self,
        *,
        device="cuda:0",
        dtype=None,
        cache_dir=None,
        local_files_only=False,
        token=None,
    ):
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
        self.device = device
        self.dtype = dtype or ("float32" if device == "cpu" else "float16")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if device == "cpu" and self.dtype == "float16":
            raise ValueError("CPU generation requires float32 or bfloat16")
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self.local_files_only = local_files_only
        self._token = token
        self._torch = None
        self._diffusers = None
        self._pipeline = None
        self._model_key = None
        self._scheduler_class = None
        self._scheduler_config = None
        self._has_adapters = False

    @staticmethod
    def _validate_weights(path, label, *, optional=False):
        if path is None and optional:
            return
        if path is None or not path.is_file():
            raise ValueError(f"{label} weights not found: {path}")
        if path.suffix.lower() != ".safetensors":
            raise ValueError(f"{label} weights must be a .safetensors file")

    def _load_runtime(self):
        if self._torch is None:
            self._torch = import_module("torch")
        if self.device.startswith("cuda"):
            if not self._torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable. Run on the configured GPU host, or explicitly use --device cpu."
                )
            if self.dtype == "bfloat16" and not self._torch.cuda.is_bf16_supported():
                raise RuntimeError("The selected CUDA device does not support bfloat16")
        if self._diffusers is None:
            self._diffusers = import_module("diffusers")

    def _load_pipeline(self, request):
        path = request.model.asset_paths["weights"]
        fingerprint = None
        if path is not None:
            stat = path.stat()
            fingerprint = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = (request.model.id, request.model.defaults.get("base_model_id"), fingerprint)
        if self._pipeline is not None and self._model_key == key:
            self.report_progress("reusing_model")
            return True
        if self._pipeline is not None:
            self.report_progress("switching_model")
        self.close()
        self.report_progress("loading_model")
        self._pipeline = self._build_pipeline(request)
        self._pipeline.to(self.device)
        self._scheduler_class = type(self._pipeline.scheduler)
        self._scheduler_config = deepcopy(dict(self._pipeline.scheduler.config))
        self._model_key = key
        return False

    def _build_pipeline(self, request):
        """Override pipeline construction without changing cache/release behavior."""
        options = {
            "dtype": getattr(self._torch, self.dtype),
            "use_safetensors": True,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir is not None:
            options["cache_dir"] = str(self.cache_dir)
        if self._token:
            options["token"] = self._token
        path = request.model.asset_paths["weights"]
        pipeline_class = self._diffusers.StableDiffusionXLPipeline
        if path is None:
            if self.dtype == "float16":
                options["variant"] = "fp16"
            return pipeline_class.from_pretrained(request.model.defaults["base_model_id"], **options)
        else:
            if request.model.defaults.get("base_model_id"):
                options["config"] = request.model.defaults["base_model_id"]
            return pipeline_class.from_single_file(str(path), **options)

    def _configure_scheduler(self, request, *, generators):
        # Start from the loader's config, never the previous job's scheduler.
        self._pipeline.scheduler = create_scheduler(
            self.sampler_workflow_id, request.settings["sampling"], request.model.defaults["requirements"],
            diffusers=self._diffusers, original_class=self._scheduler_class,
            original_config=self._scheduler_config, generators=generators,
        )

    def _configure_loras(self, request):
        reset = self._has_adapters
        if self._has_adapters:
            self.report_progress("resetting_loras")
            self._pipeline.unload_lora_weights()
            self._has_adapters = False
        for index, selection in enumerate(request.loras):
            self.report_progress("loading_loras", index, len(request.loras))
            self._has_adapters = True  # Failed loading may already have modified modules.
            path = selection.asset.asset_paths["weights"]
            try:
                self._pipeline.load_lora_weights(
                    str(path.parent),
                    weight_name=path.name,
                    adapter_name=selection.adapter_name,
                    local_files_only=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load LoRA {selection.asset.id!r}; verify checkpoint and loader compatibility: {exc}"
                ) from exc
        if request.loras:
            self._pipeline.set_adapters(
                [selection.adapter_name for selection in request.loras],
                adapter_weights=[float(selection.strength) for selection in request.loras],
            )
        self.report_progress("loras_ready", len(request.loras), len(request.loras))
        return reset

    def generate(self, request: GenerationRequest) -> GeneratedBatch:
        self.validate(request)
        try:
            self.report_progress("loading_runtime")
            self._load_runtime()
            model_reused = self._load_pipeline(request)
            loras_reset = self._configure_loras(request)
            settings = request.settings
            seed = settings["seed"]
            seeds = [
                secrets.randbelow(2**63) if seed is None else (seed + index) % (2**63)
                for index in range(settings["num_images_per_prompt"])
            ]
            generators = [self._torch.Generator(device=self.device).manual_seed(value) for value in seeds]
            self._configure_scheduler(request, generators=generators)
            arguments = {
                name: settings[name]
                for name in (
                    "width",
                    "height",
                    "num_inference_steps",
                    "guidance_scale",
                    "num_images_per_prompt",
                )
            }
            if settings["clip_skip"] is not None:
                arguments["clip_skip"] = settings["clip_skip"]
            if getattr(self, "_progress_callback", None) is not None:

                def on_step_end(pipeline, step, timestep, callback_kwargs):
                    total = getattr(pipeline, "num_timesteps", settings["num_inference_steps"])
                    if type(total) is not int or total <= 0:
                        total = settings["num_inference_steps"]
                    current = min(step + 1, total)
                    self.report_progress("denoising", current, total)
                    if current == total:
                        self.report_progress("decoding")
                    return callback_kwargs

                arguments["callback_on_step_end"] = on_step_end
                arguments["callback_on_step_end_tensor_inputs"] = []
            self.report_progress("denoising", 0, settings["num_inference_steps"])
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize(self.device)
            start = perf_counter()
            with self._torch.inference_mode():
                images = tuple(
                    self._pipeline(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        generator=generators,
                        **arguments,
                    ).images
                )
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize(self.device)
            seconds = perf_counter() - start
            if len(images) != len(seeds):
                raise RuntimeError("The pipeline returned an unexpected number of images")
            from image_generator.generation import request_snapshot

            return GeneratedBatch(
                images,
                {
                    **request_snapshot(request),
                    "seeds": seeds,
                    "scheduler": scheduler_metadata(self._pipeline.scheduler),
                    "runtime": {
                        "device": self.device,
                        "dtype": self.dtype,
                        "torch_version": self._torch.__version__,
                        "diffusers_version": self._diffusers.__version__,
                        "generation_seconds": seconds,
                        "model_reused": model_reused,
                        "loras_reset": loras_reset,
                    },
                },
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._pipeline = None
        self._model_key = None
        self._scheduler_class = None
        self._scheduler_config = None
        self._has_adapters = False
        gc.collect()
        if self._torch is not None and self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

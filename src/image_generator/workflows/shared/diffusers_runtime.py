"""Lazy Diffusers runtime and instance-owned pipeline lifecycle."""

import gc
import re
from copy import deepcopy
from importlib import import_module
from pathlib import Path

from ..base import Workflow
from .adapters import DiffusersAdapters
from .metadata import scheduler_config_snapshot
from .sampling import build_scheduler


class TransformerWorkflow(DiffusersAdapters, Workflow):
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
        options = {
            "dtype": getattr(self._torch, self.dtype),
            "local_files_only": self.local_files_only,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
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
        self._pipeline.scheduler = build_scheduler(
            cls, self._scheduler_config, requirements=request.model.defaults["requirements"]
        )

    def _synchronize(self):
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize(self.device)

    def _runtime_metadata(self, seconds):
        return {
            "device": self.device,
            "dtype": self.dtype,
            "torch_version": self._torch.__version__,
            "diffusers_version": self._diffusers.__version__,
            "generation_seconds": seconds,
        }

    def _scheduler_metadata(self):
        return {
            "class_name": type(self._pipeline.scheduler).__name__,
            "config": scheduler_config_snapshot(dict(self._pipeline.scheduler.config)),
        }

    def close(self):
        self._pipeline = self._model_key = self._scheduler_class = self._scheduler_config = None
        self._has_adapters = False
        gc.collect()
        if self._torch is not None and self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

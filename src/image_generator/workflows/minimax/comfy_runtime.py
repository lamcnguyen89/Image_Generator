"""Native ComfyUI loading and execution lifecycle for MiniMax."""

import subprocess
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import MethodType, SimpleNamespace

from ..shared.assets import file_fingerprint


def protect_quantized_transfers(root, torch):
    """Keep Comfy's quantized Parameter transfers outside inference mode.

    Guard loaded instances only, as in the local notebook; do not patch libraries.
    Rewrapping quantized tensors under inference mode can lose version counters.
    """

    def guard(original):
        def apply(module, fn, recurse=True):
            with torch.inference_mode(False), torch.no_grad():
                return original(fn, recurse=recurse)

        return apply

    for module in root.modules():
        if getattr(module, "_minimax_transfer_guard", False):
            continue
        if any(hasattr(parameter, "__tensor_flatten__") for parameter in module.parameters(recurse=False)):
            module._apply = MethodType(guard(module._apply), module)
            module._minimax_transfer_guard = True


class ComfyRuntimeMixin:
    def _load_runtime(self):
        if self._runtime is not None:
            return
        if not (self.comfyui_dir / "comfy_extras/nodes_minimax_h3.py").is_file():
            raise RuntimeError(
                f"Install ComfyUI with native H3 support at {self.comfyui_dir}; see Documents/MiniMax_Workflow.md"
            )
        self._torch = import_module("torch")
        if not self._torch.cuda.is_available():
            raise RuntimeError("MiniMax generation requires CUDA; browsing, previews, and dry runs do not")
        if str(self.comfyui_dir) not in sys.path:
            sys.path.insert(0, str(self.comfyui_dir))
        options = import_module("comfy.options")
        options.enable_args_parsing(False)
        runtime = SimpleNamespace(
            sd=import_module("comfy.sd"),
            utils=import_module("comfy.utils"),
            samplers=import_module("comfy.samplers"),
            management=import_module("comfy.model_management"),
            h3=import_module("comfy_extras.nodes_minimax_h3"),
            sampling=import_module("comfy_extras.nodes_custom_sampler"),
            nodes=import_module("nodes"),
        )
        if str(runtime.management.get_torch_device()) != self.device:
            raise RuntimeError("Use a dedicated ComfyUI process configured for cuda:0 with this MiniMax workflow")
        try:
            self._comfy_revision = subprocess.check_output(
                ["git", "-C", str(self.comfyui_dir), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            self._comfy_revision = "unknown"
        self._runtime = runtime

    def _resolve_text_encoder(self, source):
        if isinstance(source, Path):
            return source
        from huggingface_hub import hf_hub_download

        # Keep the snapshot filename: resolving HF symlinks produces extensionless blobs,
        # which ComfyUI misidentifies as pickle checkpoints instead of safetensors.
        return Path(
            hf_hub_download(
                repo_id=source["repo_id"],
                filename=source["filename"],
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                local_files_only=self.local_files_only,
                token=self._token,
            )
        )

    def _ensure_pipeline(self, checkpoint, vae, encoder):
        key = tuple(file_fingerprint(path) for path in (checkpoint, vae, encoder))
        if self._pipeline is not None and key == self._model_key:
            self.report_progress("reusing_model")
            return
        self.report_progress("switching_model" if self._pipeline is not None else "loading_model")
        self.close()
        self._owns_models = True
        runtime = self._runtime
        components = {}
        try:
            components["base"] = runtime.sd.load_diffusion_model(str(checkpoint))
            protect_quantized_transfers(components["base"].model, self._torch)
            self.report_progress("loading_text_encoder")
            components["clip"] = runtime.sd.load_clip([str(encoder)], clip_type=runtime.sd.CLIPType.MINIMAX)
            protect_quantized_transfers(components["clip"].cond_stage_model, self._torch)
            self.report_progress("loading_vae")
            state, metadata = runtime.utils.load_torch_file(str(vae), return_metadata=True)
            try:
                components["vae"] = runtime.sd.VAE(sd=state, metadata=metadata)
                protect_quantized_transfers(components["vae"].first_stage_model, self._torch)
            finally:
                del state
        except Exception:
            components.clear()
            raise
        self._pipeline, self._model_key = components, key

    @contextmanager
    def _sampling_progress(self, steps):
        """Bridge Comfy's existing sampler callback only while this worker samples.

        ProgressBar captures the global hook on construction. Do not replace the
        sampler callback, or pass its tensor/preview payload to job persistence.
        The worker serializes workflow execution in this process.
        """
        utils = self._runtime.utils
        previous = utils.PROGRESS_BAR_HOOK

        def report(current, total, preview=None, node_id=None):
            if type(current) is int and type(total) is int and 0 <= current <= total and total > 0:
                self.report_progress("denoising", current, total)

        try:
            utils.set_progress_bar_global_hook(report)
            self.report_progress("denoising", 0, steps)
            yield
        finally:
            utils.set_progress_bar_global_hook(previous)

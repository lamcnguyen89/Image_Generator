"""Instance-owned generation implementation."""

import gc
import math
from pathlib import Path
from time import perf_counter

from ..base import GeneratedBatch, Workflow
from ..shared.assets import file_fingerprint
from ..shared.seeds import image_seeds
from ..shared.schedules import validate_schedule
from ..shared.samplers import create_scheduler, get_sampler
from ..shared.validation import check_video_encoder
from .comfy_runtime import ComfyRuntimeMixin
from .conditioning import prepare_input
from .parameters import DEFAULTS, MODES
from .validation import component_sources

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class MiniMaxWorkflow(ComfyRuntimeMixin, Workflow):
    workflow_id = "minimax"
    defaults = DEFAULTS
    ui_modes = MODES
    component_sources = staticmethod(component_sources)

    def _prepare_input(self, request):
        return prepare_input(request, self.report_progress)

    def __init__(
        self, *, device="cuda:0", dtype=None, cache_dir=None, local_files_only=False, token=None, comfyui_dir=None
    ):
        if device not in {"cuda", "cuda:0"}:
            raise ValueError("MiniMax's native ComfyUI runtime currently requires device cuda:0")
        if dtype is not None:
            raise ValueError("MiniMax precision is managed by its checkpoint and ComfyUI; omit dtype")
        self.device = "cuda:0"
        self.comfyui_dir = Path(comfyui_dir or PROJECT_ROOT / "build/ComfyUI").expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self.local_files_only, self._token = local_files_only, token
        self._torch = self._runtime = self._pipeline = self._model = None
        self._model_key = self._lora_key = None
        self._owns_models = False
        self._comfy_revision = "unknown"

    def _configure_loras(self, request):
        active = tuple(item for item in request.loras if item.strength)
        key = tuple((file_fingerprint(item.asset.asset_paths["weights"]), item.strength) for item in active)
        if self._model is not None and key == self._lora_key:
            self.report_progress("reusing_loras", len(active), len(active))
            return
        self.report_progress("resetting_loras")
        self._model = self._lora_key = None
        model = self._pipeline["base"]
        self.report_progress("loading_loras", 0, len(active))
        for index, item in enumerate(active):
            weights, metadata = self._runtime.utils.load_torch_file(
                str(item.asset.asset_paths["weights"]), safe_load=True, return_metadata=True
            )
            before = sum(len(patches) for patches in model.patches.values())
            try:
                model, _ = self._runtime.sd.load_lora_for_models(
                    model, None, weights, float(item.strength), 0.0, lora_metadata=metadata
                )
            finally:
                del weights
            if sum(len(patches) for patches in model.patches.values()) <= before:
                raise ValueError(f"LoRA {item.asset.id!r} did not match any MiniMax H3 weights")
            self.report_progress("loading_loras", index + 1, len(active))
        self._model, self._lora_key = model, key
        self.report_progress("loras_ready", len(active), len(active))

    def _render_one(self, request, positive, latent, seed):
        nodes, settings = self._runtime.sampling, request.settings
        schedule = validate_schedule("comfyui_minimax_h3", settings["sampling"]["noise_schedule"],
                                     request.model.defaults["requirements"])
        with self._torch.inference_mode(False), self._torch.no_grad():
            guider = nodes.BasicGuider.execute(model=self._model, conditioning=positive)[0]
            sampler = create_scheduler("minimax", settings["sampling"], request.model.defaults["requirements"], nodes=nodes)
            sigmas = nodes.BasicScheduler.execute(
                model=self._model,
                scheduler=schedule.native_name,
                steps=settings["num_inference_steps"],
                denoise=1.0,
            )[0]
            sigma_values = sigmas.detach().cpu().tolist() if hasattr(sigmas, "detach") else sigmas.tolist()
            if (len(sigma_values) < 2 or sigma_values[-1] != 0
                    or any(not math.isfinite(value) or not 0 < value <= 1 for value in sigma_values[:-1])
                    or any(source <= target for source, target in zip(sigma_values, sigma_values[1:]))):
                raise ValueError("MiniMax requires a finite descending flow sigma grid ending at zero")
            scheduler_metadata = {
                "backend": "comfyui", "native_sampler": get_sampler("minimax", settings["sampling"]["sampler"]).native_name,
                "native_schedule": schedule.native_name, "sigmas": sigma_values,
                "num_inference_steps": len(sigma_values) - 1,
            }
            noise = nodes.RandomNoise.execute(noise_seed=seed)[0]
            self._torch.cuda.synchronize()
            start = perf_counter()
            with self._sampling_progress(len(sigma_values) - 1):
                sampled = nodes.SamplerCustomAdvanced.execute(
                    noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent
                )[0]
            self.report_progress("decoding_video")
            if settings["enable_vae_tiling"]:
                frames = self._runtime.nodes.VAEDecodeTiled().decode(
                    vae=self._pipeline["vae"],
                    samples=sampled,
                    tile_size=512,
                    overlap=64,
                    temporal_size=64,
                    temporal_overlap=8,
                )[0]
            else:
                frames = self._runtime.nodes.VAEDecode().decode(vae=self._pipeline["vae"], samples=sampled)[0]
            self._torch.cuda.synchronize()
            return frames, perf_counter() - start, scheduler_metadata

    def generate(self, request):
        if request.settings["num_videos_per_prompt"] != 1:
            raise ValueError("Use generate_batches() or run_generation() to stream multiple MiniMax videos")
        return next(self.generate_batches(request))

    def generate_batches(self, request):
        self.validate(request)
        try:
            settings = request.settings
            width, height, conditioning, input_metadata = self._prepare_input(request)
            check_video_encoder()
            self.report_progress("loading_runtime")
            self._load_runtime()
            sources = self.component_sources(request.model)
            self.report_progress("resolving_text_encoder")
            encoder = self._resolve_text_encoder(sources["text_encoder"])
            checkpoint, vae = request.model.asset_paths["weights"], sources["vae"]
            with self._torch.inference_mode(False), self._torch.no_grad():
                self._ensure_pipeline(checkpoint, vae, encoder)
                self._configure_loras(request)
                schedule = validate_schedule("comfyui_minimax_h3", settings["sampling"]["noise_schedule"],
                                             request.model.defaults["requirements"])
                if (
                    get_sampler("minimax", settings["sampling"]["sampler"]).native_name not in self._runtime.samplers.SAMPLER_NAMES
                    or schedule.native_name not in self._runtime.samplers.SCHEDULER_NAMES
                ):
                    raise ValueError("The selected sampling is unavailable in this ComfyUI checkout")
                inputs = {}
                if conditioning is not None:
                    import numpy as np

                    inputs["first_frame"] = self._torch.from_numpy(
                        np.array(conditioning).astype(np.float32) / 255.0
                    ).unsqueeze(0)
                self.report_progress("encoding_prompt")
                positive, latent = self._runtime.h3.MiniMaxH3ImageToVideo.execute(
                    clip=self._pipeline["clip"],
                    vae=self._pipeline["vae"],
                    prompt=request.prompt,
                    width=width,
                    height=height,
                    length=settings["num_frames"],
                    **inputs,
                ).result
            seeds = image_seeds(settings, settings["num_videos_per_prompt"])
            for index, seed in enumerate(seeds):
                self.report_progress("video_variation", index + 1, len(seeds))
                frames, elapsed, scheduler_metadata = self._render_one(request, positive, latent, seed)
                if len(frames) != settings["num_frames"]:
                    raise RuntimeError("The MiniMax decoder returned an unexpected frame count")
                from image_generator.generation import request_snapshot

                batch = GeneratedBatch(
                    (),
                    {
                        **request_snapshot(request),
                        "seeds": [seed],
                        "video_index": index,
                        "scheduler": scheduler_metadata,
                        "component_paths": {
                            "checkpoint": str(checkpoint),
                            "vae": str(vae),
                            "text_encoder": str(encoder),
                        },
                        "asset_fingerprints": self._model_key,
                        **input_metadata,
                        "frame_count": len(frames),
                        "duration_seconds": len(frames) / settings["fps"],
                        "audio_exported": False,
                        "runtime": {
                            "backend": "comfyui",
                            "device": self.device,
                            "torch_version": self._torch.__version__,
                            "generation_seconds": elapsed,
                            "quantized_transfer_guard": True,
                            "comfyui_revision": self._comfy_revision,
                            "precision": "checkpoint_managed",
                        },
                    },
                    videos=(frames,),
                    conditioning_image=conditioning,
                )
                yield batch
                del batch, frames
        except Exception:
            self.close()
            raise

    def close(self):
        self._model = self._pipeline = None
        self._model_key = self._lora_key = None
        if self._runtime is not None and self._owns_models:
            self._owns_models = False
            with self._torch.inference_mode(False), self._torch.no_grad():
                self._runtime.management.unload_all_models()
                gc.collect()
                self._torch.cuda.empty_cache()

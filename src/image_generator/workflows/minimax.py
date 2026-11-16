"""MiniMax H3 text/image-to-video using native nodes without a ComfyUI server."""

from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
from importlib import import_module
from io import BytesIO
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
from time import perf_counter
from types import MethodType, SimpleNamespace

from .base import GeneratedBatch, Workflow
from .common import _nonnegative_number, check_video_encoder, file_fingerprint, image_seeds, validate_weights
from .ui import FormValidationError, UIField, common_fields, definition, frame_constraints


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENCODERS = {f"text_encoders/qwen3vl_32b_minimax_h3_{kind}.safetensors"
            for kind in ("bf16", "int8_convrot", "nvfp4_awq")}


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


class MiniMaxWorkflow(Workflow):
    ui_modes = ("img2vid", "txt2vid")
    defaults = {
        "prompt": "", "negative_prompt": "", "positive_prefix": "", "positive_suffix": "",
        "trigger_words": [], "width": 832, "height": 480, "preserve_aspect_ratio": True,
        "num_frames": 124, "fps": 24, "num_videos_per_prompt": 1, "num_inference_steps": 20,
        "guidance_scale": 1.0, "seed": None, "enable_vae_tiling": True,
        "sampling": {"sampler": "res_multistep", "noise_schedule": "simple"}, "loras": [],
    }

    def __init__(self, *, device="cuda:0", dtype=None, cache_dir=None,
                 local_files_only=False, token=None, comfyui_dir=None):
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

    def get_ui_definition(self, model, mode):
        requirements = model.defaults["requirements"]
        if requirements.get("runtime") != "comfyui" or requirements.get("guider", "BasicGuider") != "BasicGuider":
            raise ValueError("MiniMax H3 requires runtime comfyui and BasicGuider")
        if requirements.get("native_fps", 24) != 24:
            raise ValueError("MiniMax H3 requires native_fps 24")
        minimum, step = frame_constraints(requirements, native_step=17, native_offset=5)
        if minimum > 3600:
            raise ValueError("MiniMax frame_count requirements leave no supported frame count")
        allowed = requirements.get("allowed_sampling")
        if allowed is not None and (not isinstance(allowed, list) or self.defaults["sampling"] not in allowed):
            raise ValueError("This MiniMax workflow requires res_multistep with the simple schedule")
        fields = []
        for field in common_fields(model, dimension_base=32, video=True):
            if field.name == "guidance_scale":
                field = replace(field, label="Guidance scale", minimum=1, maximum=1, read_only=True,
                                help="H3 BasicGuider uses CFG 1; there is no second expert.")
            elif field.name == "negative_prompt":
                field = replace(field, read_only=True, help="BasicGuider does not use a negative prompt; leave empty.")
            fields.append(field)
        if mode == "img2vid":
            fields.insert(0, UIField(
                "input_image", "Source image", "file", "Source image", scope="inputs", required=True,
                help="Choose a single-frame image. Generation saves the original source for this job; previews do not save uploads."))
            fields.append(UIField(
                "preserve_aspect_ratio", "Preserve source aspect ratio", "boolean", "Output",
                help="Width and height set the output pixel budget; the source determines aspect ratio."))
        return definition(model, mode, workflow="minimax", supported_modes=self.ui_modes,
                          variant="minimax_h3", partial="controls/minimax.html", fields=(
            *fields,
            UIField("num_frames", "Frame count", "integer", "Output", minimum=minimum, maximum=3600, step=step,
                    help="H3 uses 17*n + 5 frames, independently of WAN's frame rule."),
            UIField("fps", "Playback FPS", "number", "Output", minimum=24, maximum=24, read_only=True,
                    help="MiniMax H3's native rate is 24 FPS."),
            UIField("sampling.sampler", "Sampler", "text", "Sampling", read_only=True,
                    help="Fixed to res_multistep for H3."),
            UIField("sampling.noise_schedule", "Noise schedule", "text", "Sampling", read_only=True,
                    help="Fixed to simple for H3."),
            UIField("enable_vae_tiling", "Enable VAE tiling", "boolean", "Advanced"),
        ))

    def validate_form_settings(self, model, mode, settings):
        super().validate_form_settings(model, mode, settings)
        if settings["sampling"] != self.defaults["sampling"]:
            raise FormValidationError({"sampling": "H3 requires res_multistep with the simple schedule."})
        if settings["negative_prompt"]:
            raise FormValidationError({"negative_prompt": "BasicGuider does not support negative prompting."})
        if type(settings["preserve_aspect_ratio"]) is not bool:
            raise FormValidationError({"preserve_aspect_ratio": "Enter a boolean value."})

    @staticmethod
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
            elif name == "text_encoder" and source.get("repo_id") == "Comfy-Org/MiniMax-H3" and source.get("filename") in ENCODERS:
                local = model.manifest_path.parents[2] / "text_encoders" / Path(source["filename"]).name
                result[name] = local if local.is_file() else dict(source)
            else:
                raise ValueError(f"components.{name} needs local weights or a documented MiniMax H3 encoder reference")
        return result

    def validate(self, request):
        unknown = request.settings.keys() - self.defaults.keys()
        if unknown:
            raise ValueError(f"Unsupported MiniMax settings: {', '.join(sorted(unknown))}")
        self.validate_form_settings(request.model, request.mode, request.settings)
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
        sources = self.component_sources(request.model)
        if not sources["vae"].is_file():
            raise ValueError(f"MiniMax VAE weights not found: {sources['vae']}")
        vae_header = safetensors_header(sources["vae"])
        if "latents_mean" not in vae_header or not any(key.startswith(("decoder.transformer_blocks.", "decoder.blocks.")) for key in vae_header):
            raise ValueError("Select the MiniMax H3 video VAE, not a WAN or audio VAE")
        if isinstance(sources["text_encoder"], Path):
            validate_weights(sources["text_encoder"], "MiniMax text encoder")
        identifiers, paths, names = set(), set(), set()
        for selection in request.loras:
            asset = selection.asset
            if asset.kind != "lora" or asset.family != "minimax" or asset.defaults["compatibility"].get("model_variant") != "minimax_h3":
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

    def _load_runtime(self):
        if self._runtime is not None:
            return
        if not (self.comfyui_dir / "comfy_extras/nodes_minimax_h3.py").is_file():
            raise RuntimeError(f"Install ComfyUI with native H3 support at {self.comfyui_dir}; see Documents/MiniMax_Workflow.md")
        self._torch = import_module("torch")
        if not self._torch.cuda.is_available():
            raise RuntimeError("MiniMax generation requires CUDA; browsing, previews, and dry runs do not")
        if str(self.comfyui_dir) not in sys.path:
            sys.path.insert(0, str(self.comfyui_dir))
        options = import_module("comfy.options")
        options.enable_args_parsing(False)
        runtime = SimpleNamespace(
            sd=import_module("comfy.sd"), utils=import_module("comfy.utils"),
            samplers=import_module("comfy.samplers"), management=import_module("comfy.model_management"),
            h3=import_module("comfy_extras.nodes_minimax_h3"),
            sampling=import_module("comfy_extras.nodes_custom_sampler"), nodes=import_module("nodes"),
        )
        if str(runtime.management.get_torch_device()) != self.device:
            raise RuntimeError("Use a dedicated ComfyUI process configured for cuda:0 with this MiniMax workflow")
        try:
            self._comfy_revision = subprocess.check_output(
                ["git", "-C", str(self.comfyui_dir), "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            self._comfy_revision = "unknown"
        self._runtime = runtime

    def _resolve_text_encoder(self, source):
        if isinstance(source, Path):
            return source
        from huggingface_hub import hf_hub_download

        # Keep the snapshot filename: resolving HF symlinks produces extensionless blobs,
        # which ComfyUI misidentifies as pickle checkpoints instead of safetensors.
        return Path(hf_hub_download(repo_id=source["repo_id"], filename=source["filename"],
                                   cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                                   local_files_only=self.local_files_only, token=self._token))

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
                str(item.asset.asset_paths["weights"]), safe_load=True, return_metadata=True)
            before = sum(len(patches) for patches in model.patches.values())
            try:
                model, _ = self._runtime.sd.load_lora_for_models(
                    model, None, weights, float(item.strength), 0.0, lora_metadata=metadata)
            finally:
                del weights
            if sum(len(patches) for patches in model.patches.values()) <= before:
                raise ValueError(f"LoRA {item.asset.id!r} did not match any MiniMax H3 weights")
            self.report_progress("loading_loras", index + 1, len(active))
        self._model, self._lora_key = model, key
        self.report_progress("loras_ready", len(active), len(active))

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

    def _render_one(self, request, positive, latent, seed):
        nodes, settings = self._runtime.sampling, request.settings
        with self._torch.inference_mode(False), self._torch.no_grad():
            guider = nodes.BasicGuider.execute(model=self._model, conditioning=positive)[0]
            sampler = nodes.KSamplerSelect.execute(sampler_name=settings["sampling"]["sampler"])[0]
            sigmas = nodes.BasicScheduler.execute(model=self._model, scheduler=settings["sampling"]["noise_schedule"],
                                                   steps=settings["num_inference_steps"], denoise=1.0)[0]
            noise = nodes.RandomNoise.execute(noise_seed=seed)[0]
            self._torch.cuda.synchronize()
            start = perf_counter()
            with self._sampling_progress(settings["num_inference_steps"]):
                sampled = nodes.SamplerCustomAdvanced.execute(noise=noise, guider=guider, sampler=sampler,
                                                              sigmas=sigmas, latent_image=latent)[0]
            self.report_progress("decoding_video")
            if settings["enable_vae_tiling"]:
                frames = self._runtime.nodes.VAEDecodeTiled().decode(
                    vae=self._pipeline["vae"], samples=sampled, tile_size=512,
                    overlap=64, temporal_size=64, temporal_overlap=8)[0]
            else:
                frames = self._runtime.nodes.VAEDecode().decode(vae=self._pipeline["vae"], samples=sampled)[0]
            self._torch.cuda.synchronize()
            return frames, perf_counter() - start

    def generate(self, request):
        if request.settings["num_videos_per_prompt"] != 1:
            raise ValueError("Use generate_batches() or run_generation() to stream multiple MiniMax videos")
        return next(self.generate_batches(request))

    def _prepare_input(self, request):
        """Resolve the canvas and optional image without loading GPU components."""
        settings = request.settings
        width, height = settings["width"], settings["height"]
        if request.mode == "txt2vid":
            return width, height, None, {}

        from PIL import Image, ImageOps

        self.report_progress("conditioning_image")
        path = request.inputs["input_image"]
        payload = path.read_bytes()
        with Image.open(BytesIO(payload)) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        input_hash = hashlib.sha256(payload).hexdigest()
        if settings["preserve_aspect_ratio"]:
            ratio = source.width / source.height
            multiple = math.lcm(32, request.model.defaults["requirements"].get("dimension_multiple", 32))
            width = max(multiple, int(math.sqrt(width * height * ratio)) // multiple * multiple)
            height = max(multiple, int(math.sqrt(settings["width"] * settings["height"] / ratio)) // multiple * multiple)
            conditioning = source.resize((width, height), Image.Resampling.LANCZOS)
        else:
            conditioning = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
        return width, height, conditioning, {
            "input_image": {"path": str(path), "sha256": input_hash,
                            "source_size": list(source.size), "conditioning_size": list(conditioning.size)},
        }

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
                if settings["sampling"]["sampler"] not in self._runtime.samplers.SAMPLER_NAMES or settings["sampling"]["noise_schedule"] not in self._runtime.samplers.SCHEDULER_NAMES:
                    raise ValueError("The selected sampling is unavailable in this ComfyUI checkout")
                inputs = {}
                if conditioning is not None:
                    import numpy as np

                    inputs["first_frame"] = self._torch.from_numpy(
                        np.array(conditioning).astype(np.float32) / 255.0).unsqueeze(0)
                self.report_progress("encoding_prompt")
                positive, latent = self._runtime.h3.MiniMaxH3ImageToVideo.execute(
                    clip=self._pipeline["clip"], vae=self._pipeline["vae"], prompt=request.prompt,
                    width=width, height=height, length=settings["num_frames"], **inputs).result
            seeds = image_seeds(settings, settings["num_videos_per_prompt"])
            for index, seed in enumerate(seeds):
                self.report_progress("video_variation", index + 1, len(seeds))
                frames, elapsed = self._render_one(request, positive, latent, seed)
                if len(frames) != settings["num_frames"]:
                    raise RuntimeError("The MiniMax decoder returned an unexpected frame count")
                from image_generator.generation import request_snapshot

                batch = GeneratedBatch((), {
                    **request_snapshot(request), "seeds": [seed], "video_index": index,
                    "component_paths": {"checkpoint": str(checkpoint), "vae": str(vae), "text_encoder": str(encoder)},
                    "asset_fingerprints": self._model_key,
                    **input_metadata,
                    "frame_count": len(frames), "duration_seconds": len(frames) / settings["fps"],
                    "audio_exported": False,
                    "runtime": {"backend": "comfyui", "device": self.device, "torch_version": self._torch.__version__,
                                "generation_seconds": elapsed, "quantized_transfer_guard": True,
                                "comfyui_revision": self._comfy_revision, "precision": "checkpoint_managed"},
                }, videos=(frames,), conditioning_image=conditioning)
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

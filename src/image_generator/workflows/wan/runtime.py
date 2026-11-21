"""Instance-owned generation implementation."""

import hashlib
import math
from io import BytesIO
from time import perf_counter

from ..base import GeneratedBatch
from ..shared.assets import file_fingerprint
from ..shared.diffusers_runtime import TransformerWorkflow
from ..shared.seeds import image_seeds
from ..shared.validation import _nonnegative_number, check_video_encoder
from .adapters import configure_expert_loras
from .conditioning import prepare_conditioning
from .parameters import DEFAULTS, MODES, VARIANTS
from .validation import variant_definition


class WanWorkflow(TransformerWorkflow):
    workflow_id = "wan"
    defaults = DEFAULTS
    ui_modes = MODES
    variants = VARIANTS
    variant_definition = staticmethod(variant_definition)
    _prepare_conditioning = staticmethod(prepare_conditioning)

    def _build_pipeline(self, request):
        path = request.model.asset_paths["weights"]
        source = str(path) if path else request.model.defaults["base_model_id"]
        options = self._load_options()
        if path is not None:
            options["local_files_only"] = True
        vae = self._diffusers.AutoencoderKLWan.from_pretrained(
            source,
            subfolder="vae",
            **{**options, "dtype": self._torch.float32},
        )
        try:
            _, pipeline_class = self.variant_definition(request.model)
            return getattr(self._diffusers, pipeline_class).from_pretrained(source, vae=vae, **options)
        finally:
            del vae

    @staticmethod
    def _pipeline_fingerprint(path):
        if path is None:
            return None
        return (
            str(path),
            tuple(
                file_fingerprint(item)
                for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix in {".json", ".safetensors"}
            ),
        )

    def _canvas_multiples(self, request):
        pipe = self._pipeline
        if pipe.transformer is None or pipe.transformer_2 is None:
            raise ValueError("WAN 2.2 requires both transformer and transformer_2 experts")
        boundary = pipe.config.boundary_ratio
        if not _nonnegative_number(boundary) or not 0 < boundary < 1:
            raise ValueError("WAN 2.2 requires a valid expert boundary_ratio between 0 and 1")
        for name in request.model.defaults["requirements"].get("required_components", []):
            if getattr(pipe, name, None) is None:
                raise ValueError(f"The WAN pipeline is missing required component {name!r}")
        if request.mode == "txt2vid":
            for name in ("transformer", "transformer_2"):
                config = getattr(pipe, name).config
                if config.in_channels != 16 or config.out_channels != 16:
                    raise ValueError("WAN T2V A14B requires experts with 16 input/output channels")
        temporal = pipe.vae_scale_factor_temporal
        spatial = pipe.vae_scale_factor_spatial
        patch = pipe.transformer.config.patch_size
        if type(spatial) is not int or spatial <= 0:
            raise ValueError("The loaded VAE requires a positive spatial scale factor")
        if type(temporal) is not int or temporal <= 0 or (request.settings["num_frames"] - 1) % temporal:
            raise ValueError("num_frames does not match the loaded VAE temporal alignment")
        if len(patch) != 3 or any(type(value) is not int or value <= 0 for value in patch):
            raise ValueError("The WAN transformer has an invalid patch size")
        if ((request.settings["num_frames"] - 1) // temporal + 1) % patch[0]:
            raise ValueError("num_frames does not match the loaded transformer temporal patch size")
        multiple = request.model.defaults["requirements"].get("dimension_multiple", 16)
        height_multiple = math.lcm(16, multiple, spatial * patch[1])
        width_multiple = math.lcm(16, multiple, spatial * patch[2])
        return height_multiple, width_multiple

    def generate(self, request):
        if request.settings["num_videos_per_prompt"] != 1:
            raise ValueError("Use generate_batches() or run_generation() to stream multiple WAN videos")
        return next(self.generate_batches(request))

    def generate_batches(self, request):
        self.validate(request)
        try:
            source = conditioning = None
            input_metadata = {}
            if request.mode == "img2vid":
                from PIL import Image, ImageOps

                payload = request.inputs["input_image"].read_bytes()
                with Image.open(BytesIO(payload)) as image:
                    source = ImageOps.exif_transpose(image).convert("RGB")
                input_hash = hashlib.sha256(payload).hexdigest()
                del payload
            check_video_encoder()
            self._load_runtime()
            self._ensure_pipeline(
                request,
                (
                    request.model.defaults["model_variant"],
                    self._pipeline_fingerprint(request.model.asset_paths["weights"]),
                ),
            )
            multiples = self._canvas_multiples(request)
            height, width = request.settings["height"], request.settings["width"]
            inputs = {}
            if source is not None:
                self.report_progress("conditioning_image")
                conditioning = self._prepare_conditioning(source, request, multiples)
                width, height = conditioning.size
                inputs["image"] = conditioning
                input_metadata["input_image"] = {
                    "path": str(request.inputs["input_image"]),
                    "sha256": input_hash,
                    "source_size": list(source.size),
                    "conditioning_size": list(conditioning.size),
                }
            elif height % multiples[0] or width % multiples[1]:
                raise ValueError(f"Loaded WAN model requires height/width multiples of {multiples[0]}/{multiples[1]}")
            configure_expert_loras(self, request)
            tiling = (
                self._pipeline.vae.enable_tiling
                if request.settings["enable_vae_tiling"]
                else self._pipeline.vae.disable_tiling
            )
            tiling()
            arguments = {
                name: request.settings[name]
                for name in ("num_frames", "num_inference_steps", "guidance_scale", "guidance_scale_2")
            }
            arguments.update(height=height, width=width, num_videos_per_prompt=1, output_type="np")
            seeds = image_seeds(request.settings, request.settings["num_videos_per_prompt"])
            for index, seed in enumerate(seeds):
                self.report_progress("video_variation", index + 1, len(seeds))
                self._configure_scheduler(request)
                generator = self._torch.Generator(device=self.device).manual_seed(seed)
                self._synchronize()
                start = perf_counter()
                steps = request.settings["num_inference_steps"]
                self.report_progress("denoising", 0, steps)

                def on_step_end(pipe, step, timestep, tensors):
                    total = getattr(pipe, "num_timesteps", steps)
                    self.report_progress("denoising", step + 1, total)
                    if step + 1 == total:
                        self.report_progress("decoding_video")
                    return tensors

                with self._torch.inference_mode():
                    frames = self._pipeline(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        generator=generator,
                        callback_on_step_end=on_step_end,
                        **arguments,
                        **inputs,
                    ).frames[0]
                self._synchronize()
                elapsed = perf_counter() - start
                if len(frames) != request.settings["num_frames"]:
                    raise RuntimeError("The WAN pipeline returned an unexpected frame count")
                from image_generator.generation import request_snapshot

                batch = GeneratedBatch(
                    (),
                    {
                        **request_snapshot(request),
                        "seeds": [seed],
                        "video_index": index,
                        **input_metadata,
                        "generation_settings": arguments.copy(),
                        "scheduler": self._scheduler_metadata(),
                        "boundary_ratio": self._pipeline.config.boundary_ratio,
                        "frame_count": len(frames),
                        "duration_seconds": len(frames) / request.settings["fps"],
                        "audio_exported": False,
                        "runtime": {**self._runtime_metadata(elapsed), "vae_dtype": str(self._pipeline.vae.dtype)},
                    },
                    videos=(frames,),
                    conditioning_image=conditioning,
                )
                yield batch
                del batch, frames
        except Exception:
            self.close()
            raise

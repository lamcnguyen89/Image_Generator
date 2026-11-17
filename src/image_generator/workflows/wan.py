"""WAN 2.2 I2V/T2V A14B with targeted experts and streamed variations."""

import hashlib
from io import BytesIO
import json
import math
from time import perf_counter

from image_generator.catalog.compatibility import lora_model_variants
from .base import GeneratedBatch
from .common import TransformerWorkflow, _nonnegative_number, check_video_encoder, file_fingerprint, image_seeds, validate_common


class WanWorkflow(TransformerWorkflow):
    ui_modes = ("img2vid", "txt2vid")
    variants = {
        "wan_2_2_i2v_a14b": ("img2vid", "WanImageToVideoPipeline"),
        "wan_2_2_t2v_a14b": ("txt2vid", "WanPipeline"),
    }
    defaults = {
        "prompt": "", "negative_prompt": "", "positive_prefix": "", "positive_suffix": "",
        "trigger_words": [], "width": 832, "height": 480, "preserve_aspect_ratio": True,
        "num_frames": 161, "fps": 16, "num_videos_per_prompt": 1, "num_inference_steps": 40,
        "guidance_scale": 3.5, "guidance_scale_2": 3.5, "seed": None, "enable_vae_tiling": True,
        "sampling": {"sampler": "model_default", "noise_schedule": "default"}, "loras": [],
    }

    @classmethod
    def variant_definition(cls, model):
        variant = model.defaults.get("model_variant")
        if variant not in cls.variants:
            raise ValueError("WAN requires model_variant wan_2_2_i2v_a14b or wan_2_2_t2v_a14b")
        return cls.variants[variant]

    def validate(self, request):
        mode, pipeline_class = self.variant_definition(request.model)
        validate_common(request, self.defaults, family="wan", mode=mode,
                        count_field="num_videos_per_prompt", samplers={"model_default"})
        model, settings = request.model, request.settings
        if mode == "img2vid":
            if set(request.inputs) != {"input_image"} or not request.inputs["input_image"].is_file():
                raise ValueError("WAN img2vid requires a local input_image file")
        elif request.inputs:
            raise ValueError("WAN txt2vid does not accept image inputs")
        path = model.asset_paths["weights"]
        if path is not None:
            if model.defaults.get("weights_component") != "pipeline" or not path.is_dir() or not (path / "model_index.json").is_file():
                raise ValueError("Local WAN weights must be a complete Diffusers pipeline directory with model_index.json")
            index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
            if not isinstance(index, dict):
                raise ValueError("Local WAN model_index.json must be an object")
            declared = index.get("_class_name")
            if declared != pipeline_class and (declared is not None or mode == "txt2vid"):
                raise ValueError(f"This WAN variant requires a {pipeline_class} pipeline directory")
        elif not model.defaults.get("base_model_id"):
            raise ValueError("WAN requires base_model_id or a local pipeline directory")
        guidance = settings["guidance_scale_2"]
        if not _nonnegative_number(guidance):
            raise ValueError("guidance_scale_2 must be a finite number >= 0")
        if (settings["guidance_scale"] > 1) != (guidance > 1):
            raise ValueError("Both WAN guidance scales must be > 1 for CFG, or both <= 1")
        if not _nonnegative_number(settings["fps"]) or settings["fps"] == 0:
            raise ValueError("fps must be a finite number > 0")
        for name in ("preserve_aspect_ratio", "enable_vae_tiling"):
            if type(settings[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
        rule = model.defaults["requirements"].get("frame_count", {"minimum": 5, "step": 4, "offset": 1})
        if not isinstance(rule, dict) or any(type(rule.get(key)) is not int for key in ("minimum", "step", "offset")) or rule["step"] <= 0 or rule["minimum"] < 1:
            raise ValueError("requirements.frame_count needs integer minimum, positive step, and offset")
        count = settings["num_frames"]
        if type(count) is not int or count < max(5, rule["minimum"]) or (count - 1) % 4 or (count - rule["offset"]) % rule["step"]:
            raise ValueError("num_frames must satisfy the checkpoint frame_count rule and be 4*n + 1, at least 5")
        for selection in request.loras:
            compatibility = selection.asset.defaults["compatibility"]
            if model.defaults["model_variant"] not in lora_model_variants(compatibility):
                raise ValueError(f"WAN LoRA {selection.asset.id!r} needs an explicit compatible model_variant")
            if selection.expert_target not in {"high_noise", "low_noise"}:
                raise ValueError(f"WAN LoRA {selection.asset.id!r} needs an explicit high_noise or low_noise expert_target")

    def get_ui_definition(self, model, mode):
        from .ui import UIField, common_fields, definition, frame_constraints, sampling_fields

        minimum, step = frame_constraints(model.defaults["requirements"])
        variant_mode, _ = self.variant_definition(model)
        fields = list(common_fields(model, video=True))
        if mode == "img2vid":
            fields.insert(0, UIField(
                "input_image", "Source image", "file", "Source image", scope="inputs", required=True,
                help="Choose the image to animate. Generation saves the original with the queued job. Select it again after a preview or validation response."))
            fields.append(UIField(
                "preserve_aspect_ratio", "Preserve source aspect ratio", "boolean", "Output",
                help="Width and height set the pixel budget; actual dimensions follow the source image."))
        return definition(model, mode, workflow="wan", supported_modes=(variant_mode,),
                          partial="controls/wan.html", fields=(
            *fields,
            UIField("num_frames", "Frame count", "integer", "Output", minimum=minimum, step=step,
                    help=f"Start at {minimum} and increase by {step} frames."),
            UIField("fps", "Playback FPS", "number", "Output", minimum=0,
                    help="Must be greater than zero. Controls exported video playback."),
            *sampling_fields(model, ("model_default",)),
            UIField("guidance_scale_2", "Low-noise guidance", "number", "Sampling", minimum=0,
                    help="Both guidance values must be above 1 for CFG, or both at most 1."),
            UIField("enable_vae_tiling", "Enable VAE tiling", "boolean", "Advanced"),
        ))

    def validate_form_settings(self, model, mode, settings):
        from .ui import FormValidationError

        super().validate_form_settings(model, mode, settings)
        errors = {}
        if settings["fps"] <= 0:
            errors["fps"] = "Playback FPS must be greater than zero."
        if (settings["guidance_scale"] > 1) != (settings["guidance_scale_2"] > 1):
            errors["guidance_scale_2"] = "Both guidance values must be above 1 for CFG, or both at most 1."
        if type(settings["preserve_aspect_ratio"]) is not bool:
            errors["preserve_aspect_ratio"] = "Choose a boolean value."
        if errors:
            raise FormValidationError(errors)

    def _build_pipeline(self, request):
        path = request.model.asset_paths["weights"]
        source = str(path) if path else request.model.defaults["base_model_id"]
        options = self._load_options()
        if path is not None:
            options["local_files_only"] = True
        vae = self._diffusers.AutoencoderKLWan.from_pretrained(
            source, subfolder="vae", **{**options, "dtype": self._torch.float32},
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
        return (str(path), tuple(file_fingerprint(item) for item in sorted(path.rglob("*"))
                                 if item.is_file() and item.suffix in {".json", ".safetensors"}))

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

    def _prepare_conditioning(self, source, request, multiples):
        from PIL import Image, ImageOps

        height_multiple, width_multiple = multiples
        settings = request.settings
        height, width = settings["height"], settings["width"]
        if settings["preserve_aspect_ratio"]:
            ratio = source.height / source.width
            height = max(height_multiple, int(math.sqrt(height * width * ratio)) // height_multiple * height_multiple)
            width = max(width_multiple, int(math.sqrt(settings["height"] * settings["width"] / ratio)) // width_multiple * width_multiple)
            conditioning = source.resize((width, height), Image.Resampling.LANCZOS)
        else:
            if height % height_multiple or width % width_multiple:
                raise ValueError(f"Loaded WAN model requires height/width multiples of {height_multiple}/{width_multiple}")
            conditioning = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
        return conditioning

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
            self._ensure_pipeline(request, (request.model.defaults["model_variant"],
                                           self._pipeline_fingerprint(request.model.asset_paths["weights"])))
            multiples = self._canvas_multiples(request)
            height, width = request.settings["height"], request.settings["width"]
            inputs = {}
            if source is not None:
                self.report_progress("conditioning_image")
                conditioning = self._prepare_conditioning(source, request, multiples)
                width, height = conditioning.size
                inputs["image"] = conditioning
                input_metadata["input_image"] = {
                    "path": str(request.inputs["input_image"]), "sha256": input_hash,
                    "source_size": list(source.size), "conditioning_size": list(conditioning.size),
                }
            elif height % multiples[0] or width % multiples[1]:
                raise ValueError(f"Loaded WAN model requires height/width multiples of {multiples[0]}/{multiples[1]}")
            self._configure_loras(request, targeted=True)
            tiling = self._pipeline.vae.enable_tiling if request.settings["enable_vae_tiling"] else self._pipeline.vae.disable_tiling
            tiling()
            arguments = {name: request.settings[name] for name in ("num_frames", "num_inference_steps", "guidance_scale", "guidance_scale_2")}
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
                        prompt=request.prompt, negative_prompt=request.negative_prompt,
                        generator=generator, callback_on_step_end=on_step_end, **arguments, **inputs,
                    ).frames[0]
                self._synchronize()
                elapsed = perf_counter() - start
                if len(frames) != request.settings["num_frames"]:
                    raise RuntimeError("The WAN pipeline returned an unexpected frame count")
                from image_generator.generation import request_snapshot

                batch = GeneratedBatch((), {
                    **request_snapshot(request), "seeds": [seed], "video_index": index,
                    **input_metadata,
                    "generation_settings": arguments.copy(), "scheduler": self._scheduler_metadata(),
                    "boundary_ratio": self._pipeline.config.boundary_ratio,
                    "frame_count": len(frames), "duration_seconds": len(frames) / request.settings["fps"],
                    "audio_exported": False,
                    "runtime": {**self._runtime_metadata(elapsed), "vae_dtype": str(self._pipeline.vae.dtype)},
                }, videos=(frames,), conditioning_image=conditioning)
                yield batch
                del batch, frames
        except Exception:
            self.close()
            raise

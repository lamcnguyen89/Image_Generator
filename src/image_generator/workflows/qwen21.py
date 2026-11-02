"""Qwen-Image 2.1 text-to-image with optional local transformer and named LoRAs."""

from contextlib import nullcontext
from dataclasses import replace
import re
from time import perf_counter

from .base import GeneratedBatch
from .common import TransformerWorkflow, _finite_number, file_fingerprint, image_seeds, validate_common, validate_weights
from .ui import FormValidationError


def validate_sigmas(sigmas):
    if sigmas is None:
        return
    if (not isinstance(sigmas, list) or not sigmas
            or any(not _finite_number(value) or not 0 < value <= 1 for value in sigmas)):
        raise ValueError("Sigmas must be a nonempty list of finite numbers in (0, 1]; omit the terminal zero.")
    if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError("Sigmas must be strictly descending.")


def split_fused_lora(state_dict):
    """Convert ComfyUI's gate/up LoRA exactly: shared A and split B output rows."""
    result = dict(state_dict)
    suffix = ".img_mlp.gate_up.lora_A.weight"
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        up_key = key.removesuffix("lora_A.weight") + "lora_B.weight"
        down, up = result.pop(key), result.pop(up_key)
        if (len(down.shape) != 2 or len(up.shape) != 2
                or up.shape[0] % 2 or up.shape[1] != down.shape[0]):
            raise ValueError(f"Invalid fused gate/up LoRA dimensions: {key}")
        gate, projection = up.chunk(2, dim=0)
        for name, weight in (("gate_layer", gate), ("proj", projection)):
            for old, value in ((key, down), (up_key, weight)):
                target = old.replace(".img_mlp.gate_up.", f".img_mlp.{name}.")
                if target in result:
                    raise ValueError(f"LoRA contains both fused and separate weights: {target}")
                result[target] = value
    return result


def load_qwen21_lora(pipeline, path, adapter_name):
    """Load a complete named adapter, normalizing fused Qwen 2.1 projections."""
    # Let Diffusers normalize native/AI Toolkit keys and alpha scaling, then
    # apply the same gate-first split as its Qwen 2.1 checkpoint converter.
    state, metadata = pipeline.lora_state_dict(
        str(path.parent), weight_name=path.name, local_files_only=True,
        use_safetensors=True, return_lora_metadata=True,
    )
    state = split_fused_lora(state)
    if not state:
        raise ValueError("The LoRA contains no adapter weights")
    for key in state:
        match = re.fullmatch(r"transformer\.(.+)\.lora_[AB]\.weight", key)
        if match is None:
            raise ValueError(f"Unsupported Qwen 2.1 LoRA key: {key}")
        # PEFT otherwise logs and ignores unknown targets, silently applying
        # only part of an incompatible adapter. Fail before installing it.
        pipeline.transformer.get_submodule(match[1])
    pipeline.load_lora_into_transformer(
        state, transformer=pipeline.transformer, adapter_name=adapter_name,
        metadata=metadata, _pipeline=pipeline, low_cpu_mem_usage=True,
    )


class Qwen21Workflow(TransformerWorkflow):
    ui_modes = ("txt2img",)
    defaults = {
        "prompt": "", "negative_prompt": "", "positive_prefix": "", "positive_suffix": "",
        "trigger_words": [], "width": 2048, "height": 2048, "num_images_per_prompt": 1,
        "num_inference_steps": 40, "true_cfg_scale": 1.0, "use_kv_cache": True,
        "sigmas": None, "seed": None,
        "sampling": {"sampler": "flow_match_euler", "noise_schedule": "default"}, "loras": [],
    }

    def validate(self, request):
        validate_common(request, self.defaults, family="qwen_2_1", mode="txt2img",
                        count_field="num_images_per_prompt", samplers={"flow_match_euler", "model_default"},
                        guidance_field="true_cfg_scale", dimension_base=32)
        model = request.model
        if request.inputs:
            raise ValueError("Qwen 2.1 txt2img does not accept image inputs")
        if model.defaults.get("model_variant") != "qwen_image_2_1":
            raise ValueError("This Qwen workflow supports only model_variant qwen_image_2_1")
        if not model.defaults.get("base_model_id"):
            raise ValueError("Qwen 2.1 requires base_model_id for its pipeline components")
        path = model.asset_paths["weights"]
        if path is not None:
            if model.defaults.get("weights_component") != "transformer":
                raise ValueError("A local Qwen 2.1 checkpoint must declare weights_component transformer")
            validate_weights(path, "Qwen 2.1 transformer")
        requirements = model.defaults["requirements"]
        if requirements.get("scheduler_class", "FlowMatchEulerDiscreteScheduler") != "FlowMatchEulerDiscreteScheduler":
            raise ValueError("Qwen 2.1 requires FlowMatchEulerDiscreteScheduler")
        self.validate_form_settings(model, request.mode, request.settings)

    def get_ui_definition(self, model, mode):
        from .ui import UIField, common_fields, definition, sampling_fields

        fields = tuple(
            UIField("true_cfg_scale", "True CFG scale", "number", "Sampling", minimum=1,
                    help="1 disables guidance. Negative prompting requires a value above 1 and a nonempty negative prompt.")
            if field.name == "guidance_scale" else field
            for field in common_fields(model, dimension_base=32)
        )
        fields = tuple(
            replace(field, help="Used only when true CFG is above 1.")
            if field.name == "negative_prompt" else field for field in fields
        )
        return definition(model, mode, workflow="qwen_2_1", supported_modes=self.ui_modes,
                          variant="qwen_image_2_1", fields=(
            *fields,
            *sampling_fields(model, ("flow_match_euler", "model_default")),
            UIField("use_kv_cache", "Cache prompt attention", "boolean", "Advanced",
                    help="Reuse prompt keys and values during denoising. Keep this fixed when comparing seeded images."),
            UIField("sigmas", "Custom sigma schedule", "number_list", "Advanced", nullable=True,
                    help="Optional descending values in (0, 1], separated by commas or newlines. Omit zero. "
                         "Blank uses the model schedule; a custom list overrides the inference step count."),
        ))

    def validate_form_settings(self, model, mode, settings):
        super().validate_form_settings(model, mode, settings)
        try:
            validate_sigmas(settings["sigmas"])
        except ValueError as exc:
            raise FormValidationError({"sigmas": str(exc)}) from exc

    def _build_pipeline(self, request):
        model = request.model
        options = self._load_options()
        path = model.asset_paths["weights"]
        transformer = None
        if path is not None:
            transformer = self._diffusers.QwenImage21Transformer2DModel.from_single_file(
                str(path), config=model.defaults["base_model_id"], subfolder="transformer", **options,
            ).to(self.device)
        cuda = self.device.startswith("cuda")
        try:
            if cuda:
                options["device_map"] = "cuda"
            if transformer is not None:
                options["transformer"] = transformer
            with self._torch.cuda.device(self.device) if cuda else nullcontext():
                return self._diffusers.QwenImage21Pipeline.from_pretrained(model.defaults["base_model_id"], **options)
        finally:
            del transformer

    def _load_lora(self, selection, **options):
        load_qwen21_lora(self._pipeline, selection.asset.asset_paths["weights"], selection.adapter_name)

    def generate(self, request):
        self.validate(request)
        try:
            self._load_runtime()
            self._ensure_pipeline(request, file_fingerprint(request.model.asset_paths["weights"]))
            self._configure_scheduler(request, self._diffusers.FlowMatchEulerDiscreteScheduler)
            self._configure_loras(request)
            if request.loras:
                self._pipeline.set_lora_device(
                    adapter_names=[item.adapter_name for item in request.loras], device=self.device,
                )
            settings = request.settings
            seeds = image_seeds(settings, settings["num_images_per_prompt"])
            arguments = {name: settings[name] for name in (
                "width", "height", "num_inference_steps", "true_cfg_scale", "use_kv_cache",
                "sigmas", "num_images_per_prompt",
            )}
            # A repository (notably Turbo) can define its own schedule. Match
            # Diffusers precedence so progress and metadata use the actual steps.
            effective_sigmas = settings["sigmas"]
            if effective_sigmas is None:
                effective_sigmas = self._pipeline.config.get("sample_sigmas")
            validate_sigmas(effective_sigmas)
            steps = len(effective_sigmas) if effective_sigmas is not None else settings["num_inference_steps"]
            cfg_enabled = settings["true_cfg_scale"] > 1 and bool(request.negative_prompt)
            generators = [self._torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]
            if getattr(self, "_progress_callback", None) is not None:
                def on_step_end(pipeline, step, timestep, tensors):
                    self.report_progress("denoising", min(step + 1, steps), steps)
                    if step + 1 == steps:
                        self.report_progress("decoding")
                    return tensors

                arguments["callback_on_step_end"] = on_step_end
                arguments["callback_on_step_end_tensor_inputs"] = []
            self.report_progress("denoising", 0, steps)
            self._synchronize()
            start = perf_counter()
            with self._torch.inference_mode():
                images = tuple(self._pipeline(
                    prompt=request.prompt, negative_prompt=request.negative_prompt if cfg_enabled else None,
                    generator=generators, **arguments,
                ).images)
            self._synchronize()
            elapsed = perf_counter() - start
            if len(images) != len(seeds):
                raise RuntimeError("The Qwen 2.1 pipeline returned an unexpected image count")
            from image_generator.generation import request_snapshot

            return GeneratedBatch(images, {
                **request_snapshot(request), "seeds": seeds, "scheduler": self._scheduler_metadata(),
                "true_cfg_enabled": cfg_enabled, "negative_prompt_enabled": cfg_enabled,
                "effective_num_inference_steps": steps,
                "effective_sample_sigmas": effective_sigmas,
                "runtime": self._runtime_metadata(elapsed),
            })
        except Exception:
            self.close()
            raise

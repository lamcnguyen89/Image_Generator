"""Instance-owned generation implementation."""

from contextlib import nullcontext
from time import perf_counter

from ..base import GeneratedBatch
from ..shared.assets import file_fingerprint
from ..shared.diffusers_runtime import TransformerWorkflow
from ..shared.seeds import image_seeds
from .adapters import load_qwen21_lora
from .parameters import DEFAULTS, MODES
from .validation import validate_sigmas


class Qwen21Workflow(TransformerWorkflow):
    workflow_id = "qwen_2_1"
    defaults = DEFAULTS
    ui_modes = MODES

    def _build_pipeline(self, request):
        model = request.model
        options = self._load_options()
        path = model.asset_paths["weights"]
        transformer = None
        if path is not None:
            transformer = self._diffusers.QwenImage21Transformer2DModel.from_single_file(
                str(path),
                config=model.defaults["base_model_id"],
                subfolder="transformer",
                **options,
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
                    adapter_names=[item.adapter_name for item in request.loras],
                    device=self.device,
                )
            settings = request.settings
            seeds = image_seeds(settings, settings["num_images_per_prompt"])
            arguments = {
                name: settings[name]
                for name in (
                    "width",
                    "height",
                    "num_inference_steps",
                    "true_cfg_scale",
                    "use_kv_cache",
                    "sigmas",
                    "num_images_per_prompt",
                )
            }
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
                images = tuple(
                    self._pipeline(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt if cfg_enabled else None,
                        generator=generators,
                        **arguments,
                    ).images
                )
            self._synchronize()
            elapsed = perf_counter() - start
            if len(images) != len(seeds):
                raise RuntimeError("The Qwen 2.1 pipeline returned an unexpected image count")
            from image_generator.generation import request_snapshot

            return GeneratedBatch(
                images,
                {
                    **request_snapshot(request),
                    "seeds": seeds,
                    "scheduler": self._scheduler_metadata(),
                    "true_cfg_enabled": cfg_enabled,
                    "negative_prompt_enabled": cfg_enabled,
                    "effective_num_inference_steps": steps,
                    "effective_sample_sigmas": effective_sigmas,
                    "runtime": self._runtime_metadata(elapsed),
                },
            )
        except Exception:
            self.close()
            raise

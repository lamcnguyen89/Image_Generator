"""Instance-owned generation implementation."""

from contextlib import nullcontext
from time import perf_counter

from ..base import GeneratedBatch
from ..shared.assets import file_fingerprint
from ..shared.diffusers_runtime import TransformerWorkflow
from ..shared.seeds import image_seeds
from .parameters import DEFAULTS, MODES


class Krea2Workflow(TransformerWorkflow):
    workflow_id = "krea2"
    defaults = DEFAULTS
    ui_modes = MODES

    def _build_pipeline(self, request):
        model = request.model
        options = self._load_options()
        path = model.asset_paths["weights"]
        transformer = None
        if path is not None:
            transformer = self._diffusers.Krea2Transformer2DModel.from_single_file(
                str(path),
                config=model.defaults["base_model_id"],
                subfolder="transformer",
                **options,
            ).to(self.device)
        cuda = self.device.startswith("cuda")
        try:
            # Match the notebook's direct CUDA loading, avoiding a second CPU
            # copy of supporting components on the shared-memory GPU host.
            if cuda:
                options["device_map"] = "cuda"
            if transformer is not None:
                options["transformer"] = transformer
            with self._torch.cuda.device(self.device) if cuda else nullcontext():
                return self._diffusers.Krea2Pipeline.from_pretrained(
                    model.defaults["base_model_id"],
                    is_distilled=False,
                    **options,
                )
        finally:
            del transformer

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
                    "guidance_scale",
                    "max_sequence_length",
                    "num_images_per_prompt",
                )
            }
            generators = [self._torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]
            steps = settings["num_inference_steps"]
            if getattr(self, "_progress_callback", None) is not None:

                def on_step_end(pipeline, step, timestep, tensors):
                    total = getattr(pipeline, "num_timesteps", steps)
                    if type(total) is not int or total <= 0:
                        total = steps
                    current = min(step + 1, total)
                    self.report_progress("denoising", current, total)
                    if current == total:
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
                        negative_prompt=request.negative_prompt,
                        generator=generators,
                        **arguments,
                    ).images
                )
            self._synchronize()
            elapsed = perf_counter() - start
            if len(images) != len(seeds):
                raise RuntimeError("The Krea2 pipeline returned an unexpected image count")
            from image_generator.generation import request_snapshot

            return GeneratedBatch(
                images,
                {
                    **request_snapshot(request),
                    "seeds": seeds,
                    "scheduler": self._scheduler_metadata(),
                    "cfg_enabled": settings["guidance_scale"] > 0,
                    "negative_prompt_enabled": settings["guidance_scale"] > 0 and bool(request.negative_prompt),
                    "runtime": self._runtime_metadata(elapsed),
                },
            )
        except Exception:
            self.close()
            raise

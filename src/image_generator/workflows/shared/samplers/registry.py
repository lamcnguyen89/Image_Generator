"""Trusted, lazy sampler bindings, keyed by workflow and stable JSON identifier.

A name can have different implementations in different workflows. Registering a
binding does not enable it in other workflows or override checkpoint restrictions.
"""

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType

from ..schedules import available_schedules, validate_schedule


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    label: str
    workflow: str
    factory: tuple[str, str]
    schedule_class: str | None = None
    schedule_backend: str = "model_default"
    incompatible_options: tuple[str, ...] = ()
    scheduler_class: str | None = None
    scheduler_options: tuple[tuple[str, object], ...] = ()
    native_name: str | None = None


_FLOW_FACTORY = "image_generator.workflows.shared.samplers.flow_match"
_DIFFUSERS_FACTORY = ("image_generator.workflows.shared.samplers.diffusers", "build_native")
_SPECS = (
    SamplerSpec("flow_match_euler", "Flow match Euler", "krea2",
                (_FLOW_FACTORY, "build_euler"), "FlowMatchEulerDiscreteScheduler", schedule_backend="flow_match"),
    SamplerSpec("er_sde", "ER SDE", "krea2",
                (_FLOW_FACTORY, "build_er_sde"), "FlowMatchEulerDiscreteScheduler",
                schedule_backend="flow_match",
                incompatible_options=("invert_sigmas", "stochastic_sampling")),
    # Preserve the existing Krea2 model_default spelling and Euler behavior.
    SamplerSpec("model_default", "Model default", "krea2",
                (_FLOW_FACTORY, "build_euler"), "FlowMatchEulerDiscreteScheduler", schedule_backend="flow_match"),
    SamplerSpec("flow_match_euler", "Flow match Euler", "flux",
                (_FLOW_FACTORY, "build_euler"), "FlowMatchEulerDiscreteScheduler", schedule_backend="flow_match"),
    SamplerSpec("model_default", "Model default", "flux",
                (_FLOW_FACTORY, "build_euler"), "FlowMatchEulerDiscreteScheduler", schedule_backend="flow_match"),
    SamplerSpec("flow_match_euler", "Flow match Euler", "qwen_2_1",
                (_FLOW_FACTORY, "build_euler"), "FlowMatchEulerDiscreteScheduler", schedule_backend="flow_match"),
    SamplerSpec("model_default", "Model default", "qwen_2_1",
                (_FLOW_FACTORY, "build_euler"), "FlowMatchEulerDiscreteScheduler", schedule_backend="flow_match"),
    SamplerSpec("model_default", "Model default", "sdxl", _DIFFUSERS_FACTORY),
    SamplerSpec("euler", "Euler", "sdxl", _DIFFUSERS_FACTORY,
                scheduler_class="EulerDiscreteScheduler", schedule_backend="euler_discrete"),
    SamplerSpec("euler_ancestral", "Euler ancestral", "sdxl", _DIFFUSERS_FACTORY,
                scheduler_class="EulerAncestralDiscreteScheduler", schedule_backend="euler_ancestral"),
    SamplerSpec("dpmpp_2m_sde", "DPM++ 2M SDE", "sdxl", _DIFFUSERS_FACTORY,
                schedule_backend="dpm_multistep",
                scheduler_class="DPMSolverMultistepScheduler",
                scheduler_options=(("algorithm_type", "sde-dpmsolver++"), ("solver_order", 2))),
    SamplerSpec("model_default", "Model default", "wan", _DIFFUSERS_FACTORY, schedule_backend="wan_unipc"),
    SamplerSpec("uni_pc", "UniPC", "wan",
                ("image_generator.workflows.shared.samplers.wan", "build_wan"),
                "UniPCMultistepScheduler", schedule_backend="wan_unipc", scheduler_class="UniPCMultistepScheduler",
                scheduler_options=(("prediction_type", "flow_prediction"), ("use_flow_sigmas", True))),
    SamplerSpec("dpmpp_2m", "DPM++ 2M", "wan",
                ("image_generator.workflows.shared.samplers.wan", "build_wan"),
                "UniPCMultistepScheduler", schedule_backend="wan_unipc", scheduler_class="DPMSolverMultistepScheduler",
                scheduler_options=(("prediction_type", "flow_prediction"), ("use_flow_sigmas", True),
                                   ("algorithm_type", "dpmsolver++"), ("solver_order", 2),
                                   ("solver_type", "midpoint"), ("final_sigmas_type", "zero"))),
    *(SamplerSpec(name, label, "minimax",
                  ("image_generator.workflows.shared.samplers.comfyui", "build_native"),
                  schedule_backend="comfyui_minimax_h3", native_name=name)
      for name, label in (("res_multistep", "RES multistep"), ("euler", "Euler"))),
    *(SamplerSpec(name, label, "sdxl", _DIFFUSERS_FACTORY,
                  scheduler_class="DPMSolverMultistepScheduler", schedule_backend="dpm_multistep",
                  scheduler_options=(("algorithm_type", algorithm), ("solver_order", order)))
      for name, label, algorithm, order in (
          ("dpmpp_2m", "DPM++ 2M", "dpmsolver++", 2),
          ("dpmpp_3m", "DPM++ 3M", "dpmsolver++", 3),
          ("dpmpp_3m_sde", "DPM++ 3M SDE", "sde-dpmsolver++", 3),
      )),
    *(SamplerSpec(name, label, "sdxl", _DIFFUSERS_FACTORY,
                  scheduler_class=cls, schedule_backend=backend, scheduler_options=options,
                  incompatible_options=("rescale_betas_zero_snr",) if backend in ("native_sigma", "deis_multistep")
                  or name == "pndm" else ())
      for name, label, cls, backend, options in (
          ("uni_pc", "UniPC", "UniPCMultistepScheduler", "unipc_multistep", (("solver_order", 2),)),
          ("heun", "Heun", "HeunDiscreteScheduler", "native_sigma", ()),
          ("lms", "LMS", "LMSDiscreteScheduler", "native_sigma", ()),
          ("dpm_2", "DPM2", "KDPM2DiscreteScheduler", "native_sigma", ()),
          ("dpm_2_ancestral", "DPM2 ancestral", "KDPM2AncestralDiscreteScheduler", "native_sigma", ()),
          ("ddim", "DDIM", "DDIMScheduler", "model_default", ()),
          ("ddpm", "DDPM", "DDPMScheduler", "model_default", ()),
          ("pndm", "PNDM", "PNDMScheduler", "model_default", (("skip_prk_steps", True),)),
          ("deis", "DEIS", "DEISMultistepScheduler", "deis_multistep", (("solver_order", 2),)),
      )),
    *(SamplerSpec("er_sde", "ER SDE", workflow, (_FLOW_FACTORY, "build_er_sde"),
                  "FlowMatchEulerDiscreteScheduler", schedule_backend="flow_match",
                  incompatible_options=("invert_sigmas", "stochastic_sampling"))
      for workflow in ("flux", "qwen_2_1")),
    *(SamplerSpec(name, label, workflow, (_FLOW_FACTORY, "build_solver"),
                  "FlowMatchEulerDiscreteScheduler", schedule_backend="flow_match",
                  incompatible_options=("invert_sigmas", "stochastic_sampling"),
                  scheduler_class=cls, scheduler_options=options)
      for workflow in ("flux", "qwen_2_1", "krea2")
      for name, label, cls, options in (
          ("flow_match_heun", "Flow match Heun", "FlowMatchHeunDiscreteScheduler", ()),
          ("uni_pc", "UniPC", "UniPCMultistepScheduler",
           (("prediction_type", "flow_prediction"), ("use_flow_sigmas", True), ("solver_order", 2))),
          ("dpmpp_2m", "DPM++ 2M", "DPMSolverMultistepScheduler",
           (("prediction_type", "flow_prediction"), ("use_flow_sigmas", True),
            ("algorithm_type", "dpmsolver++"), ("solver_order", 2))),
          ("dpmpp_2m_sde", "DPM++ 2M SDE", "DPMSolverMultistepScheduler",
           (("prediction_type", "flow_prediction"), ("use_flow_sigmas", True),
            ("algorithm_type", "sde-dpmsolver++"), ("solver_order", 2))),
      )),
)
SAMPLERS = MappingProxyType({(spec.workflow, spec.name): spec for spec in _SPECS})


def get_sampler(workflow, name):
    if not isinstance(name, str):
        raise ValueError("sampling.sampler must be a sampler name")
    try:
        return SAMPLERS[workflow, name]
    except KeyError:
        raise ValueError(f"Sampler {name!r} is not implemented for workflow {workflow!r}") from None


def _allowed_sampling(requirements):
    allowed = requirements.get("allowed_sampling")
    if allowed is not None and (
        not isinstance(allowed, list)
        or any(not isinstance(pair, dict) or set(pair) != {"sampler", "noise_schedule"}
               or any(not isinstance(value, str) for value in pair.values()) for pair in allowed)
    ):
        raise ValueError("requirements.allowed_sampling must be a list of sampler/noise_schedule objects")
    return allowed


def available_samplers(workflow, requirements):
    """Return registered bindings filtered by exact checkpoint sampling pairs."""
    allowed = _allowed_sampling(requirements)
    return tuple(
        spec for spec in SAMPLERS.values() if spec.workflow == workflow and (
            allowed is None or any(
                {"sampler": spec.name, "noise_schedule": schedule.name} in allowed
                for schedule in available_schedules(spec.schedule_backend)
            )
        )
    )


def validate_sampling(workflow, sampling, requirements, *, allow_extensions=False):
    """Validate without importing the adapter, Torch, or Diffusers."""
    fields = {"sampler", "noise_schedule"}
    if not isinstance(sampling, dict) or not fields <= sampling.keys() or (not allow_extensions and set(sampling) != fields):
        raise ValueError("sampling must contain sampler and noise_schedule")
    # Form saves preserve unedited extension metadata. Execution retains the
    # strict two-field sampling contract and rejects unknown runtime settings.
    sampling = {name: sampling[name] for name in fields}
    spec = get_sampler(workflow, sampling["sampler"])
    validate_schedule(spec.schedule_backend, sampling["noise_schedule"], requirements)
    allowed = _allowed_sampling(requirements)
    if allowed is not None and sampling not in allowed:
        raise ValueError("Sampling is prohibited by this checkpoint's allowed_sampling")
    if spec.schedule_class is not None and requirements.get("scheduler_class", spec.schedule_class) != spec.schedule_class:
        raise ValueError(f"Sampler {spec.name!r} requires a {spec.schedule_class} schedule")
    options = requirements.get("scheduler_config", {})
    for option, value in spec.scheduler_options:
        if option in options and options[option] != value:
            raise ValueError(f"Sampler {spec.name!r} requires scheduler_config.{option}={value!r}")
    for option in spec.incompatible_options:
        if options.get(option):
            raise ValueError(f"Sampler {spec.name!r} does not support scheduler_config.{option}")
    return spec


def create_scheduler(workflow, sampling, requirements, **runtime):
    """Build fresh request state, always using the loader's original config.

    Factory paths come exclusively from this code registry, never a manifest.
    Generators are request-owned, one per output, including scheduler noise.
    """
    spec = validate_sampling(workflow, sampling, requirements)
    factory = getattr(import_module(spec.factory[0]), spec.factory[1])
    return factory(spec=spec, sampling=sampling, requirements=requirements, **runtime)

"""Trusted noise-schedule definitions and explicit backend implementations.

Sampler bindings select a backend profile. A profile registers only schedules
its scheduler interface can implement; checkpoints can restrict exact pairs
further through allowed_sampling. Default always preserves the loader config.
"""

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    label: str


@dataclass(frozen=True)
class ScheduleBinding:
    backend: str
    name: str
    options: tuple[tuple[str, object], ...] = ()
    native_name: str | None = None
    adapter: tuple[str, str] | None = None
    adapter_options: tuple[tuple[str, object], ...] = ()

    @property
    def label(self):
        return SCHEDULES[self.name].label


SCHEDULES = MappingProxyType({spec.name: spec for spec in (
    ScheduleSpec("default", "Default"),
    ScheduleSpec("normal", "Normal"),
    ScheduleSpec("karras", "Karras"),
    ScheduleSpec("exponential", "Exponential"),
    ScheduleSpec("beta", "Beta"),
    ScheduleSpec("simple", "Simple"),
)})

# Different Diffusers classes accept different configuration flags. Each
# explicit curve clears competing grids inherited from the component config.
_CURVE_FLAGS = {
    "karras": "use_karras_sigmas",
    "exponential": "use_exponential_sigmas",
    "beta": "use_beta_sigmas",
}
_STANDARD_CURVES = ("karras", "exponential", "beta")
_STANDARD_FLAGS = tuple(_CURVE_FLAGS[name] for name in _STANDARD_CURVES)
_SIGMA_PROFILES = {
    "flow_match": (_STANDARD_CURVES, _STANDARD_FLAGS),
    "euler_discrete": (_STANDARD_CURVES, _STANDARD_FLAGS),
    "dpm_multistep": (_STANDARD_CURVES, (*_STANDARD_FLAGS, "use_lu_lambdas", "use_flow_sigmas")),
    "unipc_multistep": (_STANDARD_CURVES, (*_STANDARD_FLAGS, "use_flow_sigmas")),
    "native_sigma": (_STANDARD_CURVES, _STANDARD_FLAGS),
    "deis_multistep": (_STANDARD_CURVES, (*_STANDARD_FLAGS, "use_flow_sigmas")),
}
_BINDINGS = (
    *(ScheduleBinding(backend, "default") for backend in (*_SIGMA_PROFILES, "model_default", "euler_ancestral")),
    *(ScheduleBinding(
        backend, "normal",
        tuple((flag, False) for flag in flags)
        + (() if backend == "flow_match" else (("timestep_spacing", "linspace"), ("final_sigmas_type", "zero"))),
        adapter=("image_generator.workflows.shared.schedules.normal", "NormalScheduleAdapter"),
    ) for backend, (_, flags) in _SIGMA_PROFILES.items()
      if backend not in ("native_sigma", "deis_multistep")),
    ScheduleBinding("euler_ancestral", "normal", (("timestep_spacing", "linspace"),),
                    adapter=("image_generator.workflows.shared.schedules.normal", "NormalScheduleAdapter")),
    *(ScheduleBinding(backend, name, tuple((flag, flag == _CURVE_FLAGS[name]) for flag in flags))
      for backend, (names, flags) in _SIGMA_PROFILES.items() for name in names),
    *(ScheduleBinding(
        "euler_ancestral", name, (("timestep_spacing", "linspace"),),
        adapter=("image_generator.workflows.shared.schedules.euler_ancestral", "EulerAncestralScheduleAdapter"),
        adapter_options=tuple((flag, flag == _CURVE_FLAGS[name]) for flag in _STANDARD_FLAGS)
        + (("final_sigmas_type", "zero"), ("interpolation_type", "linear"), ("timestep_type", "discrete")),
    ) for name in _STANDARD_CURVES),
    ScheduleBinding("comfyui_minimax_h3", "simple", native_name="simple"),
)
SCHEDULE_BINDINGS = MappingProxyType({(binding.backend, binding.name): binding for binding in _BINDINGS})


def available_schedules(backend):
    return tuple(binding for binding in SCHEDULE_BINDINGS.values() if binding.backend == backend)


def get_schedule(backend, name):
    if not isinstance(name, str):
        raise ValueError("sampling.noise_schedule must be a schedule name")
    try:
        return SCHEDULE_BINDINGS[backend, name]
    except KeyError:
        raise ValueError(f"Unsupported noise schedule {name!r} for backend {backend!r}") from None


def validate_schedule(backend, name, requirements):
    binding = get_schedule(backend, name)
    options = requirements.get("scheduler_config", {})
    if not isinstance(options, dict):
        raise ValueError("requirements.scheduler_config must be an object")
    for option, value in (*binding.options, *binding.adapter_options):
        if option in options and options[option] != value:
            raise ValueError(f"Noise schedule {name!r} requires scheduler_config.{option}={value!r}")
    return binding


def scheduler_options(backend, name, requirements):
    """Return fresh constructor options; checkpoint conflicts never fall back."""
    return dict(validate_schedule(backend, name, requirements).options)


def apply_schedule(backend, name, requirements, scheduler):
    """Attach a trusted runtime adapter only when the schedule needs one."""
    binding = validate_schedule(backend, name, requirements)
    if binding.adapter is None:
        return scheduler
    module, attribute = binding.adapter
    adapter = getattr(import_module(module), attribute)
    return adapter(scheduler, backend=backend, **dict(binding.adapter_options))

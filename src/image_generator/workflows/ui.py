"""CPU-only field definitions and form validation, independent of Flask and values."""

from dataclasses import dataclass
import math
from typing import Any, Literal


class FormValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{name}: {message}" for name, message in errors.items()))


@dataclass(frozen=True)
class UIField:
    name: str
    label: str
    type: Literal["text", "textarea", "integer", "number", "boolean", "select", "string_list", "number_list", "file"]
    group: str
    help: str = ""
    required: bool = False
    nullable: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    scope: Literal["settings", "inputs"] = "settings"
    read_only: bool = False

    def validate(self, value: Any) -> str | None:
        if value is None and self.nullable:
            return None
        if self.type in {"text", "textarea", "select"}:
            if not isinstance(value, str):
                return "Enter text."
            if self.required and not value.strip():
                return "This field is required."
            if self.type == "select" and value not in dict(self.choices):
                return "Choose one of the available options."
        elif self.type == "boolean":
            if type(value) is not bool:
                return "Choose a boolean value."
        elif self.type == "string_list":
            if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                return "Enter one nonempty value per line."
        elif self.type == "number_list":
            if not isinstance(value, list) or not value:
                return "Enter a nonempty list of numbers, or leave blank for the default."
            try:
                if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
                    return "Every list entry must be a finite number."
            except OverflowError:
                return "Number is too large."
        elif self.type in {"integer", "number"}:
            if self.type == "integer" and type(value) is not int:
                return "Enter a whole number."
            if type(value) not in (int, float):
                return "Enter a number."
            try:
                if not math.isfinite(value):
                    return "Enter a finite number."
            except OverflowError:
                return "Number is too large."
            if self.minimum is not None and value < self.minimum:
                return f"Minimum value is {self.minimum}."
            if self.maximum is not None and value > self.maximum:
                return f"Maximum value is {self.maximum}."
            if self.step is not None and (value - (self.minimum or 0)) % self.step:
                return f"Use increments of {self.step} starting at {self.minimum or 0}."
        return None


@dataclass(frozen=True)
class UIDefinition:
    workflow: str
    mode: str
    supported_modes: tuple[str, ...]
    fields: tuple[UIField, ...]
    partial: str | None = None

    @property
    def groups(self) -> dict[str, tuple[UIField, ...]]:
        return {group: tuple(field for field in self.fields if field.group == group)
                for group in dict.fromkeys(field.group for field in self.fields)}

    def validate_settings(self, settings: dict) -> None:
        errors = {}
        for field in self.fields:
            if field.scope == "settings":
                error = field.validate(setting_value(settings, field.name))
                if error:
                    errors[field.name] = error
        if errors:
            raise FormValidationError(errors)


def setting_value(settings: dict, name: str):
    value = settings
    for part in name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def set_setting(settings: dict, name: str, value) -> None:
    parts = name.split(".")
    for part in parts[:-1]:
        settings = settings.setdefault(part, {})
    settings[parts[-1]] = value


def common_fields(model, *, dimension_base=16, video=False) -> tuple[UIField, ...]:
    multiple = model.defaults["requirements"].get("dimension_multiple", dimension_base)
    if type(multiple) is not int or multiple <= 0:
        raise ValueError("requirements.dimension_multiple must be a positive integer")
    multiple = math.lcm(dimension_base, multiple)
    return (
        UIField("prompt", "Prompt", "textarea", "Prompts", required=True,
                help="Describe the scene, subject, and style."),
        UIField("negative_prompt", "Negative prompt", "textarea", "Prompts"),
        UIField("width", "Width", "integer", "Output", minimum=multiple, step=multiple,
                help=f"Pixels; a multiple of {multiple}."),
        UIField("height", "Height", "integer", "Output", minimum=multiple, step=multiple),
        UIField("num_videos_per_prompt" if video else "num_images_per_prompt",
                "Video variations" if video else "Image count", "integer", "Output", minimum=1, step=1),
        UIField("seed", "Seed", "integer", "Output", nullable=True, minimum=0, maximum=2**63 - 1,
                help="Leave blank for a random seed. Fixed seeds increment for each output."),
        UIField("num_inference_steps", "Inference steps", "integer", "Sampling", minimum=1, step=1),
        UIField("guidance_scale", "High-noise guidance" if video else "Guidance scale", "number", "Sampling", minimum=0),
        UIField("positive_prefix", "Prompt prefix", "textarea", "Advanced"),
        UIField("positive_suffix", "Prompt suffix", "textarea", "Advanced"),
        UIField("trigger_words", "Model trigger words", "string_list", "Advanced",
                help="One per line. Included only when the trigger-word option is selected."),
    )


def sampling_fields(model, samplers: tuple[str, ...]) -> tuple[UIField, ...]:
    labels = {"model_default": "Model default", "euler": "Euler", "euler_ancestral": "Euler ancestral",
              "dpmpp_2m_sde": "DPM++ 2M SDE", "flow_match_euler": "Flow match Euler"}
    allowed = model.defaults["requirements"].get("allowed_sampling")
    if allowed is not None:
        if not isinstance(allowed, list):
            raise ValueError("requirements.allowed_sampling must be a list")
        samplers = tuple(name for name in samplers
                         if {"sampler": name, "noise_schedule": "default"} in allowed)
    if not samplers:
        raise ValueError("This checkpoint has no sampling options supported by its workflow")
    return (
        UIField("sampling.sampler", "Sampler", "select", "Sampling",
                choices=tuple((name, labels[name]) for name in samplers)),
        UIField("sampling.noise_schedule", "Noise schedule", "select", "Advanced",
                choices=(("default", "Default"),)),
    )


def definition(model, mode, *, workflow, fields, supported_modes, partial=None, variant=None) -> UIDefinition:
    if model.kind != "model" or model.family != workflow or model.defaults["workflow"] != workflow:
        raise ValueError(f"This form requires a {workflow.upper()} model")
    if variant and model.defaults.get("model_variant") != variant:
        raise ValueError(f"This form supports only model_variant {variant}")
    modes = tuple(value for value in supported_modes if value in model.defaults["supported_modes"])
    if not modes or mode not in modes:
        raise ValueError("The requested mode is not supported by this model and workflow")
    return UIDefinition(workflow, mode, modes, tuple(fields), partial)


def frame_constraints(requirements: dict, *, native_step=4, native_offset=1) -> tuple[int, int]:
    """Combine checkpoint and native frame congruences (WAN by default)."""
    rule = requirements.get("frame_count", {"minimum": 5, "step": native_step, "offset": native_offset})
    if (not isinstance(rule, dict)
            or any(type(rule.get(key)) is not int for key in ("minimum", "step", "offset"))
            or rule["step"] <= 0 or rule["minimum"] < 1):
        raise ValueError("requirements.frame_count needs integer minimum, positive step, and offset")
    divisor = math.gcd(native_step, rule["step"])
    if (rule["offset"] - native_offset) % divisor:
        raise ValueError("The checkpoint frame_count rule cannot satisfy the workflow's native frame constraint")
    modulus = rule["step"] // divisor
    index = ((rule["offset"] - native_offset) // divisor * pow(native_step // divisor, -1, modulus)) % modulus
    first, step = native_offset + native_step * index, math.lcm(native_step, rule["step"])
    minimum = max(5, rule["minimum"])
    first += max(0, (minimum - first + step - 1) // step) * step
    return first, step

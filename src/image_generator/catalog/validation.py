"""Structural validation for catalog manifests; no model libraries are needed.

Workflow-specific ranges, scheduler constraints, and actual loader compatibility
belong to the workflows. Unknown metadata and setting names are retained so new
workflows can add fields without changing discovery.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal


AssetKind = Literal["model", "lora"]


@dataclass(frozen=True)
class ValidationIssue:
    path: Path
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.field}: {self.message}"


class CatalogValidationError(ValueError):
    """One or more file/field errors prevented building a valid catalog."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("\n".join(map(str, issues)))


def load_json(path: Path) -> object:
    """Read strict JSON, rejecting duplicate keys and non-finite numbers."""

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"{value} is not a finite JSON number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{value} is not a finite JSON number")
        return number

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise CatalogValidationError([ValidationIssue(path, "$", str(exc))]) from exc


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: object) -> bool:
    return isinstance(value, list) and all(_nonempty_string(item) for item in value)


def _finite_number(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def validate_settings(
    settings: object, path: Path, field: str = "settings"
) -> list[ValidationIssue]:
    """Check shared setting shapes, allowing partial overrides and new fields.

    Only seed and clip_skip among the known scalar settings accept null. New
    workflow fields may define their own null semantics in workflow validation.
    """
    issues = []

    def error(name, message):
        issues.append(ValidationIssue(path, name, message))

    if not isinstance(settings, dict):
        return [ValidationIssue(path, field, "must be an object")]

    for name, value in settings.items():
        location = f"{field}.{name}"
        if name in {"prompt", "negative_prompt", "positive_prefix", "positive_suffix"}:
            if not isinstance(value, str):
                error(location, "must be a string")
        elif name == "trigger_words":
            if not _string_list(value):
                error(location, "must be a list of nonempty strings")
        elif name in {"seed", "clip_skip"}:
            if value is not None and type(value) is not int:
                error(location, "must be an integer or null")
        elif name in {
            "width", "height", "num_inference_steps", "num_images_per_prompt",
            "num_videos_per_prompt", "num_frames", "max_sequence_length",
        }:
            if type(value) is not int:
                error(location, "must be an integer")
        elif name in {"strength", "guidance_scale", "guidance_scale_2", "true_cfg_scale", "fps"}:
            if not _finite_number(value):
                error(location, "must be a finite number")
        elif name in {"preserve_aspect_ratio", "enable_vae_tiling"}:
            if type(value) is not bool:
                error(location, "must be a boolean")
        elif name == "sampling":
            if not isinstance(value, dict):
                error(location, "must be an object")
            else:
                for key in ("sampler", "noise_schedule"):
                    if key in value and not _nonempty_string(value[key]):
                        error(f"{location}.{key}", "must be a nonempty string")
        elif name == "recommended_generation":
            issues.extend(validate_settings(value, path, location))
        elif name == "loras":
            if not isinstance(value, list):
                error(location, "must be a list")
                continue
            seen = set()
            for index, link in enumerate(value):
                link_field = f"{location}[{index}]"
                if not isinstance(link, dict):
                    error(link_field, "must be an object")
                    continue
                identifier = link.get("id")
                if not _nonempty_string(identifier):
                    error(f"{link_field}.id", "must be a nonempty string")
                elif identifier in seen:
                    error(f"{link_field}.id", f"duplicate LoRA link {identifier!r}")
                else:
                    seen.add(identifier)
                if type(link.get("enabled")) is not bool:
                    error(f"{link_field}.enabled", "must be a boolean")
                if "strength" in link:
                    strength = link["strength"]
                    if not _finite_number(strength):
                        error(f"{link_field}.strength", "must be a finite number")
    return issues


def _validate_header(document: object, path: Path) -> list[ValidationIssue]:
    if not isinstance(document, dict):
        return [ValidationIssue(path, "$", "must be an object")]
    issues = []
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        issues.append(ValidationIssue(path, "schema_version", "must be the supported version 1"))
    issues.extend(validate_settings(document.get("settings"), path))
    return issues


def validate_defaults(document: object, path: Path, kind: AssetKind) -> list[ValidationIssue]:
    issues = _validate_header(document, path)
    if not isinstance(document, dict):
        return issues

    def error(field, message):
        issues.append(ValidationIssue(path, field, message))

    for field in ("id", "display_name", "family"):
        if not _nonempty_string(document.get(field)):
            error(field, "must be a nonempty string")
    for field in ("example_prompts", "example_negative_prompts"):
        if field in document and not _string_list(document[field]):
            error(field, "must be a list of nonempty strings")
    for field in (
        "weights", "base_model_id", "notes_file", "preview_image", "sha256",
        "source_url", "source_model_id", "source_version_id", "model_variant",
        "weights_component",
    ):
        if field in document and document[field] is not None and not _nonempty_string(document[field]):
            error(field, "must be a nonempty string or null")

    if "weights" not in document:
        error("weights", "is required (use null for a pretrained model)")
    if kind == "model":
        if not _nonempty_string(document.get("workflow")):
            error("workflow", "must be a nonempty string")
        modes = document.get("supported_modes")
        if not _string_list(modes) or not modes:
            error("supported_modes", "must be a nonempty list of nonempty strings")
        if not isinstance(document.get("requirements"), dict):
            error("requirements", "must be an object")
        if document.get("weights") is None and not _nonempty_string(document.get("base_model_id")):
            error("base_model_id", "is required when weights is null")
    else:
        if document.get("adapter_type") != "lora":
            error("adapter_type", "must be 'lora'")
        if not _nonempty_string(document.get("weights")):
            error("weights", "must be a nonempty local asset path")
        compatibility = document.get("compatibility")
        if not isinstance(compatibility, dict):
            error("compatibility", "must be an object")
        else:
            if not _string_list(compatibility.get("verified_checkpoint_ids", [])):
                error("compatibility.verified_checkpoint_ids", "must be a list of nonempty strings")
            for field in ("trained_on", "model_variant", "expert_target"):
                value = compatibility.get(field)
                if value is not None and not _nonempty_string(value):
                    error(f"compatibility.{field}", "must be a nonempty string or null")
    return issues


def validate_user(document: object, path: Path) -> list[ValidationIssue]:
    issues = _validate_header(document, path)
    if isinstance(document, dict):
        for field in sorted(document.keys() - {"schema_version", "settings"}):
            issues.append(ValidationIssue(path, field, "user files may override only settings"))
    return issues

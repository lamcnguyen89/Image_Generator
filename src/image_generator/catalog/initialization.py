"""Create missing asset manifests without importing workflows or reading weights."""

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from .validation import (
    CatalogValidationError, ValidationIssue, load_json, validate_defaults, validate_user,
)


KINDS = {"checkpoints": "model", "loras": "lora"}
logger = logging.getLogger(__name__)


class _IncompleteAssetError(CatalogValidationError):
    """An asset needs more files or explicit metadata before auto-registration."""


def _exists(path: Path) -> bool:
    # Broken symlinks are existing entries too; never replace them.
    return path.exists() or path.is_symlink()


def _weight_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() == ".safetensors")


def _write_missing(path: Path, document: dict) -> bool:
    """Publish complete JSON atomically, preserving a concurrent creator's file."""
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".manifest-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _weight_path(directory: Path, kind: str, template: dict) -> str | None:
    if kind == "model" and template.get("weights_component") == "pipeline":
        candidates = ([directory] if (directory / "model_index.json").is_file() else [])
        candidates.extend(path.parent for path in sorted(directory.glob("*/model_index.json"))
                          if path.is_file())
        if not candidates and not _weight_files(directory):
            # WAN's template explicitly represents a pretrained repository.
            if template.get("weights") is None and template.get("base_model_id"):
                return None
    else:
        candidates = _weight_files(directory)
    if len(candidates) != 1:
        expected = "a complete pipeline directory" if template.get("weights_component") == "pipeline" else "one .safetensors file"
        raise _IncompleteAssetError([ValidationIssue(
            directory / f"{kind}.defaults.json", "weights",
            f"Cannot determine weights; expected {expected}. "
            "Call initialize_asset(..., weights=...) or supply explicit metadata for this asset.",
        )])
    return candidates[0].relative_to(directory).as_posix()


def initialize_asset(
    asset_dir: str | Path, *, weights: str | Path | None = None,
    metadata: dict | None = None,
) -> tuple[Path, ...]:
    """Ensure an existing <family>/{checkpoints,loras}/<asset> has both JSON files.

    New defaults come from the adjacent <kind>.defaults.example.json. Identity
    and weight paths are filled in; illustrative prompts, links, recommendations,
    and compatibility claims are cleared. User examples supply the schema only,
    never saved overrides. Existing files (including invalid JSON) are untouched.

    Upload handlers can pass a specific local weight path and metadata for NEW
    defaults, e.g. compatibility or a pretrained model's weights=None/base_model_id.
    Local paths resolve relative to asset_dir. Returns only paths actually created.
    """
    directory = Path(asset_dir).expanduser().resolve()
    kind = KINDS.get(directory.parent.name)
    if kind is None or not directory.is_dir():
        raise CatalogValidationError([ValidationIssue(
            directory, "$", "expected an existing <family>/checkpoints/<asset> or <family>/loras/<asset> directory",
        )])
    defaults_path = directory / f"{kind}.defaults.json"
    user_path = directory / f"{kind}.user.json"
    pending = []
    try:
        if not _exists(defaults_path):
            template_path = directory.parent / f"{kind}.defaults.example.json"
            try:
                document = load_json(template_path)
            except CatalogValidationError as exc:
                if isinstance(exc.__cause__, FileNotFoundError):
                    raise _IncompleteAssetError(exc.issues) from exc
                raise
            issues = validate_defaults(document, template_path, kind)
            if issues:
                raise CatalogValidationError(issues)
            document = deepcopy(document)
            document.update({
                "display_name": directory.name,
                "family": directory.parent.parent.name.lower(),
                "description": "Automatically initialized from the family template. Review metadata and settings before generation.",
                "example_prompts": [],
                "example_negative_prompts": [],
            })
            for key in ("source_url", "source_model_id", "source_version_id", "sha256", "notes_file", "preview_image"):
                document[key] = None
            for key in ("prompt", "negative_prompt", "positive_prefix", "positive_suffix"):
                if key in document["settings"]:
                    document["settings"][key] = ""
            document["settings"]["trigger_words"] = []
            if kind == "model":
                document["settings"]["loras"] = []
            else:
                document["settings"]["recommended_generation"] = {}
                document["compatibility"] = {
                    "trained_on": None, "model_variant": None, "expert_target": None,
                    "verified_checkpoint_ids": [], "notes": "",
                }
            document.update(deepcopy(metadata or {}))
            document["id"] = f"{kind}_{uuid4()}"
            if weights is not None:
                path = Path(weights).expanduser()
                path = (directory / path).resolve() if not path.is_absolute() else path.resolve()
                if not path.exists():
                    raise CatalogValidationError([ValidationIssue(defaults_path, "weights", f"weights not found: {path}")])
                document["weights"] = os.path.relpath(path, directory)
            elif metadata is None or "weights" not in metadata:
                document["weights"] = _weight_path(directory, kind, document)
            issues = validate_defaults(document, defaults_path, kind)
            if issues:
                raise CatalogValidationError(issues)
            pending.append((defaults_path, document))

        if not _exists(user_path):
            template_path = directory.parent / f"{kind}.user.example.json"
            document = load_json(template_path) if _exists(template_path) else {"schema_version": 1, "settings": {}}
            issues = validate_user(document, template_path)
            if issues:
                raise CatalogValidationError(issues)
            document = {"schema_version": document["schema_version"], "settings": {}}
            pending.append((user_path, document))

        return tuple(path for path, document in pending if _write_missing(path, document))
    except OSError as exc:
        raise CatalogValidationError([ValidationIssue(directory, "$", f"Cannot initialize manifests: {exc}")]) from exc


def initialize_catalog(root: str | Path) -> tuple[Path, ...]:
    """Initialize immediate asset folders, collecting errors across the catalog.

    Recognize existing manifests, loose safetensors, and complete pipeline
    directories. Ignore empty, hidden, and unrelated folders and example files.
    Skip assets whose defaults cannot yet be created because their family
    template or unambiguous weights are missing, logging the reason. Invalid
    configuration and I/O failures still produce aggregated errors.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise CatalogValidationError([ValidationIssue(root, "$", "catalog root must be an existing directory")])
    created, issues = [], []
    for folder, kind in KINDS.items():
        for directory in sorted(root.glob(f"*/{folder}/*")):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                recognized = (
                    _exists(directory / f"{kind}.defaults.json")
                    or _exists(directory / f"{kind}.user.json")
                    or bool(_weight_files(directory))
                    or (directory / "model_index.json").is_file()
                    or any(directory.glob("*/model_index.json"))
                )
                if recognized:
                    created.extend(initialize_asset(directory))
            except _IncompleteAssetError as exc:
                logger.warning("Skipping incomplete %s asset %s: %s", kind, directory, exc)
            except CatalogValidationError as exc:
                issues.extend(exc.issues)
            except OSError as exc:
                issues.append(ValidationIssue(directory, "$", str(exc)))
    if issues:
        raise CatalogValidationError(issues)
    return tuple(created)

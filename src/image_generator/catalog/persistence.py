"""Scoped, optimistic settings writes. No weights or runtime libraries are read."""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .discovery import discover_catalog
from .media import example_media
from .settings import lora_strength_minimum, resolve_settings
from .validation import CatalogValidationError, ValidationIssue, validate_defaults, validate_user


class SettingsConflict(ValueError):
    """The settings displayed by the editor have changed since it was opened."""


def settings_revision(catalog):
    # Include defaults and adapters: their inheritance/compatibility can change
    # the meaning of an unchanged model user file as well. The generated media
    # list only changes previews and must not invalidate an open settings form.
    documents = [(asset.id, {key: value for key, value in asset.defaults.items() if key != "example_media"},
                  asset.user_overrides)
                 for asset in sorted([*catalog.models.values(), *catalog.loras.values()], key=lambda a: a.id)]
    return hashlib.sha256(json.dumps(documents, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _difference(value, inherited):
    if isinstance(value, dict) and isinstance(inherited, dict):
        return {key: _difference(item, inherited.get(key)) for key, item in value.items()
                if key not in inherited or item != inherited[key]}
    return deepcopy(value)


def _scoped_overrides(existing, inherited, desired, scope, remove=()):
    """Replace only named leaves, pruning empty parent objects after removals."""
    result = deepcopy(existing)
    for name in dict.fromkeys([*scope, *remove]):
        parts = name.split(".")
        target, base, value = result, inherited, desired
        parents = []
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = target[part] = {}
            parents.append((target, part))
            target = child
            base = base.get(part, {}) if isinstance(base, dict) else {}
            value = value.get(part, {}) if isinstance(value, dict) else {}
        key = parts[-1]
        target.pop(key, None)
        if name not in remove and isinstance(value, dict) and key in value:
            if not isinstance(base, dict) or key not in base or value[key] != base[key]:
                target[key] = _difference(value[key], base.get(key) if isinstance(base, dict) else None)
        for parent, part in reversed(parents):
            if parent[part] == {}:
                del parent[part]
    return result


def adapter_unavailable(model, asset):
    compatibility = asset.defaults["compatibility"]
    variant = compatibility.get("model_variant")
    if asset.family != model.family:
        return "This adapter belongs to a different family."
    if variant and model.defaults.get("model_variant") and variant != model.defaults["model_variant"]:
        return "This adapter targets a different model variant."
    if model.defaults["workflow"] == "wan":
        if variant != model.defaults.get("model_variant") or compatibility.get("expert_target") not in {"high_noise", "low_noise"}:
            return "Set the WAN model variant and high-noise or low-noise expert target before use."
    if model.defaults["workflow"] == "minimax" and variant != "minimax_h3":
        return "Record explicit MiniMax H3 compatibility before using this adapter."
    return ""


@contextmanager
def _locked_catalog(catalog, expected_revision):
    # A separate lock survives os.replace and serializes this service's writers
    # across threads/processes. Editors still need an explicit revision token.
    with (catalog.root / ".settings.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = discover_catalog(catalog.root, initialize_missing=False)
        if not expected_revision or settings_revision(current) != expected_revision:
            raise SettingsConflict("Saved settings changed. Reload saved settings before saving again.")
        yield current


def _write(asset, overrides):
    document = {"schema_version": 1, "settings": overrides}
    issues = validate_user(document, asset.user_path)
    if issues:
        raise CatalogValidationError(issues)
    return _write_document(asset.user_path, document)


def _write_document(path, document):
    content = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return document


def _media_names(asset):
    directory = asset.manifest_path.parent.resolve()
    return [path.relative_to(directory).as_posix() for path in example_media(asset)]


def sync_example_media(catalog):
    """Refresh generated media lists, preserving concurrent settings edits.

    Share the settings writers' lock and reload before writing. Stable lists
    leave manifest bytes and mtimes intact; assets without media need no field.
    Read-only discovery never calls this function.
    """
    assets = [*catalog.models.values(), *catalog.loras.values()]
    if all(asset.defaults.get("example_media", []) == _media_names(asset) for asset in assets):
        return catalog
    with (catalog.root / ".settings.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = discover_catalog(catalog.root, initialize_missing=False)
        for asset in [*current.models.values(), *current.loras.values()]:
            names = _media_names(asset)
            if asset.defaults.get("example_media", []) == names:
                continue
            document = {**asset.defaults, "example_media": names}
            try:
                _write_document(asset.manifest_path, document)
            except OSError as exc:
                raise CatalogValidationError([ValidationIssue(
                    asset.manifest_path, "example_media", f"Cannot synchronize example media: {exc}",
                )]) from exc
            asset.defaults["example_media"] = names
        return current


def update_examples(catalog, asset_id, field, action, *, expected_revision, text=None, index=None):
    """Edit one example list in defaults while preserving all other metadata."""
    if not isinstance(field, str) or field not in {"example_prompts", "example_negative_prompts"}:
        raise ValueError("Choose example prompts or example negative prompts.")
    if not isinstance(action, str) or action not in {"add", "remove"}:
        raise ValueError("Choose add or remove example.")
    with _locked_catalog(catalog, expected_revision) as current:
        asset = current.get_asset(asset_id)
        document = deepcopy(asset.defaults)
        examples = document.setdefault(field, [])
        if action == "add":
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Enter a nonempty example prompt.")
            text = text.strip()
            if text in examples:
                raise ValueError("This example is already saved.")
            examples.append(text)
        else:
            if type(index) is not int or not 0 <= index < len(examples):
                raise ValueError("Select a saved example to remove.")
            examples.pop(index)
        issues = validate_defaults(document, asset.manifest_path, asset.kind)
        if issues:
            raise CatalogValidationError(issues)
        _write_document(asset.manifest_path, document)
        asset.defaults[field] = examples
        return {"examples": examples, "settings_revision": settings_revision(current)}


def save_model_settings(catalog, model_id, desired, *, scope, expected_revision, mode=None, remove=()):
    from image_generator.workflows.registry import create_workflow

    with _locked_catalog(catalog, expected_revision) as current:
        model = current.get_model(model_id)
        workflow = create_workflow(model.defaults["workflow"])
        mode = mode or workflow.ui_modes[0]
        ui = workflow.get_ui_definition(model, mode)
        allowed = {field.name for field in ui.fields if field.scope == "settings"} | {"loras"}
        if (set(scope) | set(remove)) - allowed:
            raise ValueError("The editor cannot change these settings.")
        inherited = resolve_settings(model.defaults["settings"], workflow_defaults=workflow.defaults)
        overrides = _scoped_overrides(model.user_overrides["settings"], inherited, desired, scope, remove)
        effective = resolve_settings(inherited, overrides)
        issues = validate_user({"schema_version": 1, "settings": overrides}, model.user_path)
        if issues:
            raise CatalogValidationError(issues)
        # Saving defaults permits an empty prompt. Keep all other workflow form
        # constraints, including workflow-specific validation, in force.
        validation = deepcopy(effective)
        if validation.get("prompt") == "":
            validation["prompt"] = "Settings validation"
        workflow.validate_form_settings(model, mode, validation)
        for link in effective.get("loras", []):
            asset = current.loras.get(link["id"])
            if asset is None:
                raise ValueError("Unknown adapter ID.")
            if link["enabled"] and adapter_unavailable(model, asset):
                raise ValueError(adapter_unavailable(model, asset))
            strength = link.get("strength", asset.resolve_settings().get("strength", 0.8))
            minimum = lora_strength_minimum(model.family)
            if minimum is not None and strength < minimum:
                raise ValueError("Adapter strength must be at least zero.")
        return _write(model, overrides)


def save_adapter_settings(catalog, adapter_id, desired, *, expected_revision, remove=()):
    with _locked_catalog(catalog, expected_revision) as current:
        asset = current.get_lora(adapter_id)
        allowed = {"strength", "trigger_words"}
        if (set(desired) | set(remove)) - allowed:
            raise ValueError("The adapter editor supports strength and trigger words only.")
        overrides = _scoped_overrides(asset.user_overrides["settings"], asset.defaults["settings"],
                                      desired, desired.keys(), remove)
        effective = resolve_settings(asset.defaults["settings"], overrides)
        issues = validate_user({"schema_version": 1, "settings": overrides}, asset.user_path)
        if issues:
            raise CatalogValidationError(issues)
        minimum = lora_strength_minimum(asset.family)
        if minimum is not None and effective.get("strength", 0.8) < minimum:
            raise ValueError("Adapter strength must be at least zero.")
        return _write(asset, overrides)


def reset_settings(catalog, asset_id, *, expected_revision):
    with _locked_catalog(catalog, expected_revision) as current:
        return _write(current.get_asset(asset_id), {})

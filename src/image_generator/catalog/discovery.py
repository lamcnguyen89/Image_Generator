"""Discover model and LoRA manifests without reading weights or loading pipelines."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .initialization import initialize_catalog
from .settings import resolve_settings
from .validation import (
    AssetKind,
    CatalogValidationError,
    ValidationIssue,
    load_json,
    validate_defaults,
    validate_settings,
    validate_user,
)


@dataclass(frozen=True)
class CatalogAsset:
    kind: AssetKind
    manifest_path: Path
    user_path: Path
    defaults: dict[str, Any]
    user_overrides: dict[str, Any]

    @property
    def id(self) -> str:
        return self.defaults["id"]

    @property
    def display_name(self) -> str:
        return self.defaults["display_name"]

    @property
    def family(self) -> str:
        return self.defaults["family"]

    @property
    def asset_paths(self) -> dict[str, Path | None]:
        """Resolve local paths; remote base_model_id values remain metadata.

        Existence is deliberately not checked: pretrained models and clones
        without the ignored weight files can still be browsed in the catalog.
        """
        return {
            field: (self.manifest_path.parent / value).resolve() if value is not None else None
            for field in ("weights", "notes_file", "preview_image")
            for value in (self.defaults.get(field),)
        }

    def resolve_settings(
        self,
        *,
        workflow_defaults: dict[str, Any] | None = None,
        request_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return independent effective settings, leaving all metadata intact."""
        result = resolve_settings(
            self.defaults["settings"],
            self.user_overrides["settings"],
            workflow_defaults=workflow_defaults,
            request_overrides=request_overrides,
        )
        issues = validate_settings(result, self.manifest_path, "effective_settings")
        if issues:
            raise CatalogValidationError(issues)
        return result


@dataclass(frozen=True)
class Catalog:
    root: Path
    models: dict[str, CatalogAsset]
    loras: dict[str, CatalogAsset]

    def get_model(self, model_id: str) -> CatalogAsset:
        return self.models[model_id]

    def get_lora(self, lora_id: str) -> CatalogAsset:
        return self.loras[lora_id]

    def get_asset(self, asset_id: str) -> CatalogAsset:
        if asset_id in self.models:
            return self.models[asset_id]
        return self.loras[asset_id]


def discover_catalog(root: str | Path, *, initialize_missing: bool = True) -> Catalog:
    """Index <family>/{checkpoints,loras}/<asset> manifests by permanent ID.

    Missing manifests are initialized from family examples before discovery.
    Existing media files are synchronized to defaults' example_media lists.
    Assets without defaults that cannot yet be initialized are skipped with a
    warning, so an unfinished family does not prevent loading configured assets.
    Set initialize_missing=False for strictly read-only inspection, where a
    missing user file means no overrides. Only exact defaults filenames register
    entries; example files themselves are ignored.
    Invalid files and duplicate IDs produce one aggregated exception, never a
    partially valid catalog or silent fallback to defaults.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise CatalogValidationError([ValidationIssue(root, "$", "catalog root must be an existing directory")])

    if initialize_missing:
        initialize_catalog(root)

    catalog = Catalog(root, models={}, loras={})
    issues: list[ValidationIssue] = []
    identifiers: dict[str, Path] = {}
    for kind, directory in (("model", "checkpoints"), ("lora", "loras")):
        for path in sorted(root.glob(f"*/{directory}/*/{kind}.defaults.json")):
            user_path = path.with_name(f"{kind}.user.json")
            file_issues = []
            try:
                defaults = load_json(path)
            except CatalogValidationError as exc:
                issues.extend(exc.issues)
                continue
            file_issues.extend(validate_defaults(defaults, path, kind))
            # Track IDs even if other fields are invalid, so duplicates are visible.
            identifier = defaults.get("id") if isinstance(defaults, dict) else None
            if isinstance(identifier, str) and identifier.strip():
                if identifier in identifiers:
                    file_issues.append(ValidationIssue(
                        path, "id", f"duplicate ID {identifier!r}; first declared in {identifiers[identifier]}"
                    ))
                else:
                    identifiers[identifier] = path
            try:
                user = load_json(user_path)
            except CatalogValidationError as exc:
                # Only absence means no overrides; permission/encoding/JSON errors fail.
                if isinstance(exc.__cause__, FileNotFoundError):
                    user = {"schema_version": 1, "settings": {}}
                else:
                    file_issues.extend(exc.issues)
                    user = None
            else:
                file_issues.extend(validate_user(user, user_path))
            if file_issues:
                issues.extend(file_issues)
                continue
            asset = CatalogAsset(kind, path, user_path, defaults, user)
            index = catalog.models if kind == "model" else catalog.loras
            index[asset.id] = asset

    for model in catalog.models.values():
        # Check both source lists: an invalid default link should not be hidden
        # merely because the user currently replaces it with another list.
        for document, path in ((model.defaults, model.manifest_path), (model.user_overrides, model.user_path)):
            for index, link in enumerate(document["settings"].get("loras", [])):
                if link["id"] not in catalog.loras:
                    issues.append(ValidationIssue(
                        path, f"settings.loras[{index}].id", f"unknown LoRA ID {link['id']!r}"
                    ))
    if issues:
        raise CatalogValidationError(issues)
    if initialize_missing:
        from .persistence import sync_example_media

        return sync_example_media(catalog)
    return catalog

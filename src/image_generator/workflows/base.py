"""Shared generation contracts, independent of GPU libraries and Flask."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

from image_generator.catalog import CatalogAsset
from .ui import UIDefinition


@dataclass(frozen=True)
class LoRASelection:
    asset: CatalogAsset
    settings: dict[str, Any]
    strength: float
    adapter_name: str

    @property
    def expert_target(self):
        return self.settings.get("expert_target", self.asset.defaults["compatibility"].get("expert_target"))


@dataclass(frozen=True)
class GenerationRequest:
    model: CatalogAsset
    mode: str
    settings: dict[str, Any]
    prompt: str
    negative_prompt: str
    loras: tuple[LoRASelection, ...] = ()
    inputs: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedBatch:
    images: tuple[Any, ...]
    metadata: dict[str, Any]
    videos: tuple[Any, ...] = ()
    conditioning_image: Any = None


@dataclass(frozen=True)
class SavedImage:
    image_path: Path
    metadata_path: Path
    seed: int


@dataclass(frozen=True)
class SavedVideo:
    video_path: Path
    metadata_path: Path
    seed: int
    conditioning_path: Path | None = None


@dataclass(frozen=True)
class WorkflowProgress:
    stage: str
    current: int | None = None
    total: int | None = None


class Workflow(ABC):
    defaults: dict[str, Any]
    ui_modes: tuple[str, ...] = ()

    def set_progress_callback(self, callback: Callable[[WorkflowProgress], None] | None) -> None:
        """Attach a process-local observer; callbacks are never request settings."""
        self._progress_callback = callback

    def report_progress(self, stage: str, current=None, total=None) -> None:
        callback = getattr(self, "_progress_callback", None)
        if callback is not None:
            callback(WorkflowProgress(stage, current, total))

    def get_ui_definition(self, model: CatalogAsset, mode: str) -> UIDefinition:
        """Describe controls without loading weights; current values are separate."""
        raise ValueError("This workflow does not provide a generation form")

    def validate_form_settings(self, model: CatalogAsset, mode: str, settings: dict) -> None:
        """Validate form values only; full validate() still checks assets before jobs."""
        self.get_ui_definition(model, mode).validate_settings(settings)

    @abstractmethod
    def validate(self, request: GenerationRequest) -> None:
        """Validate without importing model libraries or loading weights."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GeneratedBatch:
        """Generate with an instance-owned pipeline; calls must be serialized."""

    def generate_batches(self, request: GenerationRequest):
        """Yield outputs incrementally; video workflows can avoid retaining batches."""
        yield self.generate(request)

    @abstractmethod
    def close(self) -> None:
        """Release the loaded pipeline."""

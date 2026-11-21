"""Declarative minimax integration; imports no model libraries."""

from pathlib import Path

from ..definition import ModeDefinition, WorkflowDefinition
from ..inputs import InputSpec
from ..runtime import RuntimeOption
from .adapters import MiniMaxAdapterPolicy
from .parameters import DEFAULTS, get_ui_definition
from .validation import validate, validate_form_settings

DEFINITION = WorkflowDefinition(
    name="minimax",
    family="minimax",
    runtime=("image_generator.workflows.minimax.runtime", "MiniMaxWorkflow"),
    defaults=DEFAULTS,
    modes=(ModeDefinition("img2vid", "video", (InputSpec("input_image"),)), ModeDefinition("txt2vid", "video")),
    parameters=get_ui_definition,
    validate_settings=validate_form_settings,
    validate_request=validate,
    adapter_policy=MiniMaxAdapterPolicy(),
    default_model_id="model_b065ec3d-cd27-4cb3-bcab-48797bee8b6b",
    asset_root=Path(__file__).parent,
    scripts=("controls.js",),
    group_partials=(),
    extra_runtime_options=(RuntimeOption("comfyui_dir", Path, "ComfyUI checkout with native H3 support"),),
)

"""Declarative wan integration; imports no model libraries."""

from pathlib import Path

from ..definition import ModeDefinition, WorkflowDefinition
from ..inputs import InputSpec
from .adapters import WanAdapterPolicy
from .parameters import DEFAULTS, get_ui_definition
from .validation import validate, validate_form_settings

DEFINITION = WorkflowDefinition(
    name="wan",
    family="wan",
    runtime=("image_generator.workflows.wan.runtime", "WanWorkflow"),
    defaults=DEFAULTS,
    modes=(ModeDefinition("img2vid", "video", (InputSpec("input_image"),)), ModeDefinition("txt2vid", "video")),
    parameters=get_ui_definition,
    validate_settings=validate_form_settings,
    validate_request=validate,
    adapter_policy=WanAdapterPolicy(),
    default_model_id="model_95f8fb96-fc3b-4cf3-8b6f-e95e8cb5209b",
    asset_root=Path(__file__).parent,
    scripts=("controls.js",),
    group_partials=(),
)

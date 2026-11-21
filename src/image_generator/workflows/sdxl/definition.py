"""Declarative sdxl integration; imports no model libraries."""

from pathlib import Path

from ..adapters import AdapterPolicy
from ..definition import ModeDefinition, WorkflowDefinition
from .parameters import DEFAULTS, get_ui_definition
from .validation import validate, validate_form_settings

DEFINITION = WorkflowDefinition(
    name="sdxl",
    family="sdxl",
    runtime=("image_generator.workflows.sdxl.runtime", "SDXLWorkflow"),
    defaults=DEFAULTS,
    modes=(ModeDefinition("txt2img"),),
    parameters=get_ui_definition,
    validate_settings=validate_form_settings,
    validate_request=validate,
    adapter_policy=AdapterPolicy(),
    default_model_id="model_a4278bb0-94f4-4960-ada0-39ca89cec194",
    asset_root=Path(__file__).parent,
    scripts=(),
    group_partials=(),
)

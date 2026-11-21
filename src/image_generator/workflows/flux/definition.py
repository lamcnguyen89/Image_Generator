"""Declarative flux integration; imports no model libraries."""

from pathlib import Path

from ..adapters import AdapterPolicy
from ..definition import ModeDefinition, WorkflowDefinition
from .parameters import DEFAULTS, get_ui_definition
from .validation import validate, validate_form_settings

DEFINITION = WorkflowDefinition(
    name="flux",
    family="flux",
    runtime=("image_generator.workflows.flux.runtime", "FluxWorkflow"),
    defaults=DEFAULTS,
    modes=(ModeDefinition("txt2img"),),
    parameters=get_ui_definition,
    validate_settings=validate_form_settings,
    validate_request=validate,
    adapter_policy=AdapterPolicy(),
    default_model_id="model_6e8d8d55-6a4b-4cf9-b91b-3a4a5beaab1f",
    asset_root=Path(__file__).parent,
    scripts=("controls.js",),
    group_partials=(("Sampling", "controls/flux.html"),),
)

"""Declarative krea2 integration; imports no model libraries."""

from pathlib import Path

from ..adapters import AdapterPolicy
from ..definition import ModeDefinition, WorkflowDefinition
from .parameters import DEFAULTS, get_ui_definition
from .validation import validate, validate_form_settings

DEFINITION = WorkflowDefinition(
    name="krea2",
    family="krea2",
    runtime=("image_generator.workflows.krea2.runtime", "Krea2Workflow"),
    defaults=DEFAULTS,
    modes=(ModeDefinition("txt2img"),),
    parameters=get_ui_definition,
    validate_settings=validate_form_settings,
    validate_request=validate,
    adapter_policy=AdapterPolicy(),
    default_model_id="model_43359779-6bc0-4f45-8b70-39a2edff93fa",
    asset_root=Path(__file__).parent,
    scripts=("controls.js",),
    group_partials=(("Sampling", "controls/krea2.html"),),
)

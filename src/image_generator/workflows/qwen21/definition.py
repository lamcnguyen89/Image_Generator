"""Declarative qwen_2_1 integration; imports no model libraries."""

from pathlib import Path

from ..adapters import AdapterPolicy
from ..definition import ModeDefinition, WorkflowDefinition
from .parameters import DEFAULTS, get_ui_definition
from .validation import validate, validate_form_settings

DEFINITION = WorkflowDefinition(
    name="qwen_2_1",
    family="qwen_2_1",
    runtime=("image_generator.workflows.qwen21.runtime", "Qwen21Workflow"),
    defaults=DEFAULTS,
    modes=(ModeDefinition("txt2img"),),
    parameters=get_ui_definition,
    validate_settings=validate_form_settings,
    validate_request=validate,
    adapter_policy=AdapterPolicy(minimum_strength=None),
    default_model_id="model_73c73d8d-ba6e-462f-b3da-ac775c627979",
    asset_root=Path(__file__).parent,
    scripts=("controls.js",),
    group_partials=(("Sampling", "controls/qwen21.html"),),
)

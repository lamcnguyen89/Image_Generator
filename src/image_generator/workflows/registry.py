"""Explicit lazy registry of trusted workflow definitions, never manifest imports."""

from importlib import import_module

from .definition import WorkflowDefinition

WORKFLOWS = {
    "sdxl": ("image_generator.workflows.sdxl.definition", "DEFINITION"),
    "flux": ("image_generator.workflows.flux.definition", "DEFINITION"),
    "qwen_2_1": ("image_generator.workflows.qwen21.definition", "DEFINITION"),
    "krea2": ("image_generator.workflows.krea2.definition", "DEFINITION"),
    "wan": ("image_generator.workflows.wan.definition", "DEFINITION"),
    "minimax": ("image_generator.workflows.minimax.definition", "DEFINITION"),
}


def get_definition(name: str) -> WorkflowDefinition:
    try:
        entry = WORKFLOWS[name]
    except KeyError:
        raise ValueError(f"Workflow {name!r} is not implemented; available: {', '.join(WORKFLOWS)}") from None
    definition = entry if isinstance(entry, WorkflowDefinition) else getattr(import_module(entry[0]), entry[1])
    if not isinstance(definition, WorkflowDefinition) or definition.name != name:
        raise ValueError(f"Invalid workflow definition for {name!r}")
    return definition


def create_workflow(name: str, **options):
    return get_definition(name).create_runtime(**options)

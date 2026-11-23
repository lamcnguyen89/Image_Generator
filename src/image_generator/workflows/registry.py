"""Explicit lazy registry of trusted workflow definitions, never manifest imports."""

from importlib import import_module
import logging
import sys
from time import perf_counter

from image_generator.diagnostics import event

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
    if isinstance(entry, WorkflowDefinition):
        definition = entry
    else:
        loaded = entry[0] in sys.modules
        started = perf_counter()
        definition = getattr(import_module(entry[0]), entry[1])
        if not loaded:
            event("workflow_definition_loaded", level=logging.DEBUG, workflow=name, module=entry[0],
                  elapsed_seconds=perf_counter() - started)
    if not isinstance(definition, WorkflowDefinition) or definition.name != name:
        raise ValueError(f"Invalid workflow definition for {name!r}")
    return definition


def create_workflow(name: str, **options):
    started = perf_counter()
    event("workflow_create_started", workflow=name, device=options.get("device"), dtype=options.get("dtype"))
    try:
        workflow = get_definition(name).create_runtime(**options)
    except BaseException:
        event("workflow_create_failed", workflow=name, level=logging.ERROR, exc_info=True,
              elapsed_seconds=perf_counter() - started)
        raise
    event("workflow_created", workflow=name, runtime_module=type(workflow).__module__,
          elapsed_seconds=perf_counter() - started)
    return workflow

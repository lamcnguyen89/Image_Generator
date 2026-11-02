"""Explicit, lazy workflow registry. Manifests never supply import paths."""

from importlib import import_module

from .base import Workflow


WORKFLOWS = {
    "sdxl": ("image_generator.workflows.sdxl", "SDXLWorkflow"),
    "flux": ("image_generator.workflows.flux", "FluxWorkflow"),
    "qwen_2_1": ("image_generator.workflows.qwen21", "Qwen21Workflow"),
    "krea2": ("image_generator.workflows.krea2", "Krea2Workflow"),
    "wan": ("image_generator.workflows.wan", "WanWorkflow"),
    "minimax": ("image_generator.workflows.minimax", "MiniMaxWorkflow"),
}


def create_workflow(name: str, **options) -> Workflow:
    try:
        module, class_name = WORKFLOWS[name]
    except KeyError:
        raise ValueError(f"Workflow {name!r} is not implemented; available: {', '.join(WORKFLOWS)}") from None
    return getattr(import_module(module), class_name)(**options)

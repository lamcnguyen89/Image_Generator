"""Lightweight workflow capabilities and schema, separate from execution state."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from typing import Callable

from .adapters import AdapterPolicy
from .inputs import validate_inputs


@dataclass(frozen=True)
class ModeDefinition:
    name: str
    output_kind: str = "image"
    inputs: tuple = ()
    executable: bool = True


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    family: str
    runtime: tuple[str, str]
    defaults: dict
    modes: tuple[ModeDefinition, ...]
    parameters: Callable
    validate_settings: Callable
    validate_request: Callable
    adapter_policy: AdapterPolicy = field(default_factory=AdapterPolicy)
    default_model_id: str | None = None
    asset_root: Path | None = None
    scripts: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    group_partials: tuple[tuple[str, str], ...] = ()
    runtime_options: tuple[str, ...] = ("device", "dtype", "cache_dir", "local_files_only", "token")
    extra_runtime_options: tuple = ()

    @property
    def ui_modes(self):
        return tuple(mode.name for mode in self.modes)

    def mode(self, name):
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise ValueError(f"Workflow {self.name!r} does not support mode {name!r}")

    def check_model(self, model, mode):
        if model.kind != "model" or model.family != self.family or model.defaults["workflow"] != self.name:
            raise ValueError(f"This workflow requires a {self.family.upper()} model selecting {self.name}")
        if mode not in model.defaults["supported_modes"]:
            raise ValueError(f"Workflow {self.name} supports declared modes: {', '.join(self.ui_modes)}")
        return self.mode(mode)

    def get_ui_definition(self, model, mode):
        selected = self.check_model(model, mode)
        ui = self.parameters(model, mode)
        fields = tuple(spec.parameter() for spec in selected.inputs) + tuple(
            parameter for parameter in ui.fields if parameter.scope == "settings"
        )
        return replace(
            ui,
            workflow=self.name,
            fields=fields,
            output_kind=selected.output_kind,
            scripts=self.scripts,
            styles=self.styles,
            group_partials=self.group_partials,
            adapter_groups=self.adapter_policy.groups,
        )

    def validate_form_settings(self, model, mode, settings):
        from .ui import FormValidationError

        self.get_ui_definition(model, mode).validate_settings(settings)
        try:
            self.validate_settings(model, mode, settings)
        except FormValidationError:
            raise
        except ValueError as error:
            raise FormValidationError({"settings": str(error)}) from error

    def validate(self, request):
        selected = self.check_model(request.model, request.mode)
        unknown = request.settings.keys() - self.defaults.keys()
        if unknown:
            raise ValueError(f"Unsupported {self.name} settings: {', '.join(sorted(unknown))}")
        self.validate_request(request)
        self.validate_form_settings(request.model, request.mode, request.settings)
        validate_inputs(selected.inputs, request.inputs)
        for selection in request.loras:
            reason = self.adapter_policy.unavailable(request.model, selection.asset, link=selection.settings)
            if reason:
                raise ValueError(reason)
            self.adapter_policy.validate_link(selection.asset, {**selection.settings, "strength": selection.strength})

    def create_runtime(self, **options):
        module, name = self.runtime
        instance = getattr(import_module(module), name)(**options)
        instance.workflow_id = self.name
        instance.defaults = deepcopy(self.defaults)
        instance.ui_modes = self.ui_modes
        return instance

"""Save/reset model and shared adapter preferences, independently of generation."""

from flask import Blueprint, abort, jsonify, redirect, render_template, request, url_for

from image_generator.catalog.persistence import (
    SettingsConflict, reset_settings, save_adapter_settings, save_model_settings, settings_revision,
    update_examples,
)
from .security import valid_csrf
from .web import get_catalog


settings_bp = Blueprint("settings", __name__)


def _error(error, status=422):
    return render_template("error.html", title="Settings were not saved", message=str(error)), status


@settings_bp.post("/api/assets/<asset_id>/examples")
def examples(asset_id):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Submit an example prompt change as JSON."), 400
    if not valid_csrf(data.get("csrf_token")):
        return jsonify(error="Form expired. Reload the page before saving examples."), 400
    catalog = get_catalog()
    if asset_id not in catalog.models and asset_id not in catalog.loras:
        return jsonify(error="Unknown checkpoint or LoRA."), 404
    try:
        result = update_examples(
            catalog, asset_id, data.get("field"), data.get("action"),
            expected_revision=data.get("settings_revision"), text=data.get("text"), index=data.get("index"),
        )
    except SettingsConflict as error:
        return jsonify(error=str(error)), 409
    except (ValueError, OSError) as error:
        return jsonify(error=str(error)), 422
    return jsonify(result)


@settings_bp.post("/settings/models/<model_id>")
def model_settings(model_id):
    from image_generator.workflows.registry import create_workflow
    from image_generator.workflows.ui import FormValidationError
    from .forms import lora_options, parse_loras, parse_settings

    catalog = get_catalog()
    model = catalog.models.get(model_id)
    if model is None:
        abort(404)
    if not valid_csrf(request.form.get("csrf_token")):
        return _error("Form expired. Reload saved settings before submitting again.", 400)
    action = request.form.get("action")
    if action not in {"save_model", "reset_model"}:
        return _error("Choose save or reset model settings.", 400)
    mode = request.form.get("mode")
    try:
        workflow = create_workflow(model.defaults["workflow"])
        ui = workflow.get_ui_definition(model, mode)
        revision = request.form.get("settings_revision")
        if not revision or revision != settings_revision(catalog):
            raise SettingsConflict("Saved settings changed. Reload saved settings before saving again.")
        if action == "reset_model":
            reset_settings(catalog, model_id, expected_revision=revision)
        else:
            settings = model.resolve_settings(workflow_defaults=workflow.defaults)
            desired, errors = parse_settings(ui, settings, request.form, allow_empty_prompt=True)
            links, lora_errors = parse_loras(lora_options(catalog, model, settings, request.form),
                                             request.form, preserve_disabled=True)
            errors.update(lora_errors)
            if errors:
                # Reuse the form's rendering so invalid values can be corrected
                # without discarding the user's unsaved edits.
                from .generation_routes import configure
                return configure(model_id, settings_errors=errors)
            desired["loras"] = links
            scope = [field.name for field in ui.fields if field.scope == "settings"] + ["loras"]
            save_model_settings(catalog, model_id, desired, scope=scope, mode=mode, expected_revision=revision)
    except SettingsConflict as error:
        return _error(error, 409)
    except FormValidationError as error:
        from .generation_routes import configure
        return configure(model_id, settings_errors=error.errors)
    except (ValueError, OSError) as error:
        return _error(error)
    return redirect(url_for("generation.configure", model_id=model_id, mode=mode,
                            settings_result="model_saved" if action == "save_model" else "model_reset"), code=303)


@settings_bp.post("/settings/adapters/<adapter_id>")
def adapter_settings(adapter_id):
    catalog = get_catalog()
    asset = catalog.loras.get(adapter_id)
    if asset is None:
        abort(404)
    if not valid_csrf(request.form.get("csrf_token")):
        return _error("Form expired. Reload saved settings before submitting again.", 400)
    model_id = request.form.get("model_id")
    if model_id not in catalog.models:
        abort(404)
    action = request.form.get("action")
    try:
        if action == "reset_adapter":
            reset_settings(catalog, adapter_id, expected_revision=request.form.get("settings_revision"))
        elif action == "save_adapter":
            desired = {"strength": float(request.form.get(f"adapter_strength:{adapter_id}", "")),
                       "trigger_words": [word.strip() for word in request.form.get(f"adapter_trigger_words:{adapter_id}", "").splitlines() if word.strip()]}
            save_adapter_settings(catalog, adapter_id, desired, expected_revision=request.form.get("settings_revision"))
        else:
            return _error("Choose save or reset adapter defaults.", 400)
    except SettingsConflict as error:
        return _error(error, 409)
    except (ValueError, OSError) as error:
        return _error(error)
    return redirect(url_for("generation.configure", model_id=model_id, mode=request.form.get("mode"),
                            settings_result="adapter_saved" if action == "save_adapter" else "adapter_reset"), code=303)

"""Generation forms, previews, and image/video queue submission; no GPU execution."""

from contextlib import nullcontext
from dataclasses import replace
import re
import sqlite3
from uuid import uuid4

from flask import Blueprint, abort, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from .web import get_catalog
from .security import csrf_token, valid_csrf


generation_bp = Blueprint("generation", __name__)


@generation_bp.app_errorhandler(413)
def upload_too_large(error):
    return render_template("error.html", title="Upload too large",
                           message="The request exceeds the upload size limit. Return to the form and choose a smaller source image."), 413


@generation_bp.route("/generate/<model_id>", methods=["GET", "POST"])
def configure(model_id, settings_errors=None):
    # Catalog browsing keeps workflow imports lazy; these modules are CPU-only.
    from image_generator.workflows.registry import get_definition
    from image_generator.workflows.ui import FormValidationError
    from .forms import display_values, lora_options, parse_loras, parse_settings
    from .generation_groups import generation_targets, generation_title

    catalog = get_catalog()
    try:
        model = catalog.get_model(model_id)
    except KeyError:
        abort(404)
    try:
        workflow = get_definition(model.defaults["workflow"])
        targets = generation_targets(catalog, model, workflow.ui_modes)
        modes = tuple(targets)
        if not modes:
            raise ValueError("This model has no modes with an implemented generation form.")
        default_mode = next((mode for mode in modes if mode in model.defaults["supported_modes"]), modes[0])
        mode = request.form.get("mode", default_mode) if request.method == "POST" else request.args.get("mode", default_mode)
        if mode not in modes:
            return render_template("error.html", title="Unsupported mode",
                                   message="Choose a mode supported by this model and workflow."), 400
        model = targets[mode]
        ui = replace(workflow.get_ui_definition(model, mode), supported_modes=modes)
    except ValueError as error:
        return render_template("error.html", title="Generation form unavailable", message=str(error)), 422

    settings = model.resolve_settings(workflow_defaults=workflow.defaults)
    form = request.form if request.method == "POST" else None
    changed_checkpoint = (form is not None and form.get("action") == "change_mode"
                          and form.get("model_id", model_id) != model.id)
    values = display_values(ui, settings, None if changed_checkpoint else form)
    if changed_checkpoint:
        # Each checkpoint has its own numerical defaults and compatible adapters.
        values["prompt"] = form.get("prompt", values["prompt"])
    elif form is not None and form.get("action") == "change_mode":
        # Workflow-declared controls may appear only after a mode change.
        # Other absent checkboxes represent unchecked controls in the previous form.
        inherited = display_values(ui, settings)
        for parameter in ui.fields:
            if parameter.inherit_when_absent and parameter.name not in form:
                values[parameter.name] = inherited[parameter.name]
    options = lora_options(catalog, model, settings, None if changed_checkpoint else form)
    errors, preview = {}, None
    status_code = 200
    action = form.get("action", "preview") if form is not None else "preview"
    submission_key = form.get("submission_key") if form is not None else None
    submission_key = submission_key or uuid4().hex
    mode_definition = workflow.mode(mode)
    can_generate = mode_definition.executable
    if settings_errors is not None:
        action = "preview"
    if action not in {"preview", "generate", "change_mode"}:
        return render_template("error.html", title="Unknown action", message="Choose preview, generation, or a mode change."), 400
    if action == "generate":
        if not can_generate:
            return render_template("error.html", title="Generation unavailable", message="This model and mode do not support web submission."), 400
        if not valid_csrf(form.get("csrf_token")):
            return render_template("error.html", title="Form expired", message="Reload the generation form before submitting again."), 400
        if not form.get("submission_key") or not re.fullmatch(r"[0-9a-f]{32}", submission_key):
            return render_template("error.html", title="Invalid submission", message="Reload the generation form before submitting again."), 400
    include_triggers = form is not None and form.get("include_trigger_words") == "on"
    if settings_errors is not None:
        errors.update(settings_errors)
    elif form is not None and action != "change_mode":
        submitted, errors = parse_settings(ui, settings, form)
        links, lora_errors = parse_loras(options, form)
        errors.update(lora_errors)
        submitted["loras"] = links
        if not errors:
            try:
                workflow.validate_form_settings(model, mode, submitted)
            except FormValidationError as error:
                errors.update(error.errors)
        input_files, validated_sources = {}, {}
        declared_inputs = {spec.name: spec for spec in mode_definition.inputs}
        for name, upload in request.files.items():
            if upload.filename and name not in declared_inputs:
                errors[name] = "This mode does not accept this source file."
        for field in ui.fields:
            if field.scope == "inputs":
                upload = request.files.get(field.name)
                if upload is None or not upload.filename:
                    if field.required:
                        errors[field.name] = "Select a source image."
                else:
                    input_files[field.name] = secure_filename(upload.filename) or "selected-image"
                    if can_generate:
                        from .uploads import validate_upload

                        try:
                            validated_sources[field.name] = validate_upload(declared_inputs[field.name], upload)
                        except ValueError as error:
                            errors[field.name] = str(error)
        if not errors:
            preview = {"model_id": model.id, "mode": mode, "overrides": submitted,
                       "include_trigger_words": include_triggers, "input_files": input_files}
            if action == "generate":
                from image_generator.generation import prepare_generation
                from image_generator.jobs import SubmissionConflict
                from .job_routes import get_job_store
                from .uploads import stage_sources

                try:
                    source_context = stage_sources(submission_key, validated_sources) if validated_sources else nullcontext()
                    with source_context as source:
                        prepared = prepare_generation(catalog, model.id, mode=mode, overrides=submitted,
                                                      include_trigger_words=include_triggers,
                                                      inputs=source.paths if source else None)
                        job = get_job_store().enqueue(prepared, submission_key=submission_key)
                        if source:
                            source.retain()
                except SubmissionConflict as error:
                    preview = None
                    errors["generation"] = str(error)
                    status_code = 409
                except sqlite3.Error:
                    preview = None
                    errors["generation"] = "The generation queue is unavailable. Check database access and try again."
                    status_code = 503
                except (ValueError, OSError) as error:
                    preview = None
                    errors["generation"] = str(error)
                else:
                    return redirect(url_for("jobs.detail", job_id=job.id), code=303)

    from image_generator.catalog.persistence import settings_revision

    notices = {"model_saved": "Model settings saved.", "model_reset": "Model settings reset to inherited defaults.",
               "adapter_saved": "Shared adapter defaults saved.", "adapter_reset": "Shared adapter defaults reset."}
    return render_template(
        "generate.html", model=model, ui=ui, values=values, loras=options,
        errors=errors, preview=preview, include_triggers=include_triggers,
        can_generate=can_generate, csrf_token=csrf_token(), submission_key=submission_key,
        generation_title=generation_title(model), mode_changes_checkpoint=len({item.id for item in targets.values()}) > 1,
        settings_revision=form.get("settings_revision", "") if form is not None else settings_revision(catalog),
        settings_notice=notices.get(request.args.get("settings_result")),
    ), (status_code if status_code != 200 else 422) if errors else 200

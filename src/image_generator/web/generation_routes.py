"""Generation forms, previews, and image/video queue submission; no GPU execution."""

from contextlib import nullcontext
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
    from image_generator.workflows.registry import create_workflow
    from image_generator.workflows.ui import FormValidationError
    from .forms import display_values, lora_options, parse_loras, parse_settings

    catalog = get_catalog()
    try:
        model = catalog.get_model(model_id)
    except KeyError:
        abort(404)
    try:
        workflow = create_workflow(model.defaults["workflow"])
        modes = tuple(mode for mode in workflow.ui_modes if mode in model.defaults["supported_modes"])
        if not modes:
            raise ValueError("This model has no modes with an implemented generation form.")
        mode = request.form.get("mode", modes[0]) if request.method == "POST" else request.args.get("mode", modes[0])
        if mode not in modes:
            return render_template("error.html", title="Unsupported mode",
                                   message="Choose a mode supported by this model and workflow."), 400
        ui = workflow.get_ui_definition(model, mode)
    except ValueError as error:
        return render_template("error.html", title="Generation form unavailable", message=str(error)), 422

    settings = model.resolve_settings(workflow_defaults=workflow.defaults)
    form = request.form if request.method == "POST" else None
    values = display_values(ui, settings, form)
    options = lora_options(catalog, model, settings, form)
    errors, preview = {}, None
    status_code = 200
    action = form.get("action", "preview") if form is not None else "preview"
    submission_key = form.get("submission_key") if form is not None else None
    submission_key = submission_key or uuid4().hex
    can_generate = (model.defaults["workflow"], mode) in {
        ("sdxl", "txt2img"), ("flux", "txt2img"), ("krea2", "txt2img"),
        ("qwen_2_1", "txt2img"),
        ("wan", "img2vid"), ("minimax", "img2vid"),
    }
    if settings_errors is not None:
        action = "preview"
    if action not in {"preview", "generate"}:
        return render_template("error.html", title="Unknown action", message="Choose preview or generation."), 400
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
    elif form is not None:
        submitted, errors = parse_settings(ui, settings, form)
        links, lora_errors = parse_loras(options, form)
        errors.update(lora_errors)
        submitted["loras"] = links
        if not errors:
            try:
                workflow.validate_form_settings(model, mode, submitted)
            except FormValidationError as error:
                errors.update(error.errors)
        input_files, validated_source = {}, None
        for field in ui.fields:
            if field.scope == "inputs":
                upload = request.files.get(field.name)
                if upload is None or not upload.filename:
                    if field.required:
                        errors[field.name] = "Select a source image."
                else:
                    input_files[field.name] = secure_filename(upload.filename) or "selected-image"
                    if can_generate and mode == "img2vid":
                        from .uploads import validate_image_upload

                        try:
                            validated_source = validate_image_upload(upload)
                        except ValueError as error:
                            errors[field.name] = str(error)
        if not errors:
            preview = {"model_id": model.id, "mode": mode, "overrides": submitted,
                       "include_trigger_words": include_triggers, "input_files": input_files}
            if action == "generate":
                from image_generator.generation import prepare_generation
                from image_generator.jobs import SubmissionConflict
                from .job_routes import get_job_store
                from .uploads import stage_source

                try:
                    source_context = stage_source(submission_key, validated_source) if validated_source else nullcontext()
                    with source_context as source:
                        prepared = prepare_generation(catalog, model.id, mode=mode, overrides=submitted,
                                                      include_trigger_words=include_triggers,
                                                      inputs={"input_image": source.path} if source else None)
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
        settings_revision=form.get("settings_revision", "") if form is not None else settings_revision(catalog),
        settings_notice=notices.get(request.args.get("settings_result")),
    ), (status_code if status_code != 200 else 422) if errors else 200

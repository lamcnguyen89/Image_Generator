"""Persistent job pages, polling, and registered output serving; no GPU imports."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, send_file, url_for

from image_generator.jobs import JobDeletionConflict, JobStore, TERMINAL_STATES

from .job_timing import job_timing
from .security import csrf_token, valid_csrf


jobs_bp = Blueprint("jobs", __name__)
TERMINAL = TERMINAL_STATES
STAGE_LABELS = {
    "queued": "Waiting in queue", "validating": "Checking job assets", "preparing": "Preparing workflow",
    "generating": "Generating", "loading_runtime": "Preparing GPU runtime",
    "loading_model": "Loading checkpoint", "switching_model": "Switching checkpoint",
    "reusing_model": "Reusing loaded checkpoint", "resetting_loras": "Clearing previous LoRAs",
    "resolving_text_encoder": "Resolving text encoder", "loading_text_encoder": "Loading text encoder",
    "loading_vae": "Loading video VAE", "encoding_prompt": "Conditioning source and encoding prompt",
    "reusing_loras": "Reusing configured LoRAs",
    "loading_loras": "Loading selected LoRAs", "loras_ready": "LoRAs configured",
    "denoising": "Denoising", "decoding": "Decoding images", "saving": "Saving outputs",
    "conditioning_image": "Preparing conditioning image", "video_variation": "Generating video variation",
    "decoding_video": "Decoding video", "encoding_video": "Encoding and saving video",
    "completed": "Completed", "failed": "Failed", "interrupted": "Interrupted",
}


@jobs_bp.app_errorhandler(sqlite3.Error)
def unavailable_queue(error):
    message = "The generation queue is unavailable. Check database access and try again."
    if request.path.startswith("/api/jobs/"):
        return jsonify({"error": message}), 503
    return render_template("error.html", title="Queue unavailable", message=message), 503


def get_job_store():
    path = Path(current_app.config["JOB_DATABASE"]).expanduser().resolve()
    store = current_app.extensions.get("generation_job_store")
    if store is None or store.database != path:
        store = JobStore(path)
        current_app.extensions["generation_job_store"] = store
    return store


def get_job(job_id):
    try:
        return get_job_store().get(job_id)
    except KeyError:
        abort(404)


def stage_label(stage):
    return STAGE_LABELS.get(stage, stage.replace("_", " ").capitalize())


def progress_text(stage, current, total):
    if current is None:
        return ""
    if stage == "denoising" and total:
        return f"Step {current} of {total}"
    if stage == "video_variation":
        return f"Video {current} of {total}"
    if stage in {"loading_loras", "loras_ready", "reusing_loras"}:
        return f"{current} of {total} adapters" if total else "No adapters selected"
    if stage in {"saving", "encoding_video", "completed", "failed", "interrupted"}:
        return f"{current} output(s) saved"
    return ""


def worker_notice(store, job):
    if job.status not in {"queued", "running"}:
        return ""
    worker = store.worker_status()
    if worker is None or worker["status"] != "running":
        return "The worker is stopped. This job will wait until a worker starts."
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(worker["heartbeat_at"])).total_seconds()
    if age > 15:
        return "The worker heartbeat is stale. An interrupted job will be recovered when the worker restarts."
    return ""


def job_payload(job):
    settings = job.request["settings"]
    model = job.request["model"]["defaults"]
    width, height = settings.get("width"), settings.get("height")
    dimensions = {"width": width, "height": height} if (
        type(width) is int and type(height) is int and width > 0 and height > 0
    ) else {}

    def output_payload(output):
        details = {}
        if output.kind == "video":
            path = output_path(job, output.metadata_path)
            if path:
                try:
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    details = metadata.get("output_video", {})
                    if not details:
                        generated = metadata.get("generation_settings", {})
                        details = {"width": generated.get("width"), "height": generated.get("height"),
                                   "frame_count": metadata.get("frame_count"),
                                   "fps": metadata.get("settings", {}).get("fps"),
                                   "duration_seconds": metadata.get("duration_seconds")}
                    details = {**details, "audio_exported": metadata.get("audio_exported")}
                except (ValueError, OSError):
                    pass
        return {
            "id": output.id, "kind": output.kind, "seed": str(output.seed),
            **(details if output.kind == "video" else dimensions),
            "url": url_for("jobs.output", job_id=job.id, output_id=output.id, part="media"),
            "download_url": url_for("jobs.output", job_id=job.id, output_id=output.id, part="media", download=1),
            "metadata_url": url_for("jobs.output", job_id=job.id, output_id=output.id, part="metadata", download=1),
            "conditioning_url": url_for("jobs.output", job_id=job.id, output_id=output.id, part="conditioning")
                                if output.conditioning_path else None,
        }
    from .uploads import managed_source_path

    video_count = settings.get("num_videos_per_prompt")
    variation = next((event for event in reversed(job.events) if event.stage == "video_variation"), None)
    completion_text = f"{len(job.outputs)} of {video_count} videos saved" if video_count else ""
    if variation and job.status == "running":
        completion_text += f" · Video {variation.current} of {variation.total}"
    notice = worker_notice(get_job_store(), job)
    return {
        "id": job.id, "model": {"id": model["id"], "name": model["display_name"]},
        "workflow": job.workflow, "mode": job.request["mode"], "status": job.status,
        "terminal": job.status in TERMINAL, "error": job.error,
        "progress": {"stage": job.stage, "label": stage_label(job.stage), "current": job.current, "total": job.total,
                     "text": progress_text(job.stage, job.current, job.total),
                     "completion_text": completion_text},
        "source_url": url_for("jobs.source", job_id=job.id) if managed_source_path(job) else None,
        "worker_notice": notice,
        "timing": job_timing(job, worker_available=not notice),
        "events": [{"stage": event.stage, "label": stage_label(event.stage),
                    "current": event.current, "total": event.total,
                    "text": progress_text(event.stage, event.current, event.total)} for event in job.events],
        "outputs": [output_payload(output) for output in job.outputs],
        "created_at": job.created_at, "started_at": job.started_at, "finished_at": job.finished_at,
        "image_count": settings.get("num_images_per_prompt"),
        "video_count": video_count,
    }


def output_path(job, path):
    """Constrain reads, including metadata reads, to this job's output directory."""
    if path is None:
        return None
    path = path.resolve()
    roots = [current_app.config["JOB_OUTPUT_ROOT"]]
    if job.workflow in {"minimax", "wan"}:
        # Keep previously registered videos in Images accessible as well.
        roots.insert(0, current_app.config["JOB_VIDEO_OUTPUT_ROOT"])
    for root in roots:
        root = Path(root).expanduser().resolve()
        directory = (root / job.id).resolve()
        if directory.is_relative_to(root) and path.is_relative_to(directory) and path.is_file():
            return path
    return None


@jobs_bp.get("/jobs/<job_id>/source")
def source(job_id):
    from .uploads import managed_source_path

    path = managed_source_path(get_job(job_id))
    if path is None:
        abort(404)
    response = send_file(path, conditional=True, max_age=0,
                         as_attachment=request.args.get("download") == "1", download_name=path.name)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@jobs_bp.get("/jobs")
def index():
    return render_template("jobs.html", jobs=get_job_store().list_jobs(limit=50), stage_label=stage_label,
                           terminal_states=TERMINAL, csrf_token=csrf_token())


@jobs_bp.post("/jobs/<job_id>/delete")
def delete(job_id):
    if not valid_csrf(request.form.get("csrf_token")):
        return render_template("error.html", title="Form expired",
                               message="Reload the jobs page before deleting a job."), 400
    from image_generator.job_cleanup import delete_job

    try:
        delete_job(get_job_store(), job_id, output_roots=(
            current_app.config["JOB_OUTPUT_ROOT"], current_app.config["JOB_VIDEO_OUTPUT_ROOT"],
        ), input_root=current_app.config["JOB_INPUT_ROOT"])
    except KeyError:
        abort(404)
    except JobDeletionConflict as error:
        return render_template("error.html", title="Job is still active", message=str(error)), 409
    except (OSError, ValueError):
        return render_template("error.html", title="Job could not be deleted",
                               message="File cleanup could not finish. The job is still in history; check file permissions and try again."), 503
    return redirect(url_for("jobs.index", deleted=1), code=303)


@jobs_bp.get("/jobs/<job_id>")
def detail(job_id):
    job = get_job(job_id)
    return render_template("job.html", job=job, status=job_payload(job))


@jobs_bp.get("/api/jobs/<job_id>")
def status(job_id):
    response = jsonify(job_payload(get_job(job_id)))
    response.cache_control.no_store = True
    return response


@jobs_bp.get("/jobs/<job_id>/outputs/<output_id>/<part>")
def output(job_id, output_id, part):
    job = get_job(job_id)
    record = next((item for item in job.outputs if item.id == output_id), None)
    if record is None or part not in {"media", "metadata", "conditioning"}:
        abort(404)
    path = {"media": record.file_path, "metadata": record.metadata_path, "conditioning": record.conditioning_path}[part]
    path = output_path(job, path)
    if path is None:
        abort(404)
    mimetype = "application/json" if part == "metadata" else (
        "video/mp4" if part == "media" and record.kind == "video" else "image/png")
    response = send_file(path, mimetype=mimetype, as_attachment=request.args.get("download") == "1",
                         download_name=path.name, conditional=True, max_age=0)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

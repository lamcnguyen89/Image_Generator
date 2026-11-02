"""Finished-job deletion removes owned files and preserves active jobs and assets."""

from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.job_cleanup import delete_job
from image_generator.jobs import JobDeletionConflict, JobStore
from image_generator.worker import Worker
from image_generator.workflows.base import SavedImage, SavedVideo
from queue_fakes import FakeWorkflow, make_request


class HistoryForms(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.forms, self.current = {}, None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.current = attrs.get("action")
            self.forms[self.current] = {"data": {}, "disabled": False}
        elif self.current and tag == "input":
            self.forms[self.current]["data"][attrs["name"]] = attrs.get("value", "")
        elif self.current and tag == "button":
            self.forms[self.current]["disabled"] = "disabled" in attrs

    def handle_endtag(self, tag):
        if tag == "form":
            self.current = None


class JobDeletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images, self.videos, self.inputs = (self.root / name for name in ("Images", "Videos", "inputs"))
        self.catalog = self.root / "families"
        self.catalog.mkdir()
        self.app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "SECRET_KEY": "delete-tests",
                               "CATALOG_ROOT": self.catalog, "JOB_DATABASE": self.root / "jobs.sqlite3",
                               "JOB_OUTPUT_ROOT": self.images, "JOB_VIDEO_OUTPUT_ROOT": self.videos,
                               "JOB_INPUT_ROOT": self.inputs})
        self.client = self.app.test_client()
        self.store = JobStore(self.app.config["JOB_DATABASE"])
        self.store._start_worker("test-worker", 1)

    def create(self, *, status="completed", video=False, source=False, outside_source=False):
        key = uuid4().hex
        request = make_request(self.root)
        if source:
            directory = self.root / "originals" if outside_source else self.inputs / key
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "source.png"
            path.write_bytes(b"original source")
            request = replace(request, inputs={"input_image": path})
        job = self.store.enqueue(request, submission_key=key)
        if status == "queued":
            return job
        self.assertEqual(self.store._claim_next("test-worker").id, job.id)
        if status == "running":
            return self.store.get(job.id)
        directory = (self.videos if video else self.images) / job.id
        directory.mkdir(parents=True)
        media, metadata = directory / ("output.mp4" if video else "output.png"), directory / "output.json"
        media.write_bytes(b"generated media")
        metadata.write_text("{}")
        if video:
            conditioning = directory / "conditioning.png"
            conditioning.write_bytes(b"conditioning")
            output = SavedVideo(media, metadata, 42, conditioning)
        else:
            output = SavedImage(media, metadata, 42)
        self.store._record_outputs(job.id, "test-worker", [output])
        self.store._finish(job.id, "test-worker", status)
        return self.store.get(job.id)

    def form(self, job):
        response = self.client.get("/jobs")
        self.assertEqual(response.status_code, 200)
        action = f"/jobs/{job.id}/delete"
        return HistoryForms(response.get_data(as_text=True)).forms[action]

    def delete(self, job, **changes):
        form = self.form(job)
        return self.client.post(f"/jobs/{job.id}/delete", data={**form["data"], **changes})

    def cleanup(self, job):
        delete_job(self.store, job.id, output_roots=(self.images, self.videos), input_root=self.inputs)

    def test_history_has_rightmost_delete_buttons_and_disables_active_jobs(self):
        completed, queued = self.create(), self.create(status="queued")
        self.assertFalse(self.form(completed)["disabled"])
        self.assertTrue(self.form(queued)["disabled"])
        html = self.client.get("/jobs").get_data(as_text=True)
        self.assertIn('<th scope="col" class="job-actions">Actions</th>', html)
        self.assertIn('class="delete-job"', html)
        self.assertIn('jobs.css', html)

    def test_delete_completed_image_removes_files_rows_and_downloads_without_touching_other_jobs(self):
        job, other = self.create(source=True), self.create()
        source = Path(job.request["inputs"]["input_image"])
        output = job.outputs[0]
        # Partial or unregistered artifacts belong to this job as well.
        (output.file_path.parent / "partial.tmp").write_bytes(b"partial")
        self.catalog.joinpath("keep.json").write_text("{}")
        response = self.delete(job)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.location, "/jobs?deleted=1")
        self.assertFalse(output.file_path.parent.exists())
        self.assertFalse(source.parent.exists())
        self.assertTrue(other.outputs[0].file_path.exists())
        self.assertTrue(self.catalog.joinpath("keep.json").exists())
        with self.assertRaises(KeyError):
            self.store.get(job.id)
        for table in ("jobs", "job_outputs", "job_events", "job_submissions"):
            field = "id" if table == "jobs" else "job_id"
            with sqlite3.connect(self.store.database) as connection:
                self.assertEqual(connection.execute(f"SELECT count(*) FROM {table} WHERE {field}=?", (job.id,)).fetchone()[0], 0)
        for url in (f"/jobs/{job.id}", f"/api/jobs/{job.id}", f"/jobs/{job.id}/source",
                    f"/jobs/{job.id}/outputs/{output.id}/media", f"/jobs/{job.id}/outputs/{output.id}/metadata"):
            self.assertEqual(self.client.get(url).status_code, 404)
        self.assertIn("Job and saved files deleted.", self.client.get(response.location).get_data(as_text=True))

    def test_video_deletion_cleans_video_conditioning_metadata_and_legacy_image_directory(self):
        job = self.create(video=True)
        legacy = self.images / job.id
        legacy.mkdir(parents=True)
        (legacy / "legacy.mp4").write_bytes(b"old video")
        self.assertEqual(self.delete(job).status_code, 303)
        self.assertFalse((self.videos / job.id).exists())
        self.assertFalse(legacy.exists())

    def test_failed_and_interrupted_jobs_can_be_deleted_with_partial_outputs(self):
        for status in ("failed", "interrupted"):
            with self.subTest(status=status):
                job = self.create(status=status)
                self.assertEqual(self.delete(job).status_code, 303)
                self.assertFalse(job.outputs[0].file_path.exists())

    def test_active_jobs_cannot_be_deleted_even_with_a_forged_post(self):
        for status in ("queued", "running"):
            with self.subTest(status=status):
                job = self.create(status=status)
                response = self.delete(job)
                self.assertEqual(response.status_code, 409)
                self.assertEqual(self.store.get(job.id).status, status)
                with self.assertRaises(JobDeletionConflict):
                    self.store.delete_finished(job.id)
                if status == "queued":
                    self.store._claim_next("test-worker")
                    self.store._finish(job.id, "test-worker", "failed")

    def test_delete_requires_post_same_session_and_valid_csrf(self):
        job = self.create()
        self.assertEqual(self.client.get(f"/jobs/{job.id}/delete").status_code, 405)
        for token in ("", "bad"):
            self.assertEqual(self.delete(job, csrf_token=token).status_code, 400)
        data = self.form(job)["data"]
        self.assertEqual(self.app.test_client().post(f"/jobs/{job.id}/delete", data=data).status_code, 400)
        self.assertEqual(self.store.get(job.id).status, "completed")
        self.assertTrue(job.outputs[0].file_path.exists())

    def test_missing_files_are_tolerated_and_unknown_job_is_404(self):
        job = self.create()
        shutil.rmtree(job.outputs[0].file_path.parent)
        data = self.form(job)["data"]
        self.assertEqual(self.client.post(f"/jobs/{job.id}/delete", data=data).status_code, 303)
        self.assertEqual(self.client.post(f"/jobs/{job.id}/delete", data=data).status_code, 404)

    def test_external_original_inputs_are_preserved(self):
        job = self.create(source=True, outside_source=True)
        original = Path(job.request["inputs"]["input_image"])
        self.cleanup(job)
        self.assertEqual(original.read_bytes(), b"original source")

    def test_symlinks_in_outputs_and_replaced_output_directory_never_delete_targets(self):
        job = self.create()
        outside = self.root / "outside"
        outside.mkdir()
        keep = outside / "keep.txt"
        keep.write_text("keep")
        (job.outputs[0].file_path.parent / "linked-dir").symlink_to(outside, target_is_directory=True)
        (job.outputs[0].file_path.parent / "linked-file").symlink_to(keep)
        self.cleanup(job)
        self.assertEqual(keep.read_text(), "keep")
        job = self.create()
        directory = job.outputs[0].file_path.parent
        shutil.rmtree(directory)
        directory.symlink_to(outside, target_is_directory=True)
        self.cleanup(job)
        self.assertFalse(directory.is_symlink())
        self.assertEqual(keep.read_text(), "keep")

    def test_replaced_input_directory_and_unowned_input_never_delete_another_source(self):
        job = self.create(source=True)
        source = Path(job.request["inputs"]["input_image"])
        outside = self.root / "outside-input"
        outside.mkdir()
        (outside / "source.png").write_bytes(b"keep")
        source.unlink()
        source.parent.rmdir()
        source.parent.symlink_to(outside, target_is_directory=True)
        self.cleanup(job)
        self.assertEqual((outside / "source.png").read_bytes(), b"keep")
        other = self.create(source=True)
        job = self.create()
        # Corrupt queued metadata must not give deletion ownership of another
        # job's private input directory.
        with sqlite3.connect(self.store.database) as connection:
            request = job.request
            request["inputs"] = other.request["inputs"]
            connection.execute("UPDATE jobs SET request_json=? WHERE id=?", (json.dumps(request), job.id))
        self.cleanup(job)
        self.assertTrue(Path(other.request["inputs"]["input_image"]).exists())

    def test_cleanup_failure_keeps_job_history_for_retry(self):
        job = self.create()
        with patch("image_generator.job_cleanup.shutil.rmtree", side_effect=PermissionError("denied")):
            response = self.delete(job)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.store.get(job.id).status, "completed")
        self.assertTrue(job.outputs[0].file_path.exists())
        self.assertEqual(self.delete(job).status_code, 303)

    def test_deleting_a_just_finished_job_does_not_stop_the_worker_or_next_job(self):
        first, second = self.create(status="queued"), self.create(status="queued")

        def remove_finished():
            try:
                job = self.store.get(first.id)
            except KeyError:
                return
            if job.status == "completed":
                self.cleanup(job)

        with Worker(self.store, self.images, video_output_dir=self.videos,
                    workflow_factory=lambda name, **options: FakeWorkflow()) as worker:
            with patch.object(worker, "_process_model_unloads", side_effect=remove_finished):
                result = worker.run_once()
                self.assertEqual(result.status, "completed")
                with self.assertRaises(KeyError):
                    self.store.get(first.id)
                self.assertEqual(worker.run_once().id, second.id)
        self.assertEqual(self.store.get(second.id).status, "completed")


if __name__ == "__main__":
    unittest.main()

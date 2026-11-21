"""Multipart WAN jobs through the real worker, with synthetic frames and MP4s."""

from contextlib import nullcontext
from io import BytesIO
from itertools import count
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.catalog import discover_catalog
from image_generator.jobs import JobStore
from image_generator.worker import Worker
from test_generation_ui import FormPage
from test_transformer_workflows import Generator, Pipeline


class VideoPipeline(Pipeline):
    def __init__(self, owner, vae):
        super().__init__("wan", vae)
        self.owner, self.adapters = owner, {False: [], True: []}
        self.load_lora_weights.side_effect = lambda path, **options: self.adapters[options["load_into_transformer_2"]].append(options["adapter_name"])
        self.unload_lora_weights.side_effect = lambda: [values.clear() for values in self.adapters.values()]

    def result(self, **arguments):
        self.owner.calls.append((self, {key: tuple(values) for key, values in self.adapters.items()}))
        if self.owner.fail_on_call == len(self.owner.calls):
            raise RuntimeError("synthetic WAN inference failure")
        self.num_timesteps = arguments["num_inference_steps"]
        for step in range(self.num_timesteps):
            tensors = {"latents": object()}
            self.owner.assertIs(arguments["callback_on_step_end"](self, step, step, tensors), tensors)
            if self.owner.on_step:
                self.owner.on_step(step)
        return super().result(**arguments)


class WanExecutionFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.catalog)
        for asset in discover_catalog(self.catalog).loras.values():
            if asset.family == "wan":
                asset.asset_paths["weights"].write_bytes(b"synthetic adapter")
        self.inputs, self.outputs, self.database = self.root / "inputs", self.root / "outputs", self.root / "jobs.sqlite3"
        self.videos = self.root / "Videos"
        self.config = {"TESTING": True, "CATALOG_ROOT": self.catalog, "JOB_DATABASE": self.database,
                       "JOB_INPUT_ROOT": self.inputs, "JOB_OUTPUT_ROOT": self.outputs,
                       "JOB_VIDEO_OUTPUT_ROOT": self.videos, "SECRET_KEY": "wan-tests"}
        self.app = create_app(self.config)
        self.client, self.store = self.app.test_client(), JobStore(self.database)
        source = BytesIO()
        Image.new("RGB", (300, 180), "navy").save(source, format="PNG")
        self.source = source.getvalue()
        self.pipelines, self.calls, self.on_step, self.fail_on_call = [], [], None, None
        def load(*args, **options):
            pipe = VideoPipeline(self, options["vae"])
            self.pipelines.append(pipe)
            return pipe
        self.loader = Mock(side_effect=load)
        torch = SimpleNamespace(
            __version__="test", float32="fp32", bfloat16="bf16", Generator=Generator,
            inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: True,
                is_bf16_supported=lambda: True, device=lambda device: nullcontext(),
                synchronize=Mock(), empty_cache=Mock()))
        diffusers = SimpleNamespace(__version__="test", WanImageToVideoPipeline=SimpleNamespace(from_pretrained=self.loader),
            AutoencoderKLWan=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                dtype="fp32", enable_tiling=Mock(), disable_tiling=Mock())))
        self.diffusers = diffusers
        runtime = patch("image_generator.workflows.shared.diffusers_runtime.import_module", side_effect={"torch": torch, "diffusers": diffusers}.__getitem__)
        self.runtime = runtime.start()
        self.addCleanup(runtime.stop)

    def data(self, model="fixture-wan", **values):
        page = FormPage(self.client.get(f"/generate/{model}").get_data(as_text=True))
        data = page.data()
        for name, value in {"action": "generate", "prompt": "A moving lake", "width": "64", "height": "32",
                            "num_frames": "5", "num_inference_steps": "4", "seed": "42", **values}.items():
            data[name] = value
        data["input_image"] = (BytesIO(self.source), "../../untrusted.html")
        return data

    def post(self, data, model="fixture-wan", payload=None):
        # Werkzeug closes uploaded streams after each HTTP request.
        data["input_image"] = (BytesIO(self.source if payload is None else payload), "../../untrusted.html")
        return self.client.post(f"/generate/{model}", data=data)

    def submit(self, model="fixture-wan", **values):
        response = self.post(self.data(model, **values), model)
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True))
        return response.location.rsplit("/", 1)[1]

    def worker(self, **options):
        options.setdefault("video_output_dir", self.videos)
        return Worker(self.store, self.outputs, **options)

    def staged_files(self):
        return list(self.inputs.glob("*/source.*"))


class WanExecutionTests(WanExecutionFixture, unittest.TestCase):
    def test_upload_is_immutable_original_bytes_and_survives_web_restart(self):
        identifier = self.submit()
        self.runtime.assert_not_called()
        job = self.store.get(identifier)
        path = Path(job.request["inputs"]["input_image"])
        self.assertEqual(path.read_bytes(), self.source)
        self.assertEqual(path.name, "source.png")
        self.assertEqual(path.stat().st_mode & 0o222, 0)
        self.assertIn("sha256", job.fingerprints[str(path)])
        client = create_app(self.config).test_client()
        source = client.get(f"/jobs/{identifier}/source")
        self.assertEqual(source.data, self.source)
        self.assertEqual(source.mimetype, "image/png")
        source.close()
        self.worker().run(once=True)
        self.assertEqual(self.store.get(identifier).status, "completed")

    def test_retries_reuse_paths_and_fingerprints_and_reject_conflicts(self):
        data = self.data()
        first = self.post(data)
        path = self.staged_files()[0]
        original_stat = path.stat()
        self.assertEqual(self.post(data).location, first.location)
        data["seed"] = "99"
        self.assertEqual(self.post(data).status_code, 409)
        other = BytesIO()
        Image.new("RGB", (30, 18), "red").save(other, format="JPEG")
        self.assertEqual(self.post(data, payload=other.getvalue()).status_code, 409)
        self.assertEqual(path.read_bytes(), self.source)
        self.assertEqual(path.stat(), original_stat)
        self.assertEqual(len(self.store.list_jobs()), 1)

    def test_invalid_images_limits_and_form_errors_leave_no_inputs(self):
        for payload in (b"not an image", self.source[:40]):
            self.assertEqual(self.post(self.data(), payload=payload).status_code, 422)
        for key, limit in (("INPUT_MAX_BYTES", 10), ("INPUT_MAX_PIXELS", 10)):
            with self.subTest(key=key), patch.dict(self.app.config, {key: limit}):
                self.assertEqual(self.post(self.data()).status_code, 422)
        response = self.post(self.data(num_frames="6"))
        self.assertEqual(response.status_code, 422)
        self.assertIn("Select your source image again", response.get_data(as_text=True))
        self.assertEqual(self.staged_files(), [])
        self.assertEqual(self.store.list_jobs(), ())
        self.runtime.assert_not_called()

    def test_csrf_and_submission_checks_run_before_persistence(self):
        for field, value in (("csrf_token", "bad"), ("submission_key", "../escape")):
            self.assertEqual(self.post(self.data(**{field: value})).status_code, 400)
        self.assertEqual(self.staged_files(), [])

    def test_total_request_limit_reports_actionable_error(self):
        data = self.data()
        with patch.dict(self.app.config, {"MAX_CONTENT_LENGTH": 100}):
            response = self.post(data)
        self.assertEqual(response.status_code, 413)
        self.assertIn("choose a smaller source image", response.get_data(as_text=True))
        self.assertEqual(self.staged_files(), [])

    def test_exif_source_is_preserved_and_conditioning_is_oriented(self):
        payload = BytesIO()
        exif = Image.Exif()
        exif[274] = 6  # Rotate a landscape upload 90 degrees for conditioning.
        Image.new("RGB", (300, 180), "navy").save(payload, format="JPEG", exif=exif)
        response = self.post(self.data(), payload=payload.getvalue())
        self.assertEqual(response.status_code, 303)
        identifier = response.location.rsplit("/", 1)[1]
        source = Path(self.store.get(identifier).request["inputs"]["input_image"])
        self.assertEqual(source.read_bytes(), payload.getvalue())
        self.assertEqual(source.suffix, ".jpg")
        self.worker().run(once=True)
        metadata = json.loads(self.store.get(identifier).outputs[0].metadata_path.read_text())
        self.assertEqual(metadata["input_image"]["source_size"], [180, 300])
        self.assertEqual(metadata["input_image"]["conditioning_size"], [32, 48])

    def test_changed_queued_input_is_rejected_before_runtime_loading(self):
        identifier = self.submit()
        path = Path(self.store.get(identifier).request["inputs"]["input_image"])
        path.chmod(0o600)
        path.write_bytes(b"changed")
        self.worker().run(once=True)
        job = self.store.get(identifier)
        self.assertEqual(job.status, "failed")
        self.assertIn("input changed after submission", job.error)
        self.runtime.assert_not_called()

    def test_concurrent_staging_failure_cannot_remove_successful_retry(self):
        from image_generator.web.uploads import stage_source

        key = "a" * 32
        first_staged, release = Event(), Event()
        failures = []
        def stage(retain):
            try:
                with self.app.app_context():
                    if retain:
                        first_staged.wait(5)
                    with stage_source(key, (self.source, "png")) as source:
                        if retain:
                            source.retain()
                        else:
                            first_staged.set()
                            if not release.wait(5):
                                raise RuntimeError("test did not release upload")
                            raise ValueError("enqueue failed")
            except ValueError as error:
                if retain:
                    failures.append(error)
            except BaseException as error:
                failures.append(error)
        first = Thread(target=stage, args=(False,))
        retry = Thread(target=stage, args=(True,))
        first.start()
        self.assertTrue(first_staged.wait(5))
        retry.start()
        release.set()
        first.join(5)
        retry.join(5)
        self.assertFalse(first.is_alive() or retry.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual((self.inputs / key / "source.png").read_bytes(), self.source)

    def test_failed_validation_and_enqueue_remove_staged_sources(self):
        (self.catalog / "WAN/loras/high/missing-high.safetensors").unlink()
        response = self.post(self.data())
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.staged_files(), [])
        data = self.data()
        data.poplist("lora")
        with patch("image_generator.jobs.JobStore.enqueue", side_effect=sqlite3.OperationalError("busy")):
            self.assertEqual(self.post(data).status_code, 503)
        self.assertEqual(self.staged_files(), [])
        with patch("image_generator.jobs.JobStore.enqueue", side_effect=OSError("disk full")):
            self.assertEqual(self.post(data).status_code, 422)
        self.assertEqual(self.staged_files(), [])

    def test_preview_decodes_but_does_not_stage_or_queue(self):
        response = self.post(self.data(action="preview"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.staged_files(), [])
        self.assertEqual(self.store.list_jobs(), ())
        self.runtime.assert_not_called()

    def test_mp4_metadata_range_seeking_downloads_and_conditioning(self):
        import imageio.v2 as imageio

        identifier = self.submit(seed=str(2**63 - 1))
        self.worker().run(once=True)
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed")
        record = self.store.get(identifier).outputs[0]
        for path in (record.file_path, record.metadata_path, record.conditioning_path):
            self.assertEqual(path.parent, self.videos / identifier)
        self.assertFalse((self.outputs / identifier).exists())
        output = result["outputs"][0]
        self.assertEqual((output["width"], output["height"]), (48, 32))
        self.assertEqual((output["frame_count"], output["fps"]), (5, 12))
        self.assertAlmostEqual(output["duration_seconds"], 5 / 12)
        self.assertEqual(output["seed"], str(2**63 - 1))
        media = self.client.get(output["url"])
        self.assertEqual(media.mimetype, "video/mp4")
        self.assertGreater(len(media.data), 100)
        media.close()
        seek = self.client.get(output["url"], headers={"Range": "bytes=10-29"})
        self.assertEqual(seek.status_code, 206)
        self.assertEqual(len(seek.data), 20)
        self.assertIn("bytes 10-29/", seek.headers["Content-Range"])
        seek.close()
        download = self.client.get(output["download_url"])
        self.assertIn("attachment", download.headers["Content-Disposition"])
        download.close()
        conditioning = self.client.get(output["conditioning_url"])
        with Image.open(BytesIO(conditioning.data)) as image:
            self.assertEqual(image.size, (48, 32))
        conditioning.close()
        metadata = self.client.get(output["metadata_url"])
        self.assertEqual(metadata.json["output_video"]["width"], 48)
        metadata.close()
        with imageio.get_reader(str(self.store.get(identifier).outputs[0].file_path)) as reader:
            self.assertEqual(reader.count_frames(), 5)
            self.assertEqual(reader.get_data(4).shape, (32, 48, 3))
        html = self.client.get(f"/jobs/{identifier}").get_data(as_text=True)
        for expected in ('<video', 'controls playsinline', '48 × 32', '5 frames', '12 FPS',
                         'Uploaded source', 'Exact conditioning image', '1 of 1 videos saved'):
            self.assertIn(expected, html)

    def test_existing_videos_in_image_output_root_remain_downloadable(self):
        identifier = self.submit()
        self.worker(video_output_dir=self.outputs).run(once=True)
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed")
        output = result["outputs"][0]
        for key in ("download_url", "metadata_url", "conditioning_url"):
            response = self.client.get(output[key])
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_second_variation_progress_has_separate_steps_and_saved_count(self):
        identifier = self.submit(num_videos_per_prompt="2")
        entered, release = Event(), Event()
        def on_step(step):
            if len(self.calls) == 2 and step == 1:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test did not release inference")
        self.on_step = on_step
        failures = []
        ticks = count()
        def run():
            try:
                with patch("image_generator.worker.monotonic", side_effect=lambda: next(ticks)):
                    self.worker().run(once=True)
            except BaseException as error:
                failures.append(error)
        thread = Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(10))
            result = self.client.get(f"/api/jobs/{identifier}").json
            self.assertEqual(result["progress"]["stage"], "denoising")
            self.assertEqual((result["progress"]["current"], result["progress"]["total"]), (2, 4))
            self.assertEqual(result["progress"]["completion_text"], "1 of 2 videos saved · Video 2 of 2")
            self.assertEqual(len(result["outputs"]), 1)
        finally:
            release.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        job = self.store.get(identifier)
        self.assertEqual(job.status, "completed")
        self.assertEqual([output.seed for output in job.outputs], [42, 43])
        for stage in ("conditioning_image", "decoding_video", "encoding_video"):
            self.assertIn(stage, [event.stage for event in job.events])

    def test_experts_are_reset_before_changed_selection_and_checkpoint_switch(self):
        data = self.data()
        data.add("lora", "fixture-wan-low")
        self.assertEqual(self.post(data).status_code, 303)
        cleared = self.data()
        cleared.poplist("lora")
        self.assertEqual(self.post(cleared).status_code, 303)
        low = self.data()
        low.setlist("lora", ["fixture-wan-low"])
        self.assertEqual(self.post(low).status_code, 303)
        path = self.catalog / "WAN/checkpoints/demo/model.defaults.json"
        second = json.loads(path.read_text())
        second.update(id="wan-second", display_name="Second WAN checkpoint")
        second_path = path.parent.parent / "second/model.defaults.json"
        second_path.parent.mkdir()
        second_path.write_text(json.dumps(second))
        switched = self.submit("wan-second")
        self.worker().run(until_idle=True)
        self.assertEqual(len(self.pipelines), 2)
        self.assertEqual([adapters for pipe, adapters in self.calls[:3]],
                         [{False: ("adapter_0",), True: ("adapter_1",)}, {False: (), True: ()},
                          {False: (), True: ("adapter_0",)}])
        self.pipelines[0].transformer.set_adapters.assert_called_once_with(["adapter_0"], weights=[0.7])
        self.assertEqual(self.pipelines[0].unload_lora_weights.call_count, 1)
        self.assertIn("switching_model", [event.stage for event in self.store.get(switched).events])

    def test_later_inference_failure_preserves_video_and_next_job_recovers(self):
        failed = self.submit(num_videos_per_prompt="2")
        following = self.submit()
        self.fail_on_call = 2
        self.worker().run(until_idle=True)
        result = self.client.get(f"/api/jobs/{failed}").json
        self.assertEqual(result["status"], "failed")
        self.assertIn("synthetic WAN inference failure", result["error"])
        self.assertEqual(len(result["outputs"]), 1)
        self.assertEqual(self.store.get(following).status, "completed")
        self.assertEqual(len(self.pipelines), 2)
        self.assertIn("<video", self.client.get(f"/jobs/{failed}").get_data(as_text=True))

    def test_encoder_failure_cleans_partial_output_and_recovers(self):
        failed = self.submit()
        following = self.submit()
        def failed_export(frames, path, fps):
            path.write_bytes(b"partial")
            raise RuntimeError("synthetic encoder failure")
        with self.worker() as worker:
            with patch("image_generator.generation.export_video", side_effect=failed_export):
                self.assertEqual(worker.run_once().status, "failed")
            self.assertEqual(worker.run_once().status, "completed")
        self.assertEqual(list((self.videos / failed).iterdir()), [])
        self.assertEqual(len(self.pipelines), 2)
        self.assertEqual(self.store.get(following).status, "completed")

    def test_input_and_output_routes_reject_symlinks_outside_managed_roots(self):
        identifier = self.submit()
        self.worker().run(once=True)
        job = self.store.get(identifier)
        source = Path(job.request["inputs"]["input_image"])
        outside = self.root / "outside.png"
        outside.write_bytes(self.source)
        source.unlink()
        source.symlink_to(outside)
        self.assertEqual(self.client.get(f"/jobs/{identifier}/source").status_code, 404)
        record = job.outputs[0]
        for part, path in (("media", record.file_path), ("metadata", record.metadata_path),
                           ("conditioning", record.conditioning_path)):
            path.unlink()
            path.symlink_to(outside)
            self.assertEqual(self.client.get(f"/jobs/{identifier}/outputs/{record.id}/{part}").status_code, 404)
        self.assertEqual(self.client.get("/jobs/unknown/source").status_code, 404)
        self.assertEqual(self.client.get(f"/api/jobs/{identifier}").status_code, 200)


if __name__ == "__main__":
    unittest.main()

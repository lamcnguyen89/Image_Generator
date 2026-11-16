"""MiniMax multipart submissions through native node doubles and the real worker."""

from io import BytesIO
from itertools import count
import json
from pathlib import Path
import sqlite3
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from test_generation_ui import FormPage
from test_minimax import MiniMaxFixture
from image_generator import create_app
from image_generator.jobs import JobStore
from image_generator.worker import Worker


class MiniMaxExecutionTests(MiniMaxFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.inputs, self.outputs = self.root / "inputs", self.root / "outputs"
        self.videos = self.root / "Videos"
        self.config = {"TESTING": True, "CATALOG_ROOT": self.root,
                       "JOB_DATABASE": self.root / "jobs.sqlite3", "SECRET_KEY": "minimax-tests",
                       "JOB_INPUT_ROOT": self.inputs, "JOB_OUTPUT_ROOT": self.outputs,
                       "JOB_VIDEO_OUTPUT_ROOT": self.videos}
        self.app = create_app(self.config)
        self.client = self.app.test_client()
        self.store = JobStore(self.config["JOB_DATABASE"])
        self.original = self.source.read_bytes()
        self.worker_factory = patch("image_generator.worker.create_workflow", return_value=self.workflow)
        self.worker_factory.start()
        self.addCleanup(self.worker_factory.stop)
        self.previous_hook = Mock()
        self.runtime.utils.PROGRESS_BAR_HOOK = self.previous_hook
        self.sample_count, self.on_step, self.fail_on_sample = 0, None, None
        self.runtime.sampling.SamplerCustomAdvanced.execute.side_effect = self.sample

        def condition(**kwargs):
            # Decode to the actual conditioning dimensions, as native H3 does.
            self.frames = np.zeros((kwargs["length"], kwargs["height"], kwargs["width"], 3), dtype=np.float32)
            return type("Conditioning", (), {"result": ("positive", "latent")})()

        self.runtime.h3.MiniMaxH3ImageToVideo.execute.side_effect = condition
        for node in (self.runtime.nodes.VAEDecodeTiled, self.runtime.nodes.VAEDecode):
            node.return_value.decode.side_effect = lambda **kwargs: (self.frames,)

    def sample(self, **kwargs):
        self.sample_count += 1
        for step in range(4):
            self.runtime.utils.PROGRESS_BAR_HOOK(step + 1, 4, object(), node_id="sampler")
            if self.on_step:
                self.on_step(step)
        if self.sample_count == self.fail_on_sample:
            raise RuntimeError("synthetic H3 sampling failure")
        return ("sampled",)

    def data(self, *, mode="img2vid", **values):
        page = FormPage(self.client.get(f"/generate/fixture-minimax?mode={mode}").get_data(as_text=True))
        data = page.data()
        for name, value in {"action": "generate", "num_inference_steps": "4", **values}.items():
            data[name] = value
        return data

    def post(self, data, payload=None):
        data["input_image"] = (BytesIO(self.original if payload is None else payload), "../../source.html")
        return self.client.post("/generate/fixture-minimax", data=data)

    def submit(self, **values):
        response = self.post(self.data(**values))
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True))
        return response.location.rsplit("/", 1)[1]

    def run_worker(self, **options):
        options.setdefault("video_output_dir", self.videos)
        Worker(self.store, self.outputs, **options).run(until_idle=True)

    def staged_files(self):
        return list(self.inputs.glob("*/source.*"))

    def test_mode_switch_preserves_shared_edits_and_requires_image_only_for_img2vid(self):
        data = self.data(action="change_mode", prompt="A new scene", seed="123")
        data["mode"] = "txt2vid"
        data.pop("enable_vae_tiling", None)
        response = self.client.post("/generate/fixture-minimax", data=data)
        self.assertEqual(response.status_code, 200)
        page = FormPage(response.get_data(as_text=True))
        self.assertEqual([item["value"] for item in page.options["mode"]], ["img2vid", "txt2vid"])
        self.assertEqual(page.controls["prompt"][0]["value"], "A new scene")
        self.assertEqual(page.controls["seed"][0]["value"], "123")
        self.assertNotIn("checked", page.controls["enable_vae_tiling"][0])
        self.assertNotIn("input_image", page.controls)
        self.assertNotIn("preserve_aspect_ratio", page.controls)
        self.assertIn("Generate video", response.get_data(as_text=True))
        self.assertNotIn("Generate images", response.get_data(as_text=True))
        back = page.data()
        back["mode"], back["action"] = "img2vid", "change_mode"
        response = self.client.post("/generate/fixture-minimax", data=back)
        self.assertEqual(response.status_code, 200)
        page = FormPage(response.get_data(as_text=True))
        self.assertIn("required", page.controls["input_image"][0])
        self.assertIn("preserve_aspect_ratio", page.controls)
        missing = page.data()
        missing["action"] = "generate"
        self.assertEqual(self.client.post("/generate/fixture-minimax", data=missing).status_code, 422)
        self.assertEqual(self.store.list_jobs(), ())
        self.assertEqual(self.staged_files(), [])
        self.load_runtime.assert_not_called()

    def test_text_to_video_preview_retry_and_worker_export_without_source(self):
        import imageio.v2 as imageio

        preview = self.client.post("/generate/fixture-minimax", data=self.data(mode="txt2vid", action="preview"))
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(FormPage(preview.get_data(as_text=True)).preview["input_files"], {})
        data = self.data(mode="txt2vid", num_videos_per_prompt="2")
        response = self.client.post("/generate/fixture-minimax", data=data)
        self.assertEqual(response.status_code, 303, response.get_data(as_text=True))
        identifier = response.location.rsplit("/", 1)[1]
        self.assertEqual(self.client.post("/generate/fixture-minimax", data=data).location, response.location)
        data["seed"] = "99"
        self.assertEqual(self.client.post("/generate/fixture-minimax", data=data).status_code, 409)
        self.assertEqual(self.staged_files(), [])
        self.load_runtime.assert_not_called()
        self.run_worker()
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed", result["error"])
        self.assertEqual(result["mode"], "txt2vid")
        self.assertIsNone(result["source_url"])
        self.assertEqual(len(result["outputs"]), 2)
        for output in result["outputs"]:
            self.assertIsNone(output["conditioning_url"])
            self.assertEqual((output["width"], output["height"], output["frame_count"]), (64, 32, 5))
            response = self.client.get(output["metadata_url"])
            self.assertEqual(response.json["mode"], "txt2vid")
            self.assertEqual(response.json["inputs"], {})
            self.assertIsNone(response.json["conditioning_image"])
            self.assertNotIn("input_image", response.json)
            response.close()
            response = self.client.get(output["url"], headers={"Range": "bytes=10-29"})
            self.assertEqual(response.status_code, 206)
            response.close()
        records = self.store.get(identifier).outputs
        self.assertEqual([record.seed for record in records], [42, 43])
        self.assertTrue(all(record.conditioning_path is None for record in records))
        self.assertEqual(list((self.videos / identifier).glob("*.png")), [])
        with imageio.get_reader(str(records[0].file_path)) as reader:
            self.assertEqual(reader.count_frames(), 5)
            self.assertEqual(reader.get_data(0).shape, (32, 64, 3))
        self.assertEqual(self.client.get(f"/jobs/{identifier}/source").status_code, 404)
        self.assertEqual(self.client.get(f"/jobs/{identifier}/outputs/{records[0].id}/conditioning").status_code, 404)
        html = self.client.get(f"/jobs/{identifier}").get_data(as_text=True)
        self.assertIn("<video", html)
        self.assertNotIn("Uploaded source", html)
        self.assertNotIn("Exact conditioning image", html)

    def test_text_to_video_rejects_uploads_and_forged_fixed_settings(self):
        self.assertEqual(self.post(self.data(mode="txt2vid")).status_code, 422)
        for values, status in (({"fps": "16"}, 422), ({"num_frames": "49"}, 422),
                               ({"negative_prompt": "blur"}, 422), ({"csrf_token": "bad"}, 400)):
            with self.subTest(values=values):
                response = self.client.post("/generate/fixture-minimax", data=self.data(mode="txt2vid", **values))
                self.assertEqual(response.status_code, status)
        self.assertEqual(self.store.list_jobs(), ())
        self.assertEqual(self.staged_files(), [])
        self.load_runtime.assert_not_called()

    def test_form_exposes_generation_and_readonly_fixed_sampling(self):
        html = self.client.get("/generate/fixture-minimax").get_data(as_text=True)
        self.assertIn("Generate video", html)
        self.assertIn("17*n + 5", html)
        page = FormPage(html)
        for name in ("fps", "guidance_scale", "negative_prompt", "sampling.sampler", "sampling.noise_schedule"):
            self.assertIn("readonly", page.controls[name][0])

    def test_source_is_original_immutable_retry_stable_and_survives_restart(self):
        data = self.data()
        first = self.post(data)
        self.assertEqual(first.status_code, 303)
        self.load_runtime.assert_not_called()
        path = self.staged_files()[0]
        stat = path.stat()
        self.assertEqual(path.read_bytes(), self.original)
        self.assertEqual(path.name, "source.png")
        self.assertEqual(stat.st_mode & 0o222, 0)
        self.assertEqual(self.post(data).location, first.location)
        data["seed"] = "99"
        self.assertEqual(self.post(data).status_code, 409)
        self.assertEqual(self.post(data, b"different invalid bytes").status_code, 422)
        other = BytesIO()
        Image.new("RGB", (32, 32), "red").save(other, format="JPEG")
        self.assertEqual(self.post(data, other.getvalue()).status_code, 409)
        self.assertEqual(path.stat(), stat)
        self.assertEqual(path.read_bytes(), self.original)
        self.assertEqual(len(self.store.list_jobs()), 1)
        client = create_app(self.config).test_client()
        response = client.get(first.location + "/source")
        self.assertEqual(response.data, self.original)
        response.close()
        self.run_worker()
        self.assertEqual(self.store.list_jobs()[0].status, "completed")

    def test_invalid_uploads_limits_and_fixed_values_never_stage_or_load(self):
        for payload in (b"not an image", self.original[:40]):
            self.assertEqual(self.post(self.data(), payload).status_code, 422)
        animated = BytesIO()
        Image.new("RGB", (32, 32), "red").save(animated, format="GIF", save_all=True,
            append_images=[Image.new("RGB", (32, 32), "blue")])
        self.assertEqual(self.post(self.data(), animated.getvalue()).status_code, 422)
        for key in ("INPUT_MAX_BYTES", "INPUT_MAX_PIXELS"):
            with patch.dict(self.app.config, {key: 10}):
                self.assertEqual(self.post(self.data()).status_code, 422)
        for field, value in (("fps", "16"), ("guidance_scale", "2"), ("negative_prompt", "blur"),
                             ("sampling.sampler", "euler"), ("sampling.noise_schedule", "default"),
                             ("num_frames", "49"), ("num_frames", "3600"), ("width", "48")):
            with self.subTest(field=field):
                self.assertEqual(self.post(self.data(**{field: value})).status_code, 422)
        self.assertEqual(self.staged_files(), [])
        self.assertEqual(self.store.list_jobs(), ())
        self.load_runtime.assert_not_called()

    def test_csrf_submission_keys_and_request_limits_reject_before_storage(self):
        for field, value in (("csrf_token", "bad"), ("submission_key", "../escape")):
            self.assertEqual(self.post(self.data(**{field: value})).status_code, 400)
        with patch.dict(self.app.config, {"MAX_CONTENT_LENGTH": 100}):
            self.assertEqual(self.post(self.data()).status_code, 413)
        self.assertEqual(self.staged_files(), [])

    def test_preview_validates_image_without_staging_or_runtime(self):
        response = self.post(self.data(action="preview"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.post(self.data(action="preview"), b"bad").status_code, 422)
        self.assertEqual(self.staged_files(), [])
        self.assertEqual(self.store.list_jobs(), ())
        self.load_runtime.assert_not_called()

    def test_failed_preparation_and_enqueue_cleanup_new_sources(self):
        self.adapter.unlink()
        self.assertEqual(self.post(self.data()).status_code, 422)
        self.adapter.write_bytes(b"adapter")
        for error, status in ((sqlite3.OperationalError("busy"), 503), (OSError("disk full"), 422)):
            with patch("image_generator.jobs.JobStore.enqueue", side_effect=error):
                self.assertEqual(self.post(self.data()).status_code, status)
        self.assertEqual(self.staged_files(), [])

    def test_real_mp4_metadata_seeking_download_and_conditioning(self):
        import imageio.v2 as imageio

        identifier = self.submit(seed=str(2**63 - 1))
        self.run_worker()
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed")
        record = self.store.get(identifier).outputs[0]
        for path in (record.file_path, record.metadata_path, record.conditioning_path):
            self.assertEqual(path.parent, self.videos / identifier)
        self.assertFalse((self.outputs / identifier).exists())
        output = result["outputs"][0]
        self.assertEqual((output["width"], output["height"], output["frame_count"], output["fps"]), (32, 32, 5, 24))
        self.assertAlmostEqual(output["duration_seconds"], 5 / 24)
        self.assertFalse(output["audio_exported"])
        self.assertEqual(output["seed"], str(2**63 - 1))
        response = self.client.get(output["url"], headers={"Range": "bytes=10-29"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(len(response.data), 20)
        self.assertIn("bytes 10-29/", response.headers["Content-Range"])
        response.close()
        response = self.client.get(output["download_url"])
        self.assertIn("attachment", response.headers["Content-Disposition"])
        response.close()
        response = self.client.get(output["conditioning_url"])
        with Image.open(BytesIO(response.data)) as image:
            self.assertEqual(image.size, (32, 32))
        response.close()
        response = self.client.get(output["metadata_url"])
        self.assertFalse(response.json["audio_exported"])
        response.close()
        with imageio.get_reader(str(self.store.get(identifier).outputs[0].file_path)) as reader:
            self.assertEqual(reader.count_frames(), 5)
            self.assertEqual(reader.get_data(4).shape, (32, 32, 3))
        html = self.client.get(f"/jobs/{identifier}").get_data(as_text=True)
        for text in ("<video", "controls playsinline", "32 × 32", "24 FPS", "Silent video",
                     "Uploaded source", "Exact conditioning image"):
            self.assertIn(text, html)

    def test_existing_videos_in_image_output_root_remain_downloadable(self):
        identifier = self.submit()
        self.run_worker(video_output_dir=self.outputs)
        result = self.client.get(f"/api/jobs/{identifier}").json
        self.assertEqual(result["status"], "completed")
        output = result["outputs"][0]
        for key in ("download_url", "metadata_url", "conditioning_url"):
            response = self.client.get(output[key])
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_progress_is_visible_mid_sampling_with_previous_variation_playable(self):
        identifier = self.submit(num_videos_per_prompt="2")
        entered, release = Event(), Event()

        def on_step(step):
            if self.sample_count == 2 and step == 1:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("sampling was not released")

        self.on_step = on_step
        failures = []
        ticks = count()

        def run():
            try:
                with patch("image_generator.worker.monotonic", side_effect=lambda: next(ticks)):
                    self.run_worker()
            except BaseException as error:
                failures.append(error)

        thread = Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(10))
            result = self.client.get(f"/api/jobs/{identifier}").json
            self.assertEqual((result["progress"]["stage"], result["progress"]["current"], result["progress"]["total"]),
                             ("denoising", 2, 4))
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
        for stage in ("conditioning_image", "encoding_prompt", "decoding_video", "encoding_video"):
            self.assertIn(stage, [event.stage for event in job.events])
        self.assertIs(self.runtime.utils.PROGRESS_BAR_HOOK, self.previous_hook)

    def test_later_sampling_failure_preserves_output_and_next_job_recovers(self):
        failed = self.submit(num_videos_per_prompt="2")
        following = self.submit()
        self.fail_on_sample = 2
        self.run_worker()
        result = self.client.get(f"/api/jobs/{failed}").json
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["outputs"]), 1)
        self.assertIn("synthetic H3 sampling failure", result["error"])
        self.assertEqual(self.store.get(following).status, "completed")
        self.assertIs(self.runtime.utils.PROGRESS_BAR_HOOK, self.previous_hook)
        self.assertEqual(self.runtime.sd.load_diffusion_model.call_count, 2)

    def test_source_and_outputs_are_confined_to_the_associated_job(self):
        identifier = self.submit()
        self.run_worker()
        job = self.store.get(identifier)
        source = Path(job.request["inputs"]["input_image"])
        source.unlink()
        source.symlink_to(self.source)
        self.assertEqual(self.client.get(f"/jobs/{identifier}/source").status_code, 404)
        record = job.outputs[0]
        for part, path in (("media", record.file_path), ("metadata", record.metadata_path),
                           ("conditioning", record.conditioning_path)):
            path.unlink()
            path.symlink_to(self.source)
            self.assertEqual(self.client.get(f"/jobs/{identifier}/outputs/{record.id}/{part}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/jobs/{identifier}").status_code, 200)


if __name__ == "__main__":
    unittest.main()

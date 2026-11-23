"""Video forms and worker dispatch using real CPU solvers and synthetic frames."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from test_generation_ui import FormPage
from test_wan_execution import WanExecutionFixture

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"),
                     "WAN solver integration requires Torch and Diffusers")
class WanSamplerExecutionTests(WanExecutionFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        import diffusers
        import torch

        self.native_torch = torch
        self.diffusers.UniPCMultistepScheduler = diffusers.UniPCMultistepScheduler
        self.diffusers.DPMSolverMultistepScheduler = diffusers.DPMSolverMultistepScheduler
        original_load = self.loader.side_effect

        def load(*args, **options):
            pipe = original_load(*args, **options)
            pipe.scheduler = diffusers.UniPCMultistepScheduler(
                prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
            pipe.transformer.config.in_channels = pipe.transformer.config.out_channels = 16
            pipe.transformer_2.config = SimpleNamespace(in_channels=16, out_channels=16)
            return pipe

        self.loader.side_effect = load
        self.diffusers.WanPipeline = SimpleNamespace(from_pretrained=self.loader)
        path = self.catalog / "WAN/checkpoints/t2v/model.defaults.json"
        path.parent.mkdir()
        document = json.loads((ROOT / "Models/Model_Families/WAN/checkpoints/wan2_2_t2v/model.defaults.json").read_text())
        document.update(id="fixture-wan-t2v", notes_file=None)
        document["settings"].update(prompt="A lake", width=64, height=32, num_frames=5, seed=42)
        path.write_text(json.dumps(document))
        self.on_step = self.advance_solver

    def advance_solver(self, step):
        scheduler = self.pipelines[-1].scheduler
        if step == 0:
            scheduler.set_timesteps(4, device="cpu")
            self.latents = self.native_torch.full((1, 4, 2, 2, 2), 0.3)
        timestep = scheduler.timesteps[step]
        self.latents = scheduler.step(self.latents.tanh() * 0.1, timestep, self.latents).prev_sample
        self.assertTrue(self.native_torch.isfinite(self.latents).all())

    def test_i2v_and_t2v_save_queue_and_dispatch_all_three_wan_samplers(self):
        jobs = []
        for model in ("fixture-wan", "fixture-wan-t2v"):
            for sampler in ("model_default", "uni_pc", "dpmpp_2m"):
                with self.subTest(model=model, sampler=sampler):
                    page = FormPage(self.client.get(f"/generate/{model}").text)
                    self.assertEqual([item["value"] for item in page.options["sampling.sampler"]],
                                     ["model_default", "uni_pc", "dpmpp_2m"])
                    self.assertEqual([item["value"] for item in page.options["sampling.noise_schedule"]],
                                     ["default", "normal", "simple"])
                    data = self.data(model, action="save_model", **{"sampling.sampler": sampler})
                    data.pop("input_image", None)
                    self.assertEqual(self.client.post(f"/settings/models/{model}", data=data).status_code, 303)
                    data = self.data(model)
                    self.assertEqual(data["sampling.sampler"], sampler)
                    if model.endswith("t2v"):
                        data.pop("input_image", None)
                        response = self.client.post(f"/generate/{model}", data=data)
                    else:
                        response = self.post(data, model)
                    self.assertEqual(response.status_code, 303)
                    jobs.append((response.location.rsplit("/", 1)[1], sampler))
        self.worker().run(until_idle=True)
        reference = None
        for identifier, sampler in jobs:
            job = self.store.get(identifier)
            self.assertEqual(job.status, "completed", job.error)
            metadata = json.loads(job.outputs[0].metadata_path.read_text())
            self.assertEqual(metadata["settings"]["sampling"], {"sampler": sampler, "noise_schedule": "default"})
            grid = (metadata["scheduler"]["sigmas"], metadata["scheduler"]["timesteps"])
            if reference is None:
                reference = grid
            self.assertEqual(grid, reference)
            if sampler == "dpmpp_2m":
                self.assertEqual(metadata["scheduler"]["config"]["algorithm_type"], "dpmsolver++")
        self.assertEqual(len(self.pipelines), 2)

    def test_named_schedules_save_and_execute_in_both_modes_for_all_samplers(self):
        jobs = []
        for model in ("fixture-wan", "fixture-wan-t2v"):
            for sampler in ("model_default", "uni_pc", "dpmpp_2m"):
                for schedule in ("normal", "simple"):
                    pair = {"sampler": sampler, "noise_schedule": schedule}
                    data = self.data(model, action="save_model",
                                     **{"sampling.sampler": sampler, "sampling.noise_schedule": schedule})
                    data.pop("input_image", None)
                    self.assertEqual(self.client.post(f"/settings/models/{model}", data=data).status_code, 303)
                    data = self.data(model)
                    self.assertEqual(data["sampling.noise_schedule"], schedule)
                    if model.endswith("t2v"):
                        data.pop("input_image", None)
                        response = self.client.post(f"/generate/{model}", data=data)
                    else:
                        response = self.post(data, model)
                    self.assertEqual(response.status_code, 303)
                    jobs.append((response.location.rsplit("/", 1)[1], pair))
        self.worker().run(until_idle=True)
        grids = {}
        for identifier, pair in jobs:
            job = self.store.get(identifier)
            self.assertEqual(job.status, "completed", job.error)
            metadata = json.loads(job.outputs[0].metadata_path.read_text())
            self.assertEqual(metadata["settings"]["sampling"], pair)
            grid = (metadata["scheduler"]["sigmas"], metadata["scheduler"]["timesteps"])
            self.assertEqual(grid, grids.setdefault(pair["noise_schedule"], grid))
        self.assertNotEqual(grids["normal"], grids["simple"])
        self.assertEqual(len(self.pipelines), 2)

"""Exercise Qwen 2.1 notebook source switching and adapter state without weights."""

from contextlib import nullcontext, redirect_stdout
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
from time import perf_counter
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
from uuid import uuid4

from PIL import Image


NOTEBOOK = Path(__file__).resolve().parents[1] / "Models/Model_Families/QWEN_2_1/QWEN_2_1_Image_Gen.ipynb"


class Config(dict):
    __getattr__ = dict.__getitem__


class Qwen21NotebookTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text())
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        self.ns = {name: globals()[name] for name in (
            "gc", "hashlib", "json", "math", "os", "Path", "re", "secrets", "perf_counter", "uuid4",
        )}
        self.run_cell("generation-settings")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir()
        self.loras = self.root / "loras"
        self.loras.mkdir()
        self.checkpoint = self.checkpoints / "custom.safetensors"
        self.checkpoint.touch()
        for name in ("style", "detail"):
            (self.loras / f"{name}.safetensors").touch()
        self.ns.update(project_root=self.root, checkpoint_root=self.checkpoints, lora_root=self.loras,
                       checkpoint_file="custom.safetensors", loras=[], prompts=["a lake", "a lake", "a forest"],
                       show_images=False, width=64, height=32)
        component = SimpleNamespace(device="cuda:0", dtype="bf16")
        self.pipe = Mock(transformer=component, text_encoder=component, vae=component)
        self.pipe.config = Config(sample_sigmas=None)
        self.pipe.scheduler = SimpleNamespace(config={})
        self.pipe._notebook_qwen21_lora_key = None
        self.ns["load_qwen21_lora"] = Mock(side_effect=lambda pipe, path, name: pipe.load_lora_weights(
            str(path.parent), weight_name=path.name, adapter_name=name,
            local_files_only=True, use_safetensors=True,
        ))
        self.loader = Mock(return_value=self.pipe)
        self.transformer = Mock()
        self.transformer_loader = Mock(return_value=Mock(to=Mock(return_value=self.transformer)))
        self.hub_download = Mock()
        cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                               mem_get_info=lambda device: (2**30, 2**31), get_device_name=lambda device: "test",
                               empty_cache=Mock(), synchronize=Mock(), reset_peak_memory_stats=Mock(),
                               max_memory_allocated=lambda device: 0)
        torch = SimpleNamespace(__version__="test", bfloat16="bf16", device=str, cuda=cuda,
                                version=SimpleNamespace(cuda="test"), inference_mode=nullcontext,
                                Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed))
        self.ns.update(
            torch=torch, QwenImage21Pipeline=SimpleNamespace(from_pretrained=self.loader),
            QwenImage21Transformer2DModel=SimpleNamespace(from_single_file=self.transformer_loader),
            hf_hub_download=self.hub_download, HfHubHTTPError=OSError, GatedRepoError=PermissionError,
            load_dotenv=Mock(), dotenv_values=lambda path: {}, display=Mock(),
            diffusers=SimpleNamespace(__version__="test"), transformers=SimpleNamespace(__version__="test"),
            peft=SimpleNamespace(__version__="test"),
        )

    def run_cell(self, name):
        with redirect_stdout(io.StringIO()):
            exec(compile(self.cells[name], name, "exec"), self.ns)

    def validate(self):
        self.run_cell("validate-settings")

    def load(self):
        self.validate()
        self.run_cell("load-pipeline")

    def stack(self):
        return [
            {"name": "style", "weights": "style.safetensors", "scale": 0.7},
            {"name": "detail", "weights": "detail.safetensors", "scale": -0.2},
            {"name": "missing", "weights": "absent.safetensors", "enabled": False},
        ]

    def test_notebook_code_compiles_and_outputs_are_cleared(self):
        for cell in json.loads(NOTEBOOK.read_text())["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_auto_prefers_checkpoint_and_falls_back_only_when_missing(self):
        self.validate()
        self.assertEqual(self.ns["checkpoint_path"], self.checkpoint)
        self.checkpoint.unlink()
        self.validate()
        self.assertEqual(self.ns["resolved_model_source"], "hub")
        self.assertIsNone(self.ns["checkpoint_key"])
        self.checkpoint.mkdir()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_checkpoint_required_and_hub_ignores_local_path(self):
        self.ns.update(model_source="checkpoint", checkpoint_file="absent.safetensors")
        with self.assertRaises(FileNotFoundError):
            self.validate()
        self.ns["checkpoint_file"] = None
        with self.assertRaisesRegex(ValueError, "Set checkpoint_file"):
            self.validate()
        self.ns.update(model_source="hub", checkpoint_file="irrelevant")
        self.validate()
        self.assertIsNone(self.ns["checkpoint_path"])
        self.ns.update(model_source="auto", checkpoint_file=None)
        self.validate()
        self.assertEqual(self.ns["resolved_model_source"], "hub")

    def test_qwen_dimensions_cfg_and_schedule_validation(self):
        for key, value in (("width", 48), ("true_cfg_scale", float("nan")), ("true_cfg_scale", 3),
                           ("sigmas", []), ("sigmas", [1, 0]), ("sigmas", [0.5, 0.8]),
                           ("sigmas", [1, True]), ("use_kv_cache", 1), ("model_source", "other")):
            with self.subTest(key=key, value=value):
                previous = self.ns[key]
                self.ns[key] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.ns[key] = previous
        self.ns.update(true_cfg_scale=2.5, negative_prompt="blurry", sigmas=[1.0, 0.8, 0.4])
        self.validate()

    def test_hub_loading_skips_single_file_and_checkpoint_replaces_transformer(self):
        self.ns["model_source"] = "hub"
        self.load()
        self.transformer_loader.assert_not_called()
        self.assertNotIn("transformer", self.loader.call_args.kwargs)
        self.assertNotIn("is_distilled", self.loader.call_args.kwargs)
        self.load()
        self.assertEqual(self.loader.call_count, 1)
        self.ns["model_source"] = "checkpoint"
        self.load()
        self.assertEqual(self.transformer_loader.call_args.args, (str(self.checkpoint),))
        self.assertIs(self.loader.call_args.kwargs["transformer"], self.transformer)
        self.assertEqual(self.transformer_loader.call_args.kwargs["config"], "Qwen/Qwen-Image-2.1")
        self.assertEqual(self.loader.call_count, 2)
        self.checkpoint.write_bytes(b"replacement")
        self.load()
        self.assertEqual(self.loader.call_count, 3)

    def test_access_failure_preserves_existing_pipeline(self):
        self.load()
        self.ns["model_id"] = "private/model"
        self.hub_download.side_effect = PermissionError("denied")
        with self.assertRaisesRegex(RuntimeError, "denied access"):
            self.load()
        self.assertIs(self.ns["pipe"], self.pipe)
        self.assertEqual(self.loader.call_count, 1)

    def test_multiple_loras_scale_reuse_and_disable_clear(self):
        self.ns["loras"] = self.stack()
        self.load()
        self.run_cell("configure-adapters")
        self.assertEqual(self.pipe.load_lora_weights.call_count, 2)
        self.pipe.set_adapters.assert_called_with(["style", "detail"], adapter_weights=[0.7, -0.2])
        self.ns["loras"][0]["scale"] = 0.4
        self.validate()
        self.run_cell("configure-adapters")
        self.assertEqual(self.pipe.load_lora_weights.call_count, 2)
        self.pipe.set_adapters.assert_called_with(["style", "detail"], adapter_weights=[0.4, -0.2])
        self.ns["use_loras"] = False
        self.validate()
        self.run_cell("configure-adapters")
        self.assertEqual(self.ns["lora_settings"], [])
        self.assertEqual(self.pipe._notebook_qwen21_lora_key, ())
        self.assertEqual(self.pipe.unload_lora_weights.call_count, 2)

    def test_invalid_lora_stack_fails_before_touching_pipeline(self):
        self.ns["loras"] = self.stack()
        self.ns["loras"][1]["name"] = "style"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate()
        self.ns["loras"] = [{"name": "missing", "weights": "absent.safetensors"}]
        with self.assertRaises(FileNotFoundError):
            self.validate()
        self.loader.assert_not_called()
        self.pipe.unload_lora_weights.assert_not_called()

    def test_failed_adapter_load_clears_partial_stack(self):
        self.ns["loras"] = self.stack()
        self.load()
        self.pipe.load_lora_weights.side_effect = [None, ValueError("incompatible adapter")]
        with self.assertRaisesRegex(RuntimeError, "adapters were cleared"):
            self.run_cell("configure-adapters")
        self.assertIsNone(self.pipe._notebook_qwen21_lora_key)
        self.assertEqual(self.pipe.unload_lora_weights.call_count, 2)

    def test_generation_uses_qwen_arguments_and_records_model_stack_schedule_and_seeds(self):
        self.ns.update(model_source="hub", loras=self.stack(), num_images_per_prompt=2, seed=2**63 - 1,
                       sigmas=[1.0, 0.8, 0.4])
        self.pipe.return_value = SimpleNamespace(images=[Image.new("RGB", (64, 32)) for _ in range(2)])
        self.load()
        self.run_cell("configure-adapters")
        self.run_cell("generate-images")
        self.assertEqual(self.pipe.call_count, 2)  # Duplicate prompts are skipped.
        arguments = self.pipe.call_args.kwargs
        self.assertNotIn("guidance_scale", arguments)
        self.assertNotIn("max_sequence_length", arguments)
        self.assertIsNone(arguments["negative_prompt"])
        self.assertEqual(arguments["true_cfg_scale"], 1.0)
        records = [json.loads(path.with_suffix(".json").read_text()) for path in self.ns["saved_paths"]]
        self.assertEqual([record["seed"] for record in records], [2**63 - 1, 0, 1, 2])
        for record in records:
            self.assertEqual(record["model_source"], "hub")
            self.assertIsNone(record["checkpoint"])
            self.assertIsNone(record["checkpoint_fingerprint"])
            self.assertEqual([item["name"] for item in record["loras"]], ["style", "detail"])
            self.assertEqual(record["effective_num_inference_steps"], 3)
        self.ns["sigmas"] = None
        self.ns.update(model_source="checkpoint", true_cfg_scale=2.5, negative_prompt="blurry")
        self.pipe.config["sample_sigmas"] = [1.0, 0.5]
        self.load()
        self.run_cell("configure-adapters")
        self.run_cell("generate-images")
        record = json.loads(self.ns["saved_paths"][0].with_suffix(".json").read_text())
        self.assertEqual(record["model_source"], "checkpoint")
        self.assertEqual(record["checkpoint"], str(self.checkpoint))
        self.assertEqual(record["negative_prompt"], "blurry")
        self.assertEqual(record["effective_num_inference_steps"], 2)
        self.pipe.config["sample_sigmas"] = None
        self.run_cell("generate-images")
        record = json.loads(self.ns["saved_paths"][0].with_suffix(".json").read_text())
        self.assertEqual(record["effective_num_inference_steps"], 40)


if __name__ == "__main__":
    unittest.main()

# Qwen-Image 2.1 workflow and UI

The Qwen family follows the existing transformer workflow architecture:
`Qwen21Workflow` owns the pipeline, the shared generation layer prepares requests
and saves outputs, and the shared queue worker serializes GPU execution. Flask
only validates forms, saves settings, and enqueues frozen requests. Importing
the workflow, browsing the catalog, previewing, and saving preferences do not
import Torch or Diffusers or load model weights.

## Select and configure a model

The catalog orders families as FLUX, **QWEN 2.1**, Krea2, SDXL, MiniMax, WAN.
Both Qwen cards use the shared generation layout:

- **Qwen-Image 2.1 (Hugging Face)** loads the complete repository.
- **Big Love Qiara1** supplies the existing local BF16 transformer to
  `QwenImage21Pipeline`; supporting components come from `base_model_id`.

The default controls are 2048 × 2048, 40 steps, true CFG 1, KV caching enabled,
and no custom sigma list. Dimensions must be multiples of 32. Negative prompting
is active only above true CFG 1 with a nonempty negative prompt.

Under Advanced settings, **Custom sigma schedule** accepts descending values
separated by commas, spaces, or newlines. Values must be finite and in (0, 1];
omit the terminal zero. Leave blank to preserve the repository schedule. An
explicit schedule takes precedence over repository sigmas and the inference
step count. Progress and saved metadata use the effective number of steps,
including when a repository such as Turbo ships a preset schedule.

Enable any number of compatible LoRAs, choose independent strengths, and opt
into trigger words. Qwen accepts signed strengths, including negative Turbo
adjustments; the other families retain their existing minimum of zero.
Adapters stay separate from base weights and are reset between jobs. Any load
or inference failure discards the pipeline so the following request starts clean.
The Qwen loader normalizes native/AI Toolkit keys and splits ComfyUI fused
gate/up LoRAs into separate projections. Unknown target modules fail before
loading, rather than allowing PEFT to apply only part of an incompatible file.
The notebook uses the same adapter loader.

Save model settings retains configured disabled adapters and checkpoint
strength overrides. Shared adapter defaults affect links that inherit strength.
Reset restores shipped defaults; previews and Generate do not save preferences.
The shipped Turbo link remains disabled until explicitly enabled.

Image editing is not offered by this text-to-image workflow. RGBA output is
preserved when the model produces it; request transparency in the prompt.
See the [family settings guide](../Models/Model_Families/QWEN_2_1/README.md)
for tuning suggestions and the distinction between the local Turbo adapter and
the official Turbo model. Native API details are in the
[Diffusers documentation](https://huggingface.co/docs/diffusers/en/api/pipelines/qwenimage21).

## CLI and runtime

```bash
.venv/bin/python scripts/generate_qwen21.py \
  --prompt "A mountain lake at sunrise" --seed 42 --no-loras --dry-run
```

Omit `--dry-run` to generate. The default model ID selects the official Hub
model; use `--model-id` for Big Love or another Qwen manifest. Optional controls
include `--true-cfg-scale`, `--sigmas 1 0.8 0.4`, `--no-kv-cache`, and repeatable
`--lora ID:STRENGTH`. The shared runtime options include device, dtype, cache
directory, and `--local-files-only`. The default CUDA dtype is BF16. Explicit
CPU runs default to float32.

The worker uses the same `CACHE_LOCATION/hub`, Hugging Face credentials, output
folders, and queue settings as the other transformer families. The environment
checker now verifies the Qwen pipeline and single-file loader imports. Start
the web app and worker using the existing launcher or services; restart existing
processes after installing this workflow so they load the new registry.

## Validation

```bash
IMAGE_GENERATOR_ACCESS_PASSWORD= .venv/bin/python -m unittest discover \
  -s tests -p 'test_qwen21_execution.py' -v
IMAGE_GENERATOR_ACCESS_PASSWORD= .venv/bin/python -m unittest discover \
  -s tests -p 'test_ui_contract.py' -v
```

The Qwen execution tests use synthetic pipeline components with real Flask
routes, settings persistence, SQLite queue, worker, PNG saving, and download
routes. They cover model switching and replacement, multiple signed adapters,
cleanup after failures, custom/repository schedules, live progress, CFG, RGBA
preservation, seed wraparound, invalid requests, and a CLI dry run without model
libraries. They do not establish image quality or checkpoint compatibility.

For manual preview without weights, use the fixture catalog and open
`/generate/fixture-qwen_2_1`. Enter a prompt, try `1, 0.8, 0.4` under custom
sigmas, and preview the typed values. Generation requires real assets.

On the GPU host, a small integration smoke run can use a real checkpoint at
512 × 512, two steps, one image, and seed 42. Compare no adapter, an enabled
compatible adapter, then no adapter again using the same workflow instance.
Check that metadata records each selection and the pipeline is reused. This is
an execution check, not a recommended quality preset.

The local Big Love checkpoint and Turbo adapter passed that GPU smoke check
at strength -0.1. The adapter changed the image, and disabling it restored the
identical seeded base output across all RGBA channels. These outputs are under
`Images/Qwen21_smoke/`; the two-step preset is not an image quality evaluation.

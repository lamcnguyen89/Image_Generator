# SDXL backend and command-line runner

The SDXL workflow supports text-to-image generation with the catalog's pretrained
base model or local `.safetensors` checkpoints. Flask remains read-only; the CLI
and Python API execute generation directly, without a background queue.

## First image

After [setting up the project environment](Setup.md), run from the project root:

```bash
.venv/bin/python scripts/generate_sdxl.py \
  --prompt "A mountain lake at sunrise" --seed 42 --no-loras
```

This selects the catalog entry `model_a4278bb0-94f4-4960-ada0-39ca89cec194`
(Stable Diffusion XL Base 1.0), uses its resolved settings and default scheduler,
and generates one image. CUDA device 0 and float16 are the runtime defaults.
The runner prints the PNG path, JSON metadata path, and actual seed.

Validate the request first without loading model libraries or writing files:

```bash
.venv/bin/python scripts/generate_sdxl.py \
  --prompt "A mountain lake at sunrise" --seed 42 --no-loras --dry-run
```

`--dry-run` also works with Python's `-S` option and no installed dependencies.
It validates local weight paths when a local checkpoint or enabled adapter is
selected, but does not check pretrained cache availability or GPU access.

## Settings and model selection

Use `scripts/inspect_catalog.py` to find permanent model and LoRA IDs.
Select another checkpoint with `--model-id`; optionally supply `--root` to use
another model family directory. Script defaults are anchored to this checkout,
so invoking the script by absolute path also works outside the project.

```bash
.venv/bin/python scripts/generate_sdxl.py \
  --model-id model_058cd2e3-1853-4b84-b560-23b0d3083295 \
  --prompt "A watercolor mountain landscape" --seed 42 --no-loras
```

That NoobAI entry selects its required Euler sampler and applies the manifest's
`prediction_type: v_prediction` and `rescale_betas_zero_snr: true`. An incompatible
sampler override is rejected before importing model libraries or loading weights.

Settings resolve in workflow → model defaults → saved user → request order.
CLI options include `--width`, `--height`, `--steps`, `--guidance`,
`--negative-prompt`, `--seed`, `--clip-skip`, and `--num-images`.
`--seed random` overrides a saved fixed seed. `--num-images` defaults to 1,
even when a saved model setting requests a larger batch. Other omitted setting
options preserve saved values. CLI requests do not modify the JSON manifests.

Backend validation checks settings types/ranges, nonempty prompts, dimension
alignment, checkpoint requirements, local weight paths, and adapter selection.
Only `txt2img` is implemented. Unknown SDXL setting names fail explicitly.

### Schedulers

| Manifest/CLI sampler | Noise schedule | Behavior |
| --- | --- | --- |
| `model_default` | `default` | Recreate the loader's scheduler class with its original configuration. |
| `euler` | `default`, `normal`, `karras`, `exponential`, `beta` | EulerDiscreteScheduler with the selected sigma grid. |
| `euler_ancestral` | `default`, `normal`, `karras`, `exponential`, `beta` | Native Euler ancestral update; named curves use a grid adapter. |
| `dpmpp_2m`, `dpmpp_2m_sde` | `default`, `normal`, `karras`, `exponential`, `beta` | DPM++ multistep solver, order 2; the SDE variant uses seeded noise. |
| `dpmpp_3m`, `dpmpp_3m_sde` | `default`, `normal`, `karras`, `exponential`, `beta` | DPM++ multistep solver, order 3; the SDE variant uses seeded noise. |
| `uni_pc` | `default`, `normal`, `karras`, `exponential`, `beta` | Native UniPC multistep solver, order 2. |
| `heun`, `lms`, `dpm_2`, `dpm_2_ancestral`, `deis` | `default`, `karras`, `exponential`, `beta` | Native Diffusers schedulers with their supported curve flags. |
| `ddim`, `ddpm`, `pndm` | `default` | Native scheduler grids; PNDM skips PRK warmup. |

Choices, validation, and scheduler construction use the
[shared sampler registry](Samplers.md). `--sampler` can override the resolved
choice within a checkpoint's allowed sampling pairs. Mandatory
`requirements.scheduler_config` values preserve checkpoint prediction/training
requirements. Configuration that conflicts with a sampler's fixed algorithm,
solver order, supported training options, or explicitly selected noise schedule is
rejected. The UI filters noise schedules by sampler and checkpoint restrictions.
Definitions and backend flags come from the [shared schedule registry](Schedules.md).
Lustify GGWP v7 defaults to DPM++ 2M SDE Karras, represented as:

```json
"sampling": {"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"}
```

The equivalent CLI selection is `--sampler dpmpp_2m_sde
--sampling-noise-schedule karras`.

### LoRAs

Enabled saved links under `settings.loras` are used by default. `--no-loras`
replaces that list with an empty list for this request. Each `--lora` argument
enables an explicitly selected adapter; supplying any `--lora` arguments replaces
the saved list. Repeat the option to combine adapters:

```text
--lora LORA_ID:0.7 --lora ANOTHER_LORA_ID:0.4
```

Replace those placeholders with real SDXL adapter IDs. Without a strength suffix,
the adapter's resolved defaults/user strength is used. A checkpoint-specific
strength overrides that adapter-level value, and an explicit CLI strength takes
precedence because it replaces the saved link. Disabled adapters are not loaded
and do not require local weight files during request preparation.

The backend rejects cross-family selections and known model-variant mismatches.
An empty verified-checkpoint list means untested compatibility; an explicit
selection is allowed, and the actual loader may still reject incompatible weights.
Each selected adapter gets a distinct name and strength, without fusing weights.

Model positive prefixes and suffixes are applied to the submitted prompt.
`--include-trigger-words` explicitly inserts model and enabled LoRA trigger words,
deduplicated in order. Adapter prompt additions remain available in metadata;
include any desired adapter descriptions in your submitted prompt. Neither those
additions nor `recommended_generation` presets silently replace model settings.

## Runtime and downloads

Actual execution reads the project `.env`, preserving already-set environment
variables. It uses `HUGGINGFACE_TOKEN` or `HF_TOKEN` for repository access.
Credentials are not included in output metadata or dry-run JSON.

The default cache is `<CACHE_LOCATION>/hub`, matching the notebooks, or
`Models/hub` when `CACHE_LOCATION` is unset. Relative cache roots are anchored
to the project and `~` is expanded. `--cache-dir` supplies an exact hub directory.
Pretrained model components may download on the first run. `--local-files-only`
disables downloads, including configuration/components needed by local checkpoints.

`--device cpu` explicitly selects CPU execution and defaults to float32. Other
runtime controls are `--device cuda:0` and `--dtype float16|bfloat16|float32`.
CPU float16 is rejected. The CUDA path checks availability before downloading
weights. Missing dependencies, invalid settings, and generation failures exit
with status 1; argument syntax errors use argparse's status 2.

## Output metadata

The default output directory is `Images/`; override it with `--output-dir`.
UUID filenames preserve previous runs. Each PNG has a matching JSON containing
the catalog model ID and weight reference, resolved settings, final positive and
negative prompts, enabled adapters and actual strengths, actual per-image seed,
scheduler class/configuration, runtime versions, precision, device, timing, and
UTC creation time. Numeric seeds produce a recorded sequence for batches;
random seeds are sampled and recorded individually.

Metadata uses strict JSON. Library scheduler infinity/NaN sentinels are recorded
as the strings `"Infinity"`, `"-Infinity"`, or `"NaN"`. These describe the actual
scheduler configuration; the file is provenance, not a directly loadable manifest.
Actual results can still vary across hardware and library versions.

An error writing a PNG/JSON pair removes the incomplete pair. For multi-image
batches, already-completed pairs are retained if a later pair fails.

## Python API and pipeline ownership

```python
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation, run_generation

catalog = discover_catalog("Models/Model_Families")
request = prepare_generation(
    catalog,
    "model_a4278bb0-94f4-4960-ada0-39ca89cec194",
    overrides={"prompt": "A mountain lake at sunrise", "seed": 42, "loras": []},
)
outputs = run_generation(request, "Images", cache_dir="Models/hub")
print(outputs[0].image_path, outputs[0].metadata_path)
```

Preparation copies catalog data so later edits do not alter a prepared request.
Treat the prepared request as read-only. The CLI loads `.env`; API callers can
pass `token`, `cache_dir`, `device`, `dtype`, and `local_files_only` explicitly.

`run_generation()` owns and closes one workflow. For serialized repeated calls,
create an `SDXLWorkflow` instance, call `generate(request)`, save its
`GeneratedBatch` with `save_batch()`, and close the instance in `finally`.
The workflow reuses its pipeline until the model or local checkpoint fingerprint
changes, reconstructs the scheduler from the loader's configuration for every
request, and unloads earlier adapters before changing selections. Loading or
inference failures discard the pipeline so partially applied adapters are not
reused. Instances are intended for one caller at a time; queueing remains a later
application layer.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_sdxl.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

Tests use mocked pipelines and real PNG serialization to exercise loader calls,
schedulers, seeds, adapter strengths and reset behavior, failure cleanup,
metadata, and CLI behavior. The installed Diffusers pipeline import and scheduler
construction have also been checked without loading weights. CUDA was unavailable
in the implementation environment; real pretrained/checkpoint/adapter GPU runs
remain to be verified on the GPU host.

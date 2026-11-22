# FLUX.1-dev backend

The FLUX backend supports `txt2img` for the catalog's `flux_1_dev` variant.
It loads a local transformer `.safetensors` file and uses `base_model_id` for the
remaining components. With `weights: null`, it loads the complete pretrained
pipeline. Other FLUX architectures need their own implementation or extension.

## Runner

```bash
.venv/bin/python scripts/generate_flux.py \
  --prompt "A mountain lake at sunrise" --seed 42 --no-loras
```

The default model is `model_6e8d8d55-6a4b-4cf9-b91b-3a4a5beaab1f`.
Override it with `--model-id` and optionally `--root`. Inspect catalog IDs with
`scripts/inspect_catalog.py`. Add `--dry-run` to print the validated request
without importing Torch, Diffusers, Flask, or Pillow, writing files, or downloading
components. Dry runs still check selected local checkpoint and adapter paths.

Options include `--width`, `--height`, `--steps`, `--guidance`,
`--negative-prompt`, `--true-cfg-scale`, `--max-sequence-length`, `--seed`, and
`--num-images`. The runner defaults to one image. `--seed random` overrides a
saved fixed seed. Other omitted setting options preserve saved values.

`guidance_scale` and `true_cfg_scale` are distinct controls. Following the
notebook and installed Diffusers implementation, true classifier-free guidance
is active only when `true_cfg_scale > 1` and a negative prompt is supplied.
Metadata records whether it was active. `max_sequence_length` must be in 1–512,
or a smaller manifest limit; dimensions must satisfy the manifest alignment and
be multiples of 16.

The [shared sampler registry](Samplers.md) supplies form choices, settings-save
and request validation, and runtime scheduler construction. FLUX registers
`flow_match_euler` and the existing `model_default` Euler alias, both using
FlowMatchEulerDiscreteScheduler. The [schedule registry](Schedules.md) supplies
Default, Normal, Karras, Exponential, and Beta where checkpoint restrictions allow them.
Default preserves the component schedule. Mandatory manifest
scheduler configuration is applied to a fresh scheduler on every request from the
original loader configuration. Checkpoint `allowed_sampling` restrictions are
enforced; only registered FLUX sampling pairs are accepted. Sampling changes
reuse the loaded model without carrying scheduler state between requests.

## Adapters

Enabled saved model links are used by default. `--no-loras` disables all adapters
for the current request. Supplying `--lora ID[:STRENGTH]` replaces the saved list;
repeat the option to combine multiple named adapters. When the strength is
omitted, the adapter's resolved default/user strength is used. Explicit request
strengths take precedence over checkpoint-specific and adapter-level values.

Cross-family and known model-variant mismatches fail before loading weights.
Catalog compatibility assignments are preserved; actual tensor/loader
compatibility still needs verification, especially for entries with conflicting
original metadata. Failed loads discard the pipeline. Switching adapters unloads
earlier ones with `reset_to_overwritten_params=True`, preserving FLUX's handling
of adapters that overwrite parameters. Base weights are never fused.

Model positive prefix/suffix settings are applied. `--include-trigger-words`
explicitly inserts model and enabled adapter trigger words, deduplicated in order.
Adapter prompt additions and generation recommendations remain metadata and do
not silently modify the submitted prompt or model settings.

## Runtime and outputs

GPU defaults are `cuda:0` and bfloat16; CPU execution must be explicitly selected
with `--device cpu`, which defaults to float32. `--dtype` can select another
supported precision. CUDA availability and BF16 support are checked before
loading models. No CPU offload or quantization is enabled automatically.

The runner reads the project `.env`; already-set environment variables take
precedence. Tokens use `HUGGINGFACE_TOKEN` or `HF_TOKEN`. Cache layout is
`<CACHE_LOCATION>/hub`, or `Models/hub` by default. `--cache-dir` selects an exact
hub directory. `--local-files-only` prevents component downloads. Initial
pretrained component loading may otherwise download weights.

PNG/JSON pairs go under `Images/` or `--output-dir`, with unique `flux-seed-...`
names. Metadata records the catalog ID, transformer path, variant, resolved
settings, final prompts, enabled adapters and strengths, actual per-image seeds,
scheduler, runtime versions, precision, device, and timing. Tokens are omitted.
Catalog and user JSON files are never modified.

## Python API

```python
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation, run_generation

request = prepare_generation(
    discover_catalog("Models/Model_Families"),
    "model_6e8d8d55-6a4b-4cf9-b91b-3a4a5beaab1f",
    overrides={"prompt": "A mountain lake at sunrise", "seed": 42, "loras": []},
)
outputs = run_generation(request, "Images", cache_dir="Models/hub")
print(outputs[0].image_path)
```

For repeated serialized requests, retain a `FluxWorkflow` instance, call
`generate(request)`, save the returned batch with `save_batch()`, and close the
workflow in `finally`. A changed model or transformer file fingerprint reloads
the pipeline. API callers supply cache/token options explicitly; the runners
handle `.env` configuration. The web form can submit the same requests to the
dedicated worker; see [FLUX web execution](FLUX_UI_Execution.md).

## Verification

`python -m unittest discover -s tests -p 'test_transformer_workflows.py' -v`
checks loaders, guidance, adapters, pipeline reuse, failure handling, and CLI/output
behavior with mocked pipelines. Real pipeline imports and installed API signatures
are checked separately without model downloads. CUDA/model-weight generation
remains to be verified on the GPU host.

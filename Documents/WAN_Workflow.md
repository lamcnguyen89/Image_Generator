# WAN 2.2 I2V/T2V A14B backend

The WAN workflow supports `img2vid` for `wan_2_2_i2v_a14b` and `txt2vid` for
`wan_2_2_t2v_a14b`. Both use complete pretrained or local Diffusers pipelines
containing high-noise and low-noise experts. I2V uses `WanImageToVideoPipeline`
and requires a local starting image; T2V uses `WanPipeline` with no image input.
Other WAN variants need separate implementations or extensions.

The shipped checkpoints share one **Wan 2.2** catalog card through matching
`generation_group` metadata. The UI dropdown selects the concrete model; model
IDs, weights, saved settings, adapter compatibility, and queued requests remain
specific to each variant. The worker reuses components within a checkpoint and
reloads when the checkpoint/variant changes.

## Runner

```bash
.venv/bin/python scripts/generate_wan.py \
  --input-image Images/input.png \
  --prompt "A gentle breeze moves across the lake" \
  --frames 49 --seed 42 --no-loras
```

The default catalog model is `model_95f8fb96-fc3b-4cf3-8b6f-e95e8cb5209b`.
`--model-id` and `--root` select other catalog entries. Input/output path options
resolve relative to your working directory; default catalog/cache/output locations
are anchored to the project. The script works by absolute path from another
directory.

Add `--dry-run` to print a validated request without importing model or image
libraries, downloading weights, or writing outputs. Dry runs check that the input
path exists; image decoding, encoder availability, and loaded-model alignment are
checked during actual execution before inference.

Options include `--width`, `--height`, `--frames`, `--fps`, `--steps`,
`--guidance`, `--guidance-2`, `--negative-prompt`, `--seed`, and `--num-videos`.
The runner defaults to one video. `--seed random` overrides a saved fixed seed.
Other omitted settings preserve resolved model/user values.

Both guidance scales must enable CFG together (both > 1) or disable it together
(both <= 1). Frame counts must be `4*n + 1`, at least 5, and satisfy the manifest's
additional frame-count requirements. FPS must be positive; it changes playback
rate, not the number of generated frames. Sampling offers `model_default`, explicit
`uni_pc`, and `dpmpp_2m`, all with the `default` noise schedule. `model_default`
remains selected. Explicit bindings require the loader's UniPC flow scheduler.
UniPC reproduces that scheduler; DPM++ 2M uses its numerical update while retaining
the exact UniPC sigma grid and model timesteps, including flow shift and expert
boundary selection. Mandatory scheduler settings are reapplied per variation;
conflicts with a selected solver's fixed algorithm/order fail validation.

Use `--sampler uni_pc` or `--sampler dpmpp_2m` in the CLI, or choose the sampler
in either video generation form. Checkpoint `allowed_sampling` restrictions apply.

`--preserve-aspect-ratio` fits the source aspect ratio to the requested pixel area,
rounding dimensions to the loaded VAE/transformer's spatial alignment.
`--no-preserve-aspect-ratio` center-crops to the exact requested size. Both paths
honor the manifest's dimension constraints. The backend applies EXIF orientation
and converts input images to RGB. `--vae-tiling` / `--no-vae-tiling` control VAE
tiling; the shipped defaults enable it.

## Model loading and expert adapters

`weights: null` loads `base_model_id`. Local `weights` must point to a complete
Diffusers directory with `model_index.json` and declare `weights_component:
pipeline`. A single transformer checkpoint is rejected. Local directories are
loaded offline. The runtime verifies both experts, a valid expert boundary, and
the loaded model's spatial/temporal alignment. The VAE loads in float32 while the
other components use the selected precision, bfloat16 by default on CUDA.

Enabled saved model links are used unless `--no-loras` is specified. Repeating
`--lora ID[:STRENGTH]` selects multiple adapters and replaces the saved list.
Strength precedence is request → checkpoint link → adapter user/default setting.

Every enabled WAN LoRA must declare compatible variants in
`compatibility.model_variants`, or the legacy single `compatibility.model_variant`.
When present, the list is authoritative. This allows one adapter to declare both
`wan_2_2_i2v_a14b` and `wan_2_2_t2v_a14b`.

The UI's **Noise expert** selector chooses `high_noise` or `low_noise` per adapter.
Model LoRA links may specify `expert_target` to override the adapter's recorded
target. For example, `{"id": "adapter-id", "enabled": true, "strength": 0.8,
"expert_target": "low_noise"}` loads that selection into the low-noise expert.
The effective choice is frozen in the queued adapter settings and output
metadata. Missing/invalid targets and unsupported variants are rejected before
model loading; disabled adapters are ignored. Overrides do not rewrite adapter
metadata or weights.

Adapters load into `transformer` or `transformer_2` according to that metadata.
Each expert activates only its own adapter names and strengths. Previous adapters
are unloaded between requests, and failed loading/inference discards the pipeline.
Selecting an adapter does not apply its generation recommendations or prompt
additions. Trigger insertion is explicit through `--include-trigger-words`.

## Outputs and memory ownership

Outputs go under `Videos/` or `--output-dir`:

- `wan-seed-...mp4`: encoded video.
- Matching `.input.png` for I2V only: the exact resized/cropped conditioning image.
- Matching `.json`: settings, final prompts, model/variant, expert adapters and
  strengths, actual seed, original input path/SHA-256, oriented source size,
  conditioning size, actual pipeline arguments, scheduler, boundary, frame count,
  duration, versions, precision, device, and timing.

Variations are generated and saved sequentially. Decoded frames are released
before the next variation is generated. Unique names preserve previous runs.
A failed export/metadata write removes its incomplete MP4/PNG/JSON set;
already-completed earlier variations remain available.

FFmpeg availability is checked before loading weights. Encoding uses the installed
Diffusers/imageio FFmpeg path with `macro_block_size=1`, avoiding extra resizing
after model-aligned dimensions have been selected.

## Runtime and Python API

The runner shares SDXL/FLUX's `.env` token/cache behavior: `HUGGINGFACE_TOKEN` or
`HF_TOKEN`, `<CACHE_LOCATION>/hub` or `Models/hub`, and explicit `--cache-dir`.
`--local-files-only` disables repository downloads. `--device cpu` explicitly
selects CPU execution and defaults to float32. GPU defaults are CUDA 0 and BF16;
the runtime checks device access and BF16 support. There is no automatic CPU
offload or quantization. API callers pass cache/token options explicitly.

```python
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation, run_generation

request = prepare_generation(
    discover_catalog("Models/Model_Families"),
    "model_95f8fb96-fc3b-4cf3-8b6f-e95e8cb5209b",
    inputs={"input_image": "Images/input.png"},
    overrides={"prompt": "A gentle breeze moves across the lake", "num_frames": 49,
               "num_videos_per_prompt": 2, "seed": 42, "loras": []},
)
outputs = run_generation(request, "Videos", cache_dir="Models/hub")
print(outputs[0].video_path, outputs[0].conditioning_path)
```

For T2V, select the catalog model whose variant is `wan_2_2_t2v_a14b`, pass
`mode="txt2vid"`, and omit `inputs`. `SavedVideo.conditioning_path` is `None` for
these results. T2V uses the configured dimensions directly, with the same loaded
VAE/transformer alignment checks, rather than fitting a source aspect ratio.
Local T2V directories must declare `_class_name: "WanPipeline"` in
`model_index.json`. Both T2V experts must have 16 input/output latent channels.

Preparation infers the mode from the selected model's declared first mode. A retained
`WanWorkflow` instance can be reused for serialized requests. Use
`generate_batches(request)` and save each returned batch with `save_batch()`;
release the batch before asking for the next variation. `generate(request)` is
convenient for one video and rejects multi-video requests. Always close a retained
workflow in `finally`. See [web execution](WAN_UI_Execution.md) for the shared
mode selector, queue, and worker.

## Verification

`python -m unittest discover -s tests -p 'test_transformer_workflows.py' -v`
covers validation, expert targeting, FP32 VAE loading, preprocessing, streaming,
output cleanup, and runners with mocked pipelines. Real FFmpeg encoding/decoding
has also been checked with synthetic frames. Full WAN inference and adapter
compatibility require verification on the GPU host; CUDA is inaccessible in this
implementation environment.

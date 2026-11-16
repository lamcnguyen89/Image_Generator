# MiniMax H3 workflow

MiniMax is registered as the `minimax` workflow alongside SDXL, FLUX, and WAN.
It supports the local ComfyUI-format H3 transformer, its video VAE and Qwen3-VL
text encoder, text-to-video and image-to-video conditioning, and multiple model-only H3 LoRAs.
It uses native ComfyUI Python nodes directly; no ComfyUI web server is needed.

The implementation follows the existing `MiniMax_Video_Gen.ipynb` generation
path, including the instance-local quantized-parameter transfer guard. Your
notebook, checkpoint/adapter IDs, and saved user settings remain independent.

## Forms

Run `.venv/bin/python main.py`, then choose **Configure generation** on MiniMax H3,
or open:

<http://127.0.0.1:5000/generate/model_b065ec3d-cd27-4cb3-bcab-48797bee8b6b>

The shared UI contract renders H3-specific controls:

- Dimensions aligned to 32 pixels and any stricter checkpoint requirement.
- Frame counts `17*n + 5`, constrained by the manifest and the native 3600-frame
  ceiling. This rule is independent of WAN's `4*n + 1` constraint.
- Native 24 FPS and CFG 1, rendered read-only and validated again in Python.
- `res_multistep` sampling with the `simple` schedule and BasicGuider.
- A choice of text-to-video or image-to-video. Only image-to-video requires a
  source image and exposes source-aspect preservation; text-to-video uses the
  requested dimensions directly.
- Variation count, seed, and VAE tiling.
- Multiple H3 adapter strengths without separate high/low-noise expert groups.

Negative prompting is unavailable in this BasicGuider implementation. Supplying a
nonempty negative prompt, changing fixed FPS/CFG, or selecting an unsupported
sampler is rejected rather than silently ignored. Native audio is not decoded
or exported: this adapter produces silent MP4s, matching the notebook.

Web submission queues generation for the shared worker; see
[web execution](MiniMax_UI_Execution.md). **Preview request** checks form values
without running generation or saving uploads/settings. To try the form without
local weights or CUDA:

```bash
IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families .venv/bin/python main.py
```

Open <http://127.0.0.1:5000/generate/fixture-minimax>, enter a prompt, select an
image, and click **Preview request**. The fixture adapter starts enabled at 0.7.

## CLI and Python generation

First validate the actual catalog request without importing model libraries or
downloading components:

```bash
.venv/bin/python scripts/generate_minimax.py \
  --input-image Images/input.png \
  --prompt "A gentle camera movement across the scene" \
  --frames 124 --seed 42 --dry-run
```

The default model ID is your MiniMax H3 entry. `--model-id` and `--root` select
another catalog entry/library. The manifest's enabled adapter links are used
unless replaced with repeated `--lora ID:STRENGTH` arguments or `--no-loras`.
`--include-trigger-words` is opt-in; adapter recommendations are not merged.

For generation, run `bash scripts/setup.sh` (append `--dev` for notebook tooling).
It installs all four families into the shared `.venv`, including ComfyUI's
dependencies, while preserving the pinned platform-specific PyTorch/torchvision
builds. It fetches a pinned native-H3 checkout at `build/ComfyUI` when absent and
preserves existing checkouts. The notebook setup cell is no longer required for
application setup. See [environment setup](Setup.md) for prerequisites and checks.
Use `--comfyui-dir /path/to/ComfyUI` for another checkout when running the CLI.
This backend currently uses ComfyUI's CUDA device 0 and checkpoint-managed
precision. Omit `--dtype`; CPU generation is not implemented.

After setup, omit `--dry-run` from the command above to generate. Runtime
dependencies are imported only then. No setup or model download occurs during
catalog browsing, form previews, or dry runs.

The VAE and text encoder are declared in the model's `components` object:

- `components.vae.weights` resolves relative to the model manifest and must be
  the H3 video VAE. The checkpoint's architecture metadata and the VAE's header
  are checked before loading tensors.
- `components.text_encoder.weights` can name a local safetensors file, also
  relative to the manifest.
- Alternatively, the encoder may use the supplied `Comfy-Org/MiniMax-H3` repo
  reference and one of its documented BF16, INT8, or NVFP4 Qwen3-VL filenames.
  A matching local file under the family's `text_encoders/` directory is preferred.
  Otherwise, the Hugging Face cache is used, with a download on first generation
  unless `--local-files-only` is supplied. Existing `.env` cache/token handling
  applies; credentials are not written to output metadata.

The shared preparation/execution API also supports MiniMax:

```python
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation, run_generation

request = prepare_generation(
    discover_catalog("Models/Model_Families"),
    "model_b065ec3d-cd27-4cb3-bcab-48797bee8b6b",
    overrides={"prompt": "A gentle camera movement", "seed": 42},
    inputs={"input_image": "Images/input.png"},
)
outputs = run_generation(request, "Videos", comfyui_dir="build/ComfyUI")
```

For text-to-video, pass `mode="txt2vid"` to `prepare_generation()` and omit
`inputs`. The model manifest must declare `txt2vid` in `supported_modes` (the
shipped H3 manifest and template declare both modes). Image-to-video remains the
default and still requires `input_image`. Both modes use the native
`MiniMaxH3ImageToVideo` node; text-to-video omits its optional `first_frame` input.

Workflow instances cache the base model, encoder, and VAE by file fingerprints.
Adapter changes always start from the unpatched base; a zero-strength adapter
is skipped. The loader checks that nonzero adapters actually add model patches.
File replacements reload components. Failed loading, inference, or export
releases the instance's pipeline; CLI execution closes it on completion.
Calls must be serialized, as with the other workflows.

Variations run and save sequentially rather than retaining multiple decoded
videos. Outputs use `minimax-seed-<seed>-<uuid>.mp4` and `.json` metadata under
`Videos/`, with a matching `.input.png` only for image-to-video. Text-to-video
records `conditioning_image: null` and no source-image metadata.
Metadata includes the request snapshot,
component references and resolved paths, fingerprints, conditioning dimensions,
input hash, actual seed, runtime/ComfyUI revision, and absence of exported audio.
The shared exporter streams frames through ImageIO/FFmpeg without importing
Diffusers, and continues to support WAN outputs.

## Catalog initialization and adapters

MiniMax's existing family templates participate in automatic manifest creation.
No special-case discovery code is needed. New LoRAs must explicitly record
`compatibility.model_variant: "minimax_h3"` before selection/loading. Initialization
leaves unknown compatibility unset, so new adapters may appear disabled until
configured. Checkpoint verification remains distinct from declared compatibility;
no successful inference claim or verification ID is added automatically.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_minimax.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

MiniMax tests cover fixture form rendering and forged fixed controls, preparation
and stdlib-only CLI dry runs, architecture/component validation, native node
arguments, model reuse, adapter stacking/reset, zero-strength adapters, file
replacement, failure cleanup, quantized transfers, streaming, metadata, naming,
and encoder cache/offline options. Native ComfyUI execution is mocked in these
tests; they do not certify full H3 GPU inference or adapter quality. Successful
loading/generation on the actual GPU must be checked separately.

Primary runtime references:
[ComfyUI H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3),
[native H3 conditioning](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py),
[official I2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json).

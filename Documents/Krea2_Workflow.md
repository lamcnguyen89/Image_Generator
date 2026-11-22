# Krea 2 RAW workflow

The `krea2` workflow implements `txt2img` for `model_variant: krea_2_raw`,
following the working [Krea 2 notebook](../Models/Model_Families/Krea2/Krea2_Image_Gen.ipynb).
The local checkpoint is loaded with `Krea2Transformer2DModel.from_single_file`;
`Krea2Pipeline` loads the remaining components from `krea/Krea-2-Raw`, explicitly
using `is_distilled=False`. With `weights: null`, a manifest can load the full
pretrained RAW pipeline. Turbo and two-stage generation are not implemented.

```bash
.venv/bin/python scripts/generate_krea2.py \
  --prompt "A mountain lake at sunrise" --seed 42 --no-loras
```

The default catalog entry is `model_43359779-6bc0-4f45-8b70-39a2edff93fa`.
Its preset matches the notebook: 1024×1024, 52 steps, guidance 3.5, a 512-token
budget, and seed 42. The notebook's Vérité realism adapter is selected at 0.8;
`--no-loras` disables it for the current request. The notebook remains independent.

Use `--dry-run` to validate settings and local asset paths and print the request
without importing GPU libraries or downloading components. Other options include
`--root`, `--model-id`, `--width`, `--height`, `--steps`, `--guidance`,
`--negative-prompt`, `--max-sequence-length`, `--num-images`, `--seed random`,
`--lora ID[:STRENGTH]` (repeatable), and `--include-trigger-words`.

Krea 2 uses `cond + guidance_scale * (cond - uncond)`. Guidance 0 disables CFG;
negative prompting applies above 0. There is no `true_cfg_scale` setting.
Dimensions must be positive multiples of 16 (and satisfy checkpoint alignment).
The sequence budget is 1–512, or a smaller manifest limit. Sampling supports
`flow_match_euler` and `er_sde` through the [shared sampler registry](Samplers.md).
`model_default` retains the existing Euler behavior. The
[schedule registry](Schedules.md) supplies Default, Normal, Karras, Exponential, and Beta,
filtered by checkpoint restrictions. Default uses the component repository's
Krea 2 resolution-aware schedule; named curves transform that grid. Manifest
scheduler configuration applies on each request. `requirements.scheduler_class` identifies this
schedule source (`FlowMatchEulerDiscreteScheduler`), rather than requiring the
Euler solver. The checkpoint's `requirements.allowed_sampling` further restricts
the choices shown in the form and accepted by the backend.

Big Love KreaEdit3 defaults to ER_SDE and permits Euler as an alternative. ER_SDE
uses third-order corrections with seeded noise for each image. Its first sigma
is moved slightly below 1 to avoid the singular pure-noise endpoint. This changes
sampling behavior; it does not promise pixel-identical output to a ComfyUI
workflow with different sigmas, conditioning, precision, or random streams.

Multiple LoRAs load under independent names and strengths, without fusing base
weights. Changes unload previous adapters with Krea 2's argument-free unload
method. Adapters are placed on the selected device. Cross-family and declared
variant mismatches fail before runtime loading. Adapter loading or inference
failures discard the pipeline. Repeated requests reuse the checkpoint; changed
checkpoint file fingerprints trigger a reload.

CUDA defaults to `cuda:0` and bfloat16. Explicit CPU execution defaults to
float32. The CLI and worker read `.env`, use `HUGGINGFACE_TOKEN` or `HF_TOKEN`,
and cache components in `<CACHE_LOCATION>/hub` (default `Models/hub`).
`--cache-dir` selects an exact cache directory; `--local-files-only` prevents
downloads. Repository access must be granted to the token's account even when
using local transformer weights, because supporting components are still needed.

PNG/JSON pairs are written under `Images/` or `--output-dir`, with unique
`krea2-seed-<seed>-<uuid>` filenames. Sidecars include resolved prompts/settings,
model identity, selected adapters, actual seeds, scheduler configuration, CFG and
negative-prompt status, runtime versions, device, precision, and timing.

The standard Python API also works:

```python
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation, run_generation

request = prepare_generation(
    discover_catalog("Models/Model_Families"),
    "model_43359779-6bc0-4f45-8b70-39a2edff93fa",
    overrides={"prompt": "A mountain lake at sunrise", "loras": []},
)
outputs = run_generation(request, "Images", cache_dir="Models/hub")
```

See [web execution](Krea2_UI_Execution.md) for the form and queue.

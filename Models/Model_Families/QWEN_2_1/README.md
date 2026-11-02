# Qwen-Image 2.1

The JSON files follow the existing catalog schema. Checkpoints use
`model.defaults.json` and `model.user.json`; adapters use `lora.defaults.json`
and `lora.user.json`. Reusable `*.defaults.example.json` and
`*.user.example.json` pairs live directly under `checkpoints/` and `loras/`.
Those examples initialize new asset folders; they are not registered models
and are not a runtime inheritance layer.

Configured assets:

- `checkpoints/Qwen_Image_2_1/`: the official Hugging Face pipeline, with
  `weights: null` and `base_model_id: "Qwen/Qwen-Image-2.1"`.
- `checkpoints/Big_Love_Qwen_2/`: the existing local BF16 transformer. The
  Hugging Face repository supplies the matching supporting components.
- `loras/qwen21_turbo_rank256/`: the local adapter's manifest pair. Its weights
  remain at `loras/qwen21_turbo_rank256.safetensors`, so the manifest references
  `../qwen21_turbo_rank256.safetensors`. The notebook's existing path still works.

Each asset has a permanent ID. Keep it when editing metadata or settings.
User files initially contain `{"schema_version": 1, "settings": {}}`; put only
personal setting overrides there. Removing an override restores the asset's
default. Lists such as `loras` replace the entire default list.

## Suggested starting settings

| Setting | Qwen 2.1 base | Big Love Qiara1 |
| --- | --- | --- |
| Width × height | 2048 × 2048 | 2048 × 2048 |
| Inference steps | 40 | 40, conservative unverified baseline |
| `true_cfg_scale` | 1.0 | 1.0 |
| Negative prompt | Empty | Empty |
| `use_kv_cache` | `true` | `true` |
| `sigmas` | `null` | `null` |
| Images per prompt | 1 | 1 |
| Seed | 42 | 42 |
| LoRAs | None | Turbo linked but disabled, strength 0.3 |

The base settings follow the [official model card](https://huggingface.co/Qwen/Qwen-Image-2.1)
and [Diffusers pipeline documentation](https://huggingface.co/docs/diffusers/en/api/pipelines/qwenimage21).
Both dimensions must be divisible by 32. Use 1024 × 1024 for a quicker preview
or 1536 × 1536 as a smaller square above two megapixels. These smaller square
sizes are practical suggestions, not the model card's quality preset.

The [Big Love notes](checkpoints/Big_Love_Qwen_2/Big_Love_Qiara1.md) recommend
at least two megapixels and describe a custom sampler with extended 6/10-step
sigma schedules. Those custom schedules are not included locally; 40 steps is
the notebook's conservative starting point, not an author-verified optimum.
The notes' 1184 × 1776 and 1664 × 2512 examples have heights incompatible with
the Diffusers pipeline's 32-pixel grid. Nearby alternatives are 1184 × 1792 and
1664 × 2528.

## Qwen-specific controls

- Use `true_cfg_scale`, rather than copying Krea2's `guidance_scale: 3.5`.
  The default 1.0 disables CFG. To experiment with negative prompting, try
  2.0 with a nonempty negative prompt; this is a tuning suggestion, not a
  recommended base preset, and adds computation.
- `sigmas: null` leaves schedule selection to the pipeline. A descending list
  overrides that schedule and determines the number of steps. Exclude the
  terminal zero. Leave `scheduler_config` empty to retain repository settings.
- Do not copy Krea2/FLUX `max_sequence_length`, SDXL `clip_skip`, or SDXL
  sampler presets: they are not controls of the Qwen 2.1 workflow or notebook.
- Qwen 2.1 also supports image-conditioned editing and RGBA generation. The
  workflow and notebook implement text-to-image, so `supported_modes` declares
  only `txt2img`. For transparency, describe a transparent RGBA image in the
  prompt and preserve the PNG alpha channel; there is no separate transparency
  switch in this pipeline. Editing needs an implementation that passes
  reference images to the pipeline's `image` argument.

## LoRA strengths and Turbo sampling

The generic adapter template suggests strength 0.7 as a starting point; follow
each adapter author's instructions and compare with a fixed seed. Only adapters
trained for Qwen **2.1** are assumed candidates. The local Turbo adapter has a
loading/inference smoke check recorded against Big Love Qiara1; its training
provenance remains unknown.

For the local Turbo file, 0.3 is an optional Big Love adjustment suggested
within the checkpoint notes' -0.3 to 0.5 range. It starts disabled. Its exact
training source is unknown, so `recommended_generation` is empty: selecting
this adapter does not establish an eight-step acceleration preset.

The official `Qwen/Qwen-Image-2.1-Turbo` repository is a separate model. Its
[pipeline configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json)
provides this eight-step schedule:

```json
[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
```

Use that repository's supplied configuration when running the official Turbo
model. Do not assume the same schedule fits an unverified adapter or custom
checkpoint. Catalog LoRA recommendations are advisory and never automatically
merged into model settings.

## Workflow and web UI

`workflow: "qwen_2_1"` uses `Qwen21Workflow` in
`src/image_generator/workflows/qwen21.py`. The catalog places **QWEN 2.1**
between FLUX and Krea2. Both checkpoint cards open the shared generation form,
with true CFG, KV caching, custom sigma schedules, and multiple weighted LoRAs.
Signed adapter ratios are accepted by the form, saved preferences, and backend.
The shared notebook/workflow loader splits ComfyUI fused gate/up LoRA weights
into Diffusers projections and checks all adapter targets before loading them.

Generate submits to the same durable queue as the other image workflows.
The worker reports loading, LoRA setup, denoising, decoding, and saving; image
downloads include PNGs and JSON metadata with actual seeds and sigma step counts.
PNG alpha channels are preserved. Save/reset model settings and shared adapter
defaults use the existing settings controls and user files.

See the [workflow and validation guide](../../../Documents/Qwen21_Workflow.md)
for CLI use, fixture forms, and tests. `QWEN_2_1_Image_Gen.ipynb` remains an
independent experiment and reads its Python configuration rather than these JSON
files. Transfer chosen settings to that cell when testing the notebook.

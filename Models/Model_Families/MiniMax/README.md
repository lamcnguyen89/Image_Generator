# MiniMax H3 catalog configuration

The checkpoint manifests are in `checkpoints/minimax_h3/`. They register the local
`minimaxH3_comfy_3086249.safetensors` checkpoint and link the included
`Minimax_H3_Unlocked_V2.1.safetensors` LoRA at strength 1.0. The adapter's own
manifests are in `loras/minimax_h3_unlocked_v2_1/`. Each weight file is stored beside its
defaults and user manifests in its asset subfolder.

`model.defaults.json` and `lora.defaults.json` contain permanent IDs, asset
metadata, and baseline settings. Keep their IDs when renaming assets. Each
corresponding user file starts with `{"schema_version": 1, "settings": {}}` and
may override settings only. Objects merge recursively; lists, including `loras`,
replace the entire lower-priority list. LoRA links use registered adapter IDs,
`enabled`, and an optional checkpoint-specific `strength`.

The H3 defaults mirror the notebook: 832×480 output budget, source aspect-ratio
preservation, 124 frames at 24 fps, 20 `res_multistep` sampling steps with the
`simple` schedule, CFG 1 / `BasicGuider`, and tiled video decoding. Dimensions
must be multiples of 32; valid frame counts satisfy `num_frames % 17 == 5`.
The negative prompt is empty because this BasicGuider workflow does not use it.
Input images are supplied for each generation, rather than saved as model defaults.
The checkpoint supports both `img2vid` and `txt2vid`. Choose the mode in the
generation UI. Text-to-video uses the requested width and height directly,
requires no source image, and saves an MP4 and metadata without a conditioning PNG.
Image-to-video still requires a source image and offers aspect preservation.

The `components` block records the included video VAE and the required
Comfy-Org Qwen3-VL text encoder. `components.vae.weights` is relative to the
checkpoint manifest directory; the text encoder uses a Hugging Face repository
ID and repository-relative filename. These are MiniMax-specific metadata:
the current catalog preserves them, but its `asset_paths` property resolves
only top-level `weights`, `notes_file`, and `preview_image`.

The notebook still uses its Python settings and does not read these manifests.
The application registers a `minimax` workflow for these manifests. It provides
an H3-specific generation form/request preview and a catalog-backed CLI using
native ComfyUI Python nodes. Component loading, multiple adapters, quantized
transfer protection, H3 validation, and silent video export are implemented.
See [MiniMax workflow setup and tests](../../../Documents/MiniMax_Workflow.md).
The web interface can queue either mode for the shared generation worker; see
[MiniMax web execution](../../../Documents/MiniMax_UI_Execution.md).

The adapter's 200 target matrix shapes match the local checkpoint. This does
not establish successful full-video inference, so `verified_checkpoint_ids`
remains empty. Unrecorded source details, hashes, and trigger words are left
unset rather than inferred from filenames.

## Templates

- `checkpoints/model.defaults.example.json`
- `checkpoints/model.user.example.json`
- `loras/lora.defaults.example.json`
- `loras/lora.user.example.json`

The defaults templates have empty prompts, trigger words, and adapter links,
placeholder IDs and weight filenames, and valid H3 numerical defaults. User
templates contain an empty settings object. Example files are not registered
assets. For manual creation, copy a pair into an asset subfolder, remove
`.example` from the names, generate a permanent UUID, and set the weight path.
Review relative component paths if the folder layout changes. Automatic
initialization uses these templates to fill missing manifests in new asset
folders and assigns IDs without copying adapter links or compatibility claims.

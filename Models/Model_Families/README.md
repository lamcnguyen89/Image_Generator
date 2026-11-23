# Model family layout

Each selectable model and each LoRA has its own folder and two JSON files.
The nine SDXL checkpoint folders now contain populated `model.defaults.json`
files and initial `model.user.json` override files. SDXL's ChelXL LoRA also has
populated defaults and an initial user override file. The FLUX checkpoint also
has populated defaults and an initial user override file. All 18 FLUX-directory
LoRA folders have populated defaults and initial user override files. All are
classified as FLUX.1-dev by user request, with original metadata observations
retained in compatibility notes. WAN's one checkpoint and six LoRAs also have
populated default/user JSON pairs, with unknown targets left explicit. Reusable
checkpoint and LoRA examples are available directly in the `checkpoints/` and
`loras/` directories of all three architecture families.
The notebooks still use their existing Python settings and do not read these
JSON files yet.

Qwen-Image 2.1 now has checkpoint and LoRA manifest pairs plus reusable family
examples. See [its configuration guide](QWEN_2_1/README.md) for Qwen-specific
guidance, sigma schedules, resolution suggestions, and the current integration
status.

```text
Model_Families/
├── SDXL/
│   ├── checkpoints/
│   │   ├── bigLust_v16/
│   │   │   ├── BigLust_Image_Gen.ipynb
│   │   │   ├── bigLust_v16.safetensors
│   │   │   ├── bigLust_Notes.md
│   │   │   ├── model.defaults.json
│   │   │   └── model.user.json
│   │   ├── noobaiXLNAIXL_vPred10Version/
│   │   │   ├── NoobAIXL_Image_Gen.ipynb
│   │   │   ├── noobaiXLNAIXL_vPred10Version.safetensors
│   │   │   ├── noobAIXL_Notes.md
│   │   │   ├── model.defaults.json
│   │   │   └── model.user.json
│   │   ├── sdxl_base_1_0/
│   │   │   ├── SDXL_Image_Gen.ipynb
│   │   │   ├── model.defaults.json
│   │   │   └── model.user.json
│   │   └── <other SDXL checkpoint folders>/
│   └── loras/
│       └── ChelXL/
│           ├── ChelXL.safetensors
│           ├── ChelXL_Notes.md
│           ├── lora.defaults.json
│           └── lora.user.json
├── FLUX/
│   ├── Flux_Image_Gen.ipynb
│   ├── checkpoints/
│   │   └── fluxNSFWUNLOCKED_v30FP16/
│   │       ├── fluxNSFWUNLOCKED_v30FP16.safetensors
│   │       ├── fluxNSFWUnlocked_Notes.md
│   │       ├── model.defaults.json
│   │       └── model.user.json
│   └── loras/
│       └── <existing LoRA folders>/
└── WAN/
    ├── WAN_Video_Gen.ipynb
    ├── checkpoints/
    │   └── wan2_2_i2v_a14b/
    │       ├── model.defaults.json
    │       └── model.user.json
    └── loras/
        └── <existing LoRA folder>/
            ├── <existing weights and notes>
            ├── lora.defaults.json
            └── lora.user.json
```

## Organization

- The architecture folders are `SDXL/`, `FLUX/`, and `WAN/`.
- SDXL contains the base model and the former BigLust, IllustriousXL, LustifyXL,
  NoobAIXL, PonyDiffusionV6XL, PyrosSDXL, juggarnautXL, and waiIllustrious models.
- Each local checkpoint and its notes live under
  `checkpoints/<checkpoint filename without its extension>/`.
- Each SDXL notebook lives alongside its own checkpoint and JSON pair. The
  FLUX and WAN notebooks retain their existing locations.
- SDXL LoRAs live under `SDXL/loras/`, including `ChelXL/`. Sharing an architecture
  folder does not establish compatibility with every checkpoint; model-specific
  compatibility will be defined when the JSON contents are designed.
- SDXL's `sdxl_base_1_0` and WAN's `wan2_2_i2v_a14b` folders represent the
  pretrained models already selected by their notebooks. These folders contain
  configuration files, not downloaded model weights. Hugging Face cache
  behavior is unchanged.
- Weight filenames are preserved and remain excluded from Git.

## Notebook paths

Checkpoint paths remain relative to the architecture family's `checkpoints/`
directory and include the checkpoint subfolder. LoRA paths remain relative to
that family's `loras/`. All SDXL notebooks now use `model_family = "SDXL"`.
They find the project root from their nested working directory. Absolute asset
paths continue to work. Paths shown in the notebook descriptions and code
comments are relative to the family directory.

No generation logic or scheduler behavior was changed. In particular, NoobAI's
existing scheduler handling remains in its notebook until it is moved into a
shared SDXL workflow in a later step.

## SDXL checkpoint JSON files

`model.defaults.json` contains:

- `schema_version`, a permanent `id`, `display_name`, and `description`.
- `family` and `workflow`, both `sdxl` for these checkpoints.
- `source_url`, `source_model_id`, and `source_version_id` for provenance.
  CivitAI URLs and IDs were copied from local notes, without fetching the pages.
  The SDXL base entry references its official Hugging Face repository. Pony's
  source fields remain `null` because its local notes do not record an origin URL.
  Source IDs are strings when present.
- `weights`, `base_model_id`, `notes_file`, and `preview_image` for loading and
  presentation. Asset paths are relative to the JSON file. `weights: null` on
  SDXL base means to load the pretrained `base_model_id` instead of a local file.
- Optional `sha256`, currently `null`; weight fingerprints have not been computed.
- `supported_modes`, currently only `txt2img`, matching the implemented notebooks.
- `requirements`, containing dimension constraints and mandatory scheduler
  configuration. NoobAI additionally restricts sampling to Euler/default and
  specifies `v_prediction` and `rescale_betas_zero_snr: true`.
- `settings`, containing the prompt, negative prompt, positive prefix/suffix,
  trigger words, dimensions, image count, CFG, steps, seed, CLIP skip, sampling,
  and the checkpoint's `loras` list.

Numeric defaults start from the existing notebooks. Local recommendations were
used for Juggernaut's resolution, steps, CFG, and sampler; NoobAI's steps, CFG,
prompt additions, and scheduler; Pony's steps, sampler, and quality prefix;
Pyros's resolution; and WAI v17's resolution, steps, CFG, sampler, and prompt
additions. Other undocumented values retain the notebook defaults. No required
trigger words have been inferred from optional subject/style tags.

Sampling names are catalog values for the future SDXL workflow:
`model_default` preserves the loader's scheduler, `euler` identifies Euler,
`euler_ancestral` identifies Euler a, and `dpmpp_2m_sde` identifies DPM++ 2M SDE.
`noise_schedule: "default"` preserves the scheduler's default schedule. No Karras
schedule was inferred where the notes did not specify one. The future workflow
will translate these values into Diffusers scheduler construction.

All `clip_skip` values are `null`, preserving Diffusers SDXL's default prompt
layer. Pony's original Clip skip recommendation remains in its notes; it is not
copied as a numeric Diffusers override. All default `loras` lists are empty;
checkpoint links and selections will be added explicitly to user settings.

`model.user.json` starts as `{"schema_version": 1, "settings": {}}`. It is reserved
for user overrides. The planned resolver will merge nested objects by key and
replace lists as whole values. Removing an override restores its default;
`null` is an explicit value, such as a random seed. Linking a LoRA will add an
entry with `id`, `enabled`, and `strength` under `settings.loras`; inclusion means
linked, while `enabled` controls its saved selection for generation.

The generated `model_<UUID>` IDs are assigned once per checkpoint entry. Preserve
them when renaming or moving folders; filenames and display names are not IDs.

## FLUX checkpoint JSON files

The single local FLUX checkpoint, `fluxNSFWUNLOCKED_v30FP16`, has populated
`model.defaults.json` and `model.user.json` files. It uses the same metadata and
override structure as SDXL, with FLUX-specific fields and settings:

- `family` and `workflow` are `flux`; `model_variant` is `flux_1_dev`.
- `weights_component` is `transformer`. The local weight file supplies the
  transformer; `black-forest-labs/FLUX.1-dev` supplies its configuration and the
  remaining pipeline components, matching the existing notebook.
- `requirements` records a dimension multiple of 16, a sequence-length limit of
  512, and `FlowMatchEulerDiscreteScheduler`. `allowed_sampling` permits
  Flow Match Euler, ER SDE, Flow Match Heun, UniPC, DPM++ 2M, and DPM++ 2M SDE
  with Default, Normal, Karras, Exponential, and Beta noise schedules. Flow Match
  Euler with Default remains selected.
- Settings include separate `guidance_scale` and `true_cfg_scale` values,
  `max_sequence_length`, prompts, dimensions, seed, image count, sampling, and an
  empty `loras` list. There is no SDXL-style `clip_skip` setting.
- Source URL/version IDs and trigger words are copied from the local checkpoint
  notes. Missing preview and hash fields remain `null`.

Generation values preserve the notebook's defaults: 1024x1024, 50 steps,
`guidance_scale: 3.5`, `true_cfg_scale: 1.0`, and `max_sequence_length: 512`.
The notes' Euler recommendation is represented by the flow-matching Euler
scheduler rather than SDXL's Euler scheduler. The notes recommend 20+ steps;
the existing 50-step preset is retained. Their `CFG: 1` notation has not been
assumed to mean `guidance_scale: 1`. The original recommendation remains in the
notes until its source workflow can clarify the intended control.

The future workflow will enable true classifier-free guidance when
`true_cfg_scale > 1` and a negative prompt is provided, matching the notebook.
Trigger words are stored separately for explicit prompt insertion. No existing
notebook settings or checkpoint/LoRA associations were changed.

Reusable templates are stored directly in the checkpoint directory:

- [FLUX checkpoint defaults](FLUX/checkpoints/model.defaults.example.json)
- [FLUX checkpoint user overrides](FLUX/checkpoints/model.user.example.json)

These examples describe a FLUX.1-dev transformer checkpoint, not every FLUX
variant. Copy them into a new checkpoint folder, remove `.example` from the
filenames, assign a new permanent ID, and update paths and metadata. Replace or
remove the example user file's placeholder LoRA link; it does not register a real
adapter. A fresh user file can contain only `schema_version: 1` and an empty
`settings` object. Future catalog discovery should exclude `*.example.json`.

## WAN checkpoint and LoRA JSON files

WAN has populated default/user JSON pairs for its pretrained WAN 2.2 I2V A14B
checkpoint entry and all six adapters. Model defaults preserve the notebook's
numerical settings and record frame-count, dimension, input-image, and expert
requirements. LoRA defaults record exact variants and expert targets when the
local notes, filenames, or headers establish them; unknown targets remain `null`.

Examples are stored directly in `WAN/checkpoints/` and `WAN/loras/`. See the
[WAN configuration documentation](WAN/README.md) for field semantics, per-adapter
metadata, unresolved expert targets, and links to all four examples. No notebook
code, checkpoint/LoRA associations, or weight files were changed.

## FLUX-directory LoRA JSON files

All 18 folders under `FLUX/loras/` have populated default/user JSON pairs following
the agreed LoRA structure, with `compatibility.model_variant` added. Source URLs
are copied from local notes; available training model names and architecture
declarations come from safetensors headers. Live CivitAI pages could not be
retrieved, so undocumented recommendations remain empty. User override files
start with an empty `settings` object.

All 18 entries now have `family: "flux"` and
`compatibility.model_variant: "flux_1_dev"` by user request. Original metadata
observations for Cowgirl, Dog, Missionary, Master, and kera2unlocked are preserved
in their compatibility notes as the background to this explicit override. No
weight files were moved, verified-checkpoint IDs added, or checkpoint links created.

See the [FLUX LoRA configuration notes](FLUX/loras/README.md) for field meanings,
strength defaults, source gaps, and the compatibility review.

## SDXL LoRA JSON files

ChelXL is currently the only SDXL LoRA. Its `lora.defaults.json` follows the LoRA
example structure and contains a permanent `lora_<UUID>`, source URL and version
IDs copied from its local notes, relative weight and notes paths, and the
documented trigger word `cheldorado`. The optional positive prefix contains the
documented character description with the trigger kept separately.

`compatibility.trained_on` is copied from the weight file's `ss_sd_model_name`
metadata: `stablediffusionapi/pony-diffusion-v6-xl`. This is the recorded training
model name, not a newly verified download location. `verified_checkpoint_ids`
remains empty because no checkpoint compatibility test was performed.

The initial strength of `0.8` comes from the existing SDXL notebook's selected
LoRA strength, not a claimed creator recommendation. `recommended_generation`
is empty because the local notes do not specify generation settings. Optional
hash and preview fields remain `null`. Its `lora.user.json` starts with
`{"schema_version": 1, "settings": {}}` and is ready for user overrides.

No checkpoint links were added automatically, and notebooks still use their
existing Python settings rather than reading these files.

## SDXL example JSON files

The following files are templates, not registered assets:

- [Checkpoint defaults](SDXL/checkpoints/model.defaults.example.json)
- [Checkpoint user overrides](SDXL/checkpoints/model.user.example.json)
- [LoRA defaults](SDXL/loras/lora.defaults.example.json)
- [LoRA user overrides](SDXL/loras/lora.user.example.json)

Copy a pair into a new asset's subfolder and remove `.example` from the filenames.
Replace the placeholder ID with a new `model_<UUID>` or `lora_<UUID>` once, then
preserve that ID. Paths resolve relative to the copied JSON, not the examples'
current directory. Example weight files, trigger words, and numeric settings are
illustrative and are not recommendations for any existing asset.

The checkpoint user example links the example LoRA through `settings.loras`.
Replace that placeholder with the imported LoRA's actual ID. Its `enabled: false`
demonstrates a linked adapter that is not selected for generation. Do not copy
this example link into a new checkpoint unless the adapter exists; a fresh user
file can simply contain `{"schema_version": 1, "settings": {}}`.

LoRA defaults include source metadata, a weight path, optional notes and preview,
compatibility information, strength, trigger words, prompt additions, and optional
generation recommendations. `compatibility.verified_checkpoint_ids` records
confirmed compatibility; checkpoint links remain in `model.user.json` and are not
duplicated there. An empty verified list means no compatibility has been recorded,
not that the adapter has been proved incompatible. `recommended_generation` can
be `{}` when recommendations are unknown; it is intended for an explicit UI
action and should not silently overwrite generation settings when selecting LoRAs.

The LoRA user example shows editable strength and trigger words. Its settings
override the LoRA defaults. A checkpoint-specific saved strength then takes
precedence, followed by the current generation request. User files do not need
to repeat metadata or unchanged settings.

Catalog discovery looks for asset-subfolder `model.defaults.json` and
`lora.defaults.json` files and exclude `*.example.json`. Examples do not register
assets or change existing settings. Missing manifest pairs are initialized on
catalog load; see [automatic initialization](../../Documents/Catalog.md#automatic-manifest-initialization).

## Adding assets

Create a folder beneath the appropriate family's `checkpoints/` or `loras/`.
Keep that asset's weights, notes, and optional previews together. On the next
catalog load, missing `model.defaults.json` / `model.user.json` or
`lora.defaults.json` / `lora.user.json` files are initialized from family examples,
with new IDs, detected weight paths, and empty user overrides. Existing files
are preserved. Supply the weight path explicitly through `initialize_asset()`
when a folder has multiple candidates. Review the generated metadata and
settings; compatibility, WAN expert targets, and actual trigger words require
documented values. You can also prepare manifests manually. For SDXL and FLUX checkpoints, use
the appropriate family's example as a field template, assign a new permanent ID,
and start the user file with an empty `settings` object. For SDXL LoRAs, use the
LoRA example pair and populate documented fields, assigning a new permanent LoRA
ID. FLUX-directory LoRAs follow the same fields with a target `model_variant`;
record unknown or conflicting architectures explicitly. For WAN, use the WAN
checkpoint and LoRA examples, recording the correct variant and expert target.
Models
consisting of multiple components can keep those components together in one
model folder.

## Krea 2 RAW

Krea 2 has a [catalog configuration and family templates](Krea2/README.md),
a [reusable workflow](../../Documents/Krea2_Workflow.md), and a
[matching generation UI](../../Documents/Krea2_UI_Execution.md). The working
notebook stays independent. RAW guidance and its native Krea2 loader are used.

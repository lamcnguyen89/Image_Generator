# Catalog discovery and settings resolution

The catalog reads model and LoRA configuration without importing Torch,
Diffusers, or Flask, opening weights, or downloading models. Loading initializes
missing JSON manifests; existing files and saved settings are preserved.
It is the data layer for the planned web interface and generation workflows.

## Inspect the catalog

From the project root, using Python 3.11 (the project interpreter is
`.venv/bin/python`):

```bash
python scripts/inspect_catalog.py
python scripts/inspect_catalog.py --json
python scripts/inspect_catalog.py --id model_a4278bb0-94f4-4960-ada0-39ca89cec194
python scripts/inspect_catalog.py --root /path/to/Model_Families
```

The default command lists names, families, and permanent IDs. `--id` prints one
asset's manifests, resolved local paths, and effective settings as JSON. `--json`
prints this information for every asset. The default catalog root is relative
to the script, so it also works when invoked by absolute path from another
directory. A custom relative `--root` is relative to the working directory.

Invalid catalog data and unknown IDs exit with status 1. Validation errors name
the source file and field; `--json` returns an `errors` list. Discovery collects
errors across configured files. Assets without defaults that cannot yet be
initialized are skipped with a warning; they do not block configured assets. An existing,
empty root is a valid empty catalog; a missing root is an error.

## Python API

After installing the local package, or with `PYTHONPATH=src`:

```python
from image_generator.catalog import discover_catalog, CatalogValidationError

try:
    catalog = discover_catalog("Models/Model_Families")
except CatalogValidationError as error:
    for issue in error.issues:
        print(issue.path, issue.field, issue.message)
    raise

model = catalog.get_model("model_a4278bb0-94f4-4960-ada0-39ca89cec194")
settings = model.resolve_settings()
job_settings = model.resolve_settings(request_overrides={"num_inference_steps": 40})

print(model.display_name, model.defaults["workflow"])
print(model.asset_paths["weights"])  # None for this pretrained model
print(settings["sampling"])

# LoRAs have the same asset interface and resolve their settings separately.
for lora_id, lora in catalog.loras.items():
    print(lora_id, lora.resolve_settings()["strength"])
```

`catalog.models` and `catalog.loras` index assets by permanent ID. `get_model()`,
`get_lora()`, and `get_asset()` raise `KeyError` for an unknown ID. Each asset
exposes its kind, defaults manifest, user manifest, manifest paths, and resolved
local asset paths. Treat the source manifest dictionaries as read-only;
`resolve_settings()` returns an independent tree that callers can edit.

## Discovery and validation

Discovery registers these paths beneath the supplied root:

```text
<family>/checkpoints/<asset>/model.defaults.json
<family>/loras/<asset>/lora.defaults.json
```

Before registration, discovery initializes missing defaults/user files in asset
folders. Example files and unrelated JSON do not register assets. Malformed,
unreadable, or unsupported-version existing files are errors and are never
replaced. Use `discover_catalog(root, initialize_missing=False)` for strictly
read-only inspection; in that mode, absent user files mean no overrides in memory.

## Automatic manifest initialization

Each immediate folder under `<family>/checkpoints/` or `<family>/loras/` is an
asset candidate when it contains a defaults/user manifest, a `.safetensors` file,
or a Diffusers pipeline with `model_index.json`. Empty, hidden, and unrelated
folders are ignored. Weights stored directly in `checkpoints/` or `loras/` must
first be placed inside an asset subfolder.

For missing defaults, initialization reads the family directory's adjacent
`model.defaults.example.json` or `lora.defaults.example.json`. It assigns a new
permanent UUID, uses the asset folder name for display, and detects a single
local `.safetensors` file. WAN model templates instead detect a complete pipeline
in the asset folder or one immediate child; without local weights, WAN's
pretrained template repository is retained. Multiple candidate files/pipelines,
missing templates, and incomplete WAN checkpoints prevent automatic registration.
Catalog loading skips these unfinished assets and logs their paths and the
reason, allowing other models and LoRAs to remain available while a new family
is being set up. No manifests are created for skipped assets; they are retried
on the next load and appear once initialization succeeds. Malformed templates,
invalid existing manifests, and permission or other I/O failures remain errors.
An explicit `initialize_asset(...)` call still reports incomplete assets as
`CatalogValidationError` so upload handlers can request the missing information.

Numerical settings and pipeline requirements come from the template and need
review for the actual checkpoint. Illustrative prompts, trigger words, links,
source details, LoRA recommendations, and compatibility claims are cleared.
In particular, a new WAN LoRA has an unknown expert target until explicitly
configured. Family placement alone does not prove compatibility.

Missing user files are initialized as `{"schema_version": 1, "settings": {}}`.
The user example supplies the schema version when available; its illustrative
overrides and placeholder adapter links are not copied. Existing defaults need
no templates or local weights to receive a missing empty user file.

Files are published atomically without overwriting existing entries, including
when concurrent requests initialize the same asset. Repeated loads preserve IDs
and existing bytes. If a write fails, a later load can retry missing files.
Deleting a defaults file discards its identity; recreating it assigns a new UUID.

The same helpers are available for a future upload handler after it saves the
weights into the appropriate asset subfolder:

```python
from image_generator.catalog import initialize_asset, initialize_catalog

created_paths = initialize_asset(
    "Models/Model_Families/FLUX/loras/new_style",
    weights="uploaded.safetensors",  # Select explicitly if multiple files exist.
    metadata={
        "display_name": "New style",
        "compatibility": {
            "model_variant": "flux_1_dev",
            "verified_checkpoint_ids": [],
        },
    },
)
```

`weights` and `metadata` affect only newly created defaults; the helper never
edits existing files. Metadata replaces template fields at the top level, so
provide the complete value when supplying `settings` or `compatibility`.
For an explicitly configured pretrained model, pass
`metadata={"weights": None, "base_model_id": "organization/model"}`.
`initialize_catalog(root)` initializes all recognized assets and returns paths
created, while `discover_catalog(root)` does this automatically and then returns
the validated catalog. Both web pages and generation/inspection CLIs use this path;
a generation `--dry-run` can create missing catalog manifests but no outputs.

## Validation

Validation checks schema version 1, required identity/loading fields, shared
setting types, duplicate IDs across the whole catalog, and model links to
unknown LoRA IDs (including disabled links). JSON duplicate keys and non-finite
numbers are rejected. User manifests may contain only `schema_version` and
`settings`; they cannot replace identity, requirements, compatibility, workflow,
supported modes, or weight paths.

`weights`, `notes_file`, and `preview_image` paths resolve relative to the asset
manifest. Absolute paths remain absolute. Discovery does not check that local
weights referenced by existing manifests exist, hash their contents, or inspect
the Hugging Face cache. Initialization only inspects paths to select weights for
new manifests; it never reads weight contents.
`weights: null` is valid for models with a nonempty `base_model_id`; that remote
repository identifier is preserved as metadata, not resolved as a local path.

Family and workflow names are not restricted to today's three implementations.
New metadata and settings are retained. Validation covers shared setting types;
workflow-specific numeric ranges, dimensions, scheduler rules, and input requirements
are checked by the generation backends. Actual LoRA tensor compatibility is
checked by the model loader.
An empty verified-checkpoint list or unknown WAN expert target remains valid
catalog metadata and does not establish loading compatibility.

## Settings precedence

Model and LoRA defaults can store example prompts and example negative prompts
as top-level metadata fields, beside `description` and outside `settings`:

```json
"example_prompts": [
  "A mountain lake at sunrise, soft golden light",
  "A watercolor painting of a quiet village in autumn"
],
"example_negative_prompts": [
  "blurry, low quality, watermark",
  "overexposed, washed out colors"
]
```

Use an empty array (`[]`) when there are no examples. Each entry must be a
nonempty string. Both fields are optional for older manifests. Examples are retained
as metadata and do not change generation settings or insert text into the active
prompt. Newly initialized assets start with empty arrays for both fields.

```text
Workflow defaults → Asset defaults → Saved user overrides → Request overrides
```

The optional workflow and request layers are supplied by callers; generation
preparation provides the selected workflow's defaults. The resolver merges only the
`settings` objects, never the manifest metadata:

- Objects merge recursively by key. Overriding `sampling.sampler` preserves
  `sampling.noise_schedule` from a lower-priority layer.
- Lists replace as whole values, including an empty list. LoRA links are not
  merged by ID or appended automatically.
- Scalars replace earlier values. Explicit `null` is preserved where supported,
  such as `seed` and `clip_skip`; it does not mean “remove this override.”
- Removing a saved override restores the lower-priority default on the next load.
- Resolution does not mutate the source objects or save anything to disk.

The lower-level `merge_settings(*layers)` and
`resolve_settings(defaults, user_overrides=None, *, workflow_defaults=None,
request_overrides=None)` functions are pure merges of settings dictionaries.
`CatalogAsset.resolve_settings()` also checks the effective shared setting types.

LoRA settings resolve independently of model settings. Model links remain under
`settings.loras` with their saved `enabled` and optional checkpoint-specific
`strength` values. Discovery does not create links, infer compatibility from a
family folder, insert trigger words, or apply `recommended_generation` to model
settings. The future job-preparation layer will combine enabled adapter settings
with checkpoint-specific and request-specific strengths.

## Tests

```bash
python -m unittest discover -s tests -p 'test_catalog.py' -v
python -m unittest discover -s tests -v
```

Tests use temporary manifests for merge behavior, error handling, paths, links,
and CLI output. CLI tests run outside the project with Python's `-S` option to
exclude installed packages. A separate smoke test discovers the repository's
actual manifests without loading any models.

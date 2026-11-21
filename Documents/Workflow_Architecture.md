# Workflow integrations

Model assets remain in `Models/Model_Families/<family>/checkpoints/<model>/`
and `loras/<adapter>/`. This refactor does not require a manifest migration,
new model IDs, or changes to the saved queue format.

Each integration under `src/image_generator/workflows/` owns its definition,
parameter defaults and controls, validation, runtime, and optional UI assets.
The six built-in workflow names remain `sdxl`, `flux`, `qwen_2_1`, `krea2`,
`wan`, and `minimax`. Existing imports such as
`from image_generator.workflows.wan import WanWorkflow` continue to work.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `definition.py` | Register modes, named inputs, output kinds, backend reference, adapter policy, and optional assets/runtime options. |
| `parameters.py` | Own fallback settings and typed controls, including constraints and CLI aliases. |
| `validation.py` | Enforce cross-field constraints and validate assets without importing GPU libraries. |
| `runtime.py` | Own pipeline instances, load components, run inference, report progress, and release resources. |
| `adapters.py` | Optional specialized adapter policy, conversion, or loading. |
| `conditioning.py` | Optional workflow-specific source preprocessing. |
| `templates/` and `static/` | Optional specialized markup and browser behavior. |

The web and CLI use the same parameter declarations. `Parameter` provides type
validation and HTML value parsing; `UIField` remains a compatibility alias.
Cross-field validators may raise `FormValidationError` for field-level errors
or `ValueError` for a general settings error.

`shared/` contains reusable Diffusers lifecycle, adapter, scheduler, asset,
seed, and metadata helpers. Its code does not select implementations by family.
`common.py` retains compatibility imports for the former shared module.

## Request lifecycle

1. Resolve a manifest's `workflow` through the explicit registry.
2. Load its lightweight definition and select a declared mode.
3. Resolve workflow defaults, model defaults, saved overrides, and current edits.
4. Parse controls and validate settings without constructing a runtime.
5. For generation, validate and stage every named source, prepare a request,
   and enqueue its existing serialized snapshot. A failed submission rolls back
   all newly staged sources. Previews and settings saves do not stage sources.
6. The worker constructs the registered backend lazily, validates the snapshot,
   and executes serialized requests. The backend owns pipeline reuse and cleanup.
7. Shared saving code writes images or videos and matching metadata. The result
   type determines the output directory, independently of the workflow name.

`ModeDefinition.executable` controls whether generation is available. There is
no separate web submission allowlist. Only modes that are implemented should be
declared executable; a preview-only mode can set `executable=False`.

Definitions can be inspected with Python's `-S` flag. Neither model runtimes nor
Flask are needed to inspect built-in definitions. Definitions must not import
`torch`, `diffusers`, ComfyUI nodes, or backend execution modules.

## Add a workflow

Create an integration package and a lightweight definition:

```python
from pathlib import Path

from image_generator.workflows.definition import ModeDefinition, WorkflowDefinition
from image_generator.workflows.inputs import InputSpec
from .parameters import DEFAULTS, get_ui_definition
from .validation import validate, validate_form_settings

DEFINITION = WorkflowDefinition(
    name="my_workflow",
    family="sdxl",
    runtime=("image_generator.workflows.my_workflow.runtime", "MyWorkflow"),
    defaults=DEFAULTS,
    modes=(ModeDefinition(
        "img2img", output_kind="image",
        inputs=(InputSpec("input_image"),),
    ),),
    parameters=get_ui_definition,
    validate_settings=validate_form_settings,
    validate_request=validate,
    asset_root=Path(__file__).parent,
)
```

The callbacks receive `(model, mode)` for parameter construction,
`(model, mode, settings)` for settings validation, and a `GenerationRequest`
for asset validation. The parameter factory returns a `UIDefinition` containing
settings controls. Input controls are supplied automatically from the mode's
`InputSpec` declarations. The built-in upload handler supports single-frame
images, including multiple named images such as a source and mask. A new media
kind requires adding the appropriate bounded upload validator and serving support.

Register the definition once in `workflows/registry.py`:

```python
WORKFLOWS["my_workflow"] = (
    "image_generator.workflows.my_workflow.definition", "DEFINITION",
)
```

Registry import paths are trusted application code. Model manifests select a
registered name; they cannot supply import paths, template paths, or scripts.

Implement the `Workflow` contract in the runtime. `validate()`, form validation,
and definition lookup are supplied by the base class. Implement `generate()` and
`close()`, and optionally `generate_batches()` for incremental outputs. Reuse
shared components or an existing runtime where their behavior fits. Keep a new
generation sequence explicit when it differs substantially.
SDXL and the shared Diffusers runtime expose `_build_pipeline(request)` so a
derived runtime can customize construction while retaining pipeline caching,
device placement, scheduler capture, and cleanup.

Add the model's normal manifest pair under `Models/`, referencing the new
workflow and its actual supported modes. A family and workflow may have different
names. Additional checkpoints using the implementation need only asset manifests.

## Customize one model

Create a separate package with its own definition and runtime. Reuse a family
definition with `dataclasses.replace` where appropriate:

```python
from dataclasses import replace
from pathlib import Path
from image_generator.workflows.sdxl.definition import DEFINITION as STANDARD

DEFINITION = replace(
    STANDARD,
    name="my_custom_sdxl",
    runtime=("image_generator.workflows.my_custom_sdxl.runtime", "CustomSDXLWorkflow"),
    asset_root=Path(__file__).parent,
)
```

Point only the intended model manifests at `my_custom_sdxl`. Other SDXL models
continue using `sdxl`. Override parameter/validation callbacks when introducing
new settings or rules; extend defaults using independent copies rather than
mutating the standard definition. A runtime receives an independent copy of its
definition's defaults.

If the derived workflow reuses a standard template or script, either keep the
standard asset root or provide those assets in its own package. When overriding
the asset root, also update inherited `scripts`, `styles`, `group_partials`, and
the parameter factory's optional `partial` as needed. SDXL currently uses only
generic controls and needs no package-local UI assets.

## UI assets and adapters

Templates render through a workflow namespace. A UI partial such as
`controls/custom.html` is resolved from the selected definition's
`asset_root/templates/`; declared scripts and styles are served from its
`asset_root/static/`. Group partials can add markup to an existing control group.
Keep scripts scoped with a module or IIFE so workflows cannot collide through
global variable declarations. Package data includes integration HTML, JS, and CSS
in built wheels.

`AdapterPolicy` supplies eligibility, strength bounds, optional typed per-link
fields, and optional adapter groups. WAN owns its expert fields, variant rules,
and targeted loading. MiniMax owns its explicit H3 rules. Qwen's definition
permits signed strengths and its package owns LoRA conversion. Shared forms and
model settings persistence consult the selected workflow's policy. Shared adapter
defaults use the family's standard policy.

## CLI and runtime configuration

`run_family(workflow_name, argv)` builds its controls from the selected model and
mode, retaining declared aliases such as `--steps`, `--frames`, and `--kv-cache`.
`--mode` selects a supported mode; video generation does not imply an input image.
Use the text-to-video checkpoint's model ID for a text-to-video request.
The generic entry point needs no additional wrapper script:

```bash
python -m image_generator.cli wan --model-id YOUR_T2V_MODEL_ID --prompt "Clouds move over mountains" --dry-run
```

Definitions declare additional process-only options through `RuntimeOption`.
Both worker and CLI discover these declarations. MiniMax declares `comfyui_dir`
inside its own definition. Runtime options and credentials stay outside persisted
generation settings and request snapshots.

## Validation

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p 'test_workflow_modularity.py' -v
```

The modularity tests register a test-only integration with an existing family,
a new workflow/mode, custom parameters and UI assets, and two source images.
They exercise web preview/settings, CLI, upload rollback, queue snapshots, worker
execution, and source/output downloads without modifying shared application code.
Existing execution tests inject fake GPU runtimes at the new implementation
module paths. Actual checkpoint inference still requires GPU smoke tests.

On macOS, use a canonical temporary directory (for example `TMPDIR=/private/tmp`)
when running the suite: existing path assertions compare resolved catalog paths.

# Generation forms and the workflow UI contract

The shared UI provides editable forms and stateless request previews for
SDXL text-to-image, FLUX.1-dev text-to-image, WAN 2.2 image-to-video, and MiniMax H3
image-to-video. It uses
the existing Flask/Jinja application and model catalog. No GPU, model downloads,
or weight files are required to render or preview a form.

SDXL and FLUX support **Generate images**; WAN and MiniMax support **Generate video**
with managed source uploads, live job progress, and saved results through the
durable worker. See [SDXL execution](SDXL_UI_Execution.md) for
the two-process startup, model reuse/switching, and LoRA-reset checks. See
[FLUX execution](FLUX_UI_Execution.md) for its guidance controls and image results. Fixture
manifests are preview-only unless tests supply fake weights and a mocked runtime.

## Try the fixture library

From the project root, with the project's Python environment and Flask installed:

```bash
IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families .venv/bin/python main.py
```

Open <http://127.0.0.1:5000/> and select **Configure generation**, or use:

| Family | Form | Initial values to check |
| --- | --- | --- |
| SDXL | <http://127.0.0.1:5000/generate/fixture-sdxl> | Width 768, seed 42, Euler ancestral, blank clip skip, style adapter enabled at 0.65. |
| FLUX | <http://127.0.0.1:5000/generate/fixture-flux> | Guidance 4, true CFG 1, maximum sequence length 256. |
| WAN | <http://127.0.0.1:5000/generate/fixture-wan> | 49 frames, 12 FPS, both guidance scales 3.5, high-noise adapter enabled at 0.7. |
| MiniMax | <http://127.0.0.1:5000/generate/fixture-minimax> | 124 frames, fixed 24 FPS and CFG 1, RES multistep/simple sampling, one adapter at 0.7. |

The fixture library contains all defaults/user JSON pairs, but deliberately no
weights. It is separate from the real library. Stop the server with Ctrl+C;
running `.venv/bin/python main.py` without the environment variable uses the
actual library again. Real models also have a **Configure generation** link.

MiniMax uses `17*n + 5` frames and its own partial, without WAN's expert groups.
Negative prompt, CFG, FPS, sampler, and schedule controls are read-only for its BasicGuider mode;
Python rejects attempts to change the fixed settings. See the
[MiniMax web guide](MiniMax_UI_Execution.md) for generation, progress, and runtime requirements.

Manual checks:

1. Enter a prompt on SDXL and click **Preview request**. Check the JSON's
   `overrides`: numbers are numbers, blank clip skip is `null`, and nested
   sampling values reflect the selected sampler. Width 770 should fail the
   browser's dimension increment check. A blank seed means random, not zero.
2. Enable SDXL's detail adapter alongside the saved style adapter and change its
   strength. Both should appear in `overrides.loras`. The model's 24 inference
   steps should remain unchanged by the adapter's displayed 18-step suggestion.
   Trigger-word inclusion is a separate opt-in checkbox.
3. On FLUX, enter a negative prompt and set true CFG above 1. The inline status
   should say negative prompting is active. The sequence limit is 256 because
   the fixture checkpoint restricts it, rather than the workflow's usual 512.
4. On WAN, select a local image. Its thumbnail and estimated playback duration
   should appear; 49 frames at 12 FPS is approximately 4.08 seconds. Check that
   adapters are grouped by expert and the unknown-target adapter is disabled.
   Set low-noise guidance to 0.5 while high-noise guidance remains 3.5, then
   preview: the server should report their paired-CFG constraint.
5. Click **Reload saved settings** or revisit a model. Unsaved edits should be
   discarded. Try the page at a narrow window width and using keyboard navigation.

The browser checks ordinary input constraints; Python checks submitted values
independently, including direct POST requests that bypass browser checks. Errors
return HTTP 422 with entered values preserved. Unknown model IDs return 404,
unsupported modes 400, and unavailable workflows/forms 422.

## Contract and rendering

`workflows/ui.py` defines frozen `UIField` and `UIDefinition` dataclasses and
`FormValidationError`. The workflow interface in `workflows/base.py` exposes:

```python
workflow.ui_modes
ui = workflow.get_ui_definition(model, mode)
workflow.validate_form_settings(model, mode, settings)
```

Each workflow declares its modes in its package's `definition.py` and its
field definitions/defaults in `parameters.py`. The registered lightweight
definition supplies forms without constructing the backend runtime. See the
[workflow extension guide](Workflow_Architecture.md) for the package structure.
Definitions contain field names, types, labels, groups, constraints, help, select
choices, and an optional specialized template partial. They do not contain current
values. For example, a field can address `sampling.sampler` inside nested settings.

Supported field types are text, textarea, integer, number, boolean, select,
newline-separated string lists, and file inputs. Fields with `scope="settings"`
become typed settings; `scope="inputs"` is for transient source files. Model
identity, asset paths, requirements, and compatibility are never editable fields.
Mode, adapter selections, and trigger-word inclusion are shared form controls.

Current values resolve through the existing precedence:

```text
Workflow defaults → Model defaults → Saved user overrides → Submitted form
```

Dimension alignment combines workflow and checkpoint requirements. Sampler
choices intersect implemented options with `requirements.allowed_sampling`.
FLUX respects the checkpoint sequence limit. WAN combines checkpoint frame-count
rules with `4*n+1` and validates its paired guidance scales and positive FPS.
Seeds render as text inputs with numeric hints to preserve the full 63-bit
integer range without JavaScript number rounding.

`web/generation_routes.py` handles `GET` and `POST /generate/<model_id>`;
`web/forms.py` converts HTML values and resolves adapter choices. The shared
`generate.html` template uses `controls/field.html` and `controls/loras.html`.
WAN adds `controls/wan.html` for source preview and video information. Partial
paths come from Python workflow definitions, never arbitrary manifest values.
The CSS and optional JavaScript are included in package builds.

Adapter options use family/variant metadata and separately display checkpoint
verification. Unknown WAN variants/expert targets remain unavailable. Strength
precedence is model link → adapter saved override → adapter default. Adapter
recommendations are displayed but never applied automatically.

## Save and reset settings

**Save model settings** persists the generation controls and configured LoRA
list to `model.user.json`. An empty prompt is allowed when saving defaults;
previewing or generating still requires one. Saving does not check weights,
upload source images, or load a runtime. Video source files remain transient.

Enable any number of compatible adapters. **Keep in model settings when
disabled** retains an adapter and its strength without using it in generation.
**Remove from model settings** clears both checkboxes; save to persist removal.
Choose **Use adapter default** to follow the adapter's shared strength, or
**Checkpoint-specific override** to save a strength for this model only.
Changing the strength input selects a checkpoint override. Recommendations
change supported form fields only after **Apply recommended settings** is
clicked; save separately to keep them.

Expand **Edit shared adapter defaults** to save strength and trigger words in
`lora.user.json`. This affects models that follow the adapter default and leaves
checkpoint overrides intact. **Reset adapter defaults** clears all saved
overrides for that adapter. Shared-default actions reload the form, discarding
unsaved model edits.

**Reset model settings** clears all model overrides and restores workflow/model
defaults, including shipped adapter links. **Reload saved settings** discards
unsaved form edits without changing files. Saving a LoRA list replaces the
inherited list; removing every adapter saves an explicit empty list when needed.

The CPU-only `catalog/persistence.py` service writes only differences from
workflow plus model defaults (adapter defaults for shared preferences). It
preserves settings outside the edited fields, including nested keys, and keeps
explicit `null` values such as random seeds. Its `remove` parameter deletes named
overrides to restore inheritance. `web/settings_routes.py` provides CSRF-protected
POST actions at `/settings/models/<model_id>` and
`/settings/adapters/<adapter_id>`. An editor revision includes catalog defaults
and user settings; stale saves/resets return 409 and require reloading. Service
writes are serialized by a catalog lock and publish validated JSON through an
atomic file replacement. External manifest editors should reload before saving
and avoid writing concurrently with the app.

To check persistence using the fixture library, save FLUX's style adapter enabled
at 0.35 and detail adapter disabled at 0.7, reload, then preview with a prompt.
Both configurations should survive, while only style appears in the preview.
Try default/override strength modes, shared defaults, removing all adapters,
resetting, and submitting a second tab opened before the first tab's save.

To add another checkpoint for an existing workflow, provide its manifests and
requirements. For another workflow, register a `WorkflowDefinition` declaring
its modes, inputs, fields, validation, and backend. Executable modes enable
submission automatically. Add package-local templates/scripts if generic
controls are insufficient; shared routes need no workflow-specific branch.

## Preview and generation behavior

A successful POST displays a JSON preview with `model_id`, `mode`, fully resolved
`overrides`, `include_trigger_words`, and `input_files`. The first four correspond
to the existing generation-preparation arguments. `input_files` contains selected
filenames only, not executable filesystem paths or uploaded asset references.

Request previews do not save uploaded files, persist settings, queue jobs, load
weights, or produce media. WAN and MiniMax previews validate image contents.
The separate Generate action queues validated requests. Source files must be selected
again after submitting the preview. The browser thumbnail uses a temporary local
object URL. JavaScript supplies hints only; the server works without it.

WAN and MiniMax generation save source images as managed inputs, translate them
to `prepare_generation(..., inputs={"input_image": path})`, and run full workflow
validation before queuing a snapshot. Existing `workflow.validate()` remains
authoritative for weights, supported loaders, and generation requests; loaded
pipeline checks still enforce runtime-specific WAN alignment requirements.

Catalog initialization can still create missing manifests in the real library
according to its existing rules. The supplied fixtures include all manifests,
so form previews do not change them.

## Automated checks

Run the UI tests only:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*ui*.py' -v
```

Run the full regression suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

`test_ui_contract.py` checks workflow field coverage, checkpoint restrictions,
mode support, frame alignment, types, and stdlib-only contract imports with
Python's `-S` flag. `test_generation_ui.py` copies fixture manifests into temporary
directories, inspects rendered controls, and submits their values through Flask's
test client. It covers typed previews, multiple adapters, seed precision,
validation errors, HTML escaping, unchanged files, absent weight handling,
navigation, static assets, and rendering the actual library's models.

`test_settings.py` uses temporary copies of the manifests to check model/adapter
save and reset, disabled adapters, checkpoint isolation, shared inheritance,
nested overrides, null seeds, empty lists/prompts, validation and CSRF rejection,
stale/concurrent submissions, and failed atomic replacements. Run it with:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_settings.py' -v
```

These are HTTP/template and Python checks; they do not execute browser JavaScript
or certify visual layout. Use the manual checks above for thumbnails, dynamic
hints, responsive layout, and browser-native validation.

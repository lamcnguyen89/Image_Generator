# Flask catalog and generation forms

Run from the project root after the [environment setup](Setup.md):

```bash
bash scripts/start.sh
```

This starts the web server and generation worker together using `.venv`. Logs
appear in the same terminal. Ctrl+C stops both, allowing active generation to
finish. See [launcher options and GPU testing](SDXL_UI_Execution.md).

## Example media on cards

Model catalog cards, linked LoRA cards, and generation-form LoRA cards show
centered example media, with the asset title overlaid near the top and settings
and actions below. Single examples retain their original aspect ratio;
carousels use a fixed frame that contains each example without cropping. Cards
fit their own content rather than stretching to match taller neighbors.

With JavaScript enabled, only the selected slide in each visible carousel has
an active media source. Other slides keep URLs in data attributes so the browser
cannot prefetch them. Navigating away removes image sources and pauses and
resets video elements to discard playback buffers. Scrolling the carousel out
of view, collapsing its section, or hiding/leaving the page releases its active
media; returning restores the selected slide without autoplay. Browser-managed
caches and the timing of memory reclamation remain under the browser's control.
Without JavaScript, a link opens the first example. Single-example cards retain
native image lazy loading or video metadata preloading.

Set `preview_image` in an asset's defaults manifest to an image or video path
inside that asset's directory. When it is unset or the file is missing, the UI
uses supported media files in filename order, ignoring case. Any filename
works, and refreshing automatically synchronizes the defaults JSON's
`example_media` list with files in the folder. Multiple examples form a carousel
with arrows, a counter, swipe/scroll, and keyboard navigation.
Supported extensions are PNG, JPG/JPEG, WebP, GIF, AVIF, MP4, WebM, and MOV/QT;
video playback depends on browser codec support. Assets without media keep their
title and show a placeholder. Media uses the same sign-in protection as the UI.
Model card images link to the same generation page as their titles. Template
edits reload on the next request; Python code changes still require a restart.

Checkpoint and LoRA entries also show a **Source website** link when their
defaults JSON includes an HTTP or HTTPS `source_url`. Links open in a new tab.
The generation page groups its title and links in a violet card, with a compact
checkpoint example image or video on the right. The preview matches the text
section's height and moves below it on small screens.

## Example prompt menus

Each generation page has separate **Example prompts** and **Example negative
prompts** menus for the checkpoint and every LoRA. Expand a menu, select a saved
example, and use the text area to review it or type a new example. **Use as
prompt** and **Use as negative prompt** copy the text into the corresponding
generation field. Selecting an example alone does not change the active prompt.

**Add example** saves the text as a new entry. **Remove selected** deletes the
selected saved entry. Changes save immediately to `example_prompts` or
`example_negative_prompts` in that asset's defaults JSON, shared wherever that
asset is used. Other metadata and settings are preserved, and the form stays
open with its unsaved edits intact. Concurrent edits are detected; reload the
page if the saved catalog changed in another editor. Empty or duplicate examples
cannot be added.

## Local Wi-Fi hostname

Set these values in the project's `.env` file:

```dotenv
IMAGE_GENERATOR_HOSTNAME=imagegen
IMAGE_GENERATOR_PORT=5000
IMAGE_GENERATOR_HOST=0.0.0.0
```

Restart with `bash scripts/start.sh`, then open **http://imagegen.local:5000**
from a device on the same local network. Enter just `imagegen` in the hostname
setting; the launcher appends `.local`. An existing `.local` suffix is also
accepted. Keep the port separate, and do not put `http://` or `:5000` in the
hostname. Changing the name does not rename your computer.

The launcher publishes the alias through Avahi while it runs and removes it
when stopped. On Ubuntu, install the system tools if needed:

```bash
sudo apt install avahi-utils avahi-daemon
sudo systemctl enable --now avahi-daemon
```

Clients must support mDNS, and the Wi-Fi network must allow communication and
multicast between devices. If a firewall is enabled, it must allow the web port
and mDNS traffic on the local network. No router port forwarding is needed.
The alias uses the IPv4 address selected by the system's multicast route. If
you have multiple network interfaces, set `IMAGE_GENERATOR_HOST` to the desired
LAN IPv4 address instead of `0.0.0.0`. Restart after the server's IP changes.
A duplicate hostname or unavailable Avahi daemon causes the launcher to stop
with an error; choose a unique name or start Avahi before retrying.

`--host` and `--port` override the corresponding `.env` settings. Environment
variables already set in the shell take precedence over `.env`. Leave
`IMAGE_GENERATOR_HOSTNAME` empty to disable alias publishing. Without a hostname
or explicit listening address, the launcher defaults to `127.0.0.1`; the default
port is `5000`. The hostname settings apply to `scripts/start.sh`; starting only
the web server with `main.py` does not publish an alias.

To start only the web server:

```bash
.venv/bin/python main.py
```

Open <http://127.0.0.1:5000>. The server uses Flask's local development server
with debugging off. The installed package also supports Flask's application
factory CLI:

```bash
.venv/bin/python -m flask --app image_generator:create_app run
```

## Password protection

Set a shared password in the project's `.env` file and restart the service:

```dotenv
IMAGE_GENERATOR_ACCESS_PASSWORD="your-password-here"
```

Visitors must sign in before opening any app page, using an API, submitting an
upload, or viewing/downloading generated media and metadata. The sign-in page
and its stylesheet remain public. Unauthenticated API requests return JSON
with status 401. **Menu → Sign out** ends the current browser session.
Passwords are compared exactly, including spaces; numeric-looking passwords
also work. Leave the setting empty or unset to disable the gate.

Both `scripts/start.sh` and `main.py` load the project's `.env`; values already
exported in the shell take precedence. Restart after changing the password.
Sessions normally end when the browser session ends or the server restarts.
Optionally set `IMAGE_GENERATOR_SECRET_KEY` to a long random string to preserve
sessions across restarts or share sessions between web processes. Changing the
access password still invalidates existing sign-ins.

This is a shared-password gate for casual access control. HTTP does not encrypt
the password or session cookie; use HTTPS if you need encrypted access.
The command-line runners and local files remain available to the server's user.

## Pages

Every page includes a **Shared Memory** card beside the header menu. It refreshes
every three seconds with system RAM usage, a usage bar, percentage, and used/total
GiB. On unified-memory hosts, this is the physical pool shared by CPU and GPU;
reclaimable caches count as available memory.

**Clear all models** asks the generation worker to release its cached workflow,
including checkpoint, adapters, and runtime model caches. An active generation
finishes first; cleanup runs before the next queued job. The card confirms when
cleanup completes or reports an error if the worker is unavailable. Queued jobs
continue normally and load their models again as needed. Other processes' models
are outside this application's control. Restart the web server and worker after
updating to enable these controls.

- `/` lists discovered models grouped by family, their declared modes, and
  registered LoRA counts. Choose **View settings** to inspect a model.
- `/models/<model_id>` shows model metadata, effective settings, saved linked
  LoRAs and their strengths/selections, defaults, user overrides, requirements,
  and manifest paths. Values retain JSON types: `null`, `false`, lists, and
  nested objects remain distinguishable.
- `/generate/<model_id>` renders the workflow's editable form. POST previews
  typed settings and reports errors without saving settings. SDXL and FLUX forms
  submit jobs through **Generate images**; WAN and MiniMax forms use **Generate
  video** with a source-image upload. All execute on the dedicated worker.
  See [generation forms](Generation_UI.md) for fixture startup and testing.
- `/jobs` lists recent jobs; `/jobs/<job_id>` displays live progress, execution
  history, errors, and saved outputs. Below the progress bar, elapsed time and
  estimated remaining time update every second for SDXL, FLUX, WAN, and MiniMax.
  Elapsed time starts when the worker claims the job and stops when it finishes.
  Estimates use observed denoising speed and include later videos in a batch;
  loading, decoding, and saving can add time. The display shows “Estimating…”
  before timing samples are available and “Finishing…” during final processing.
  Timing survives page refreshes. `/api/jobs/<job_id>` supplies polling data,
  including a `timing` object with numeric seconds and formatted display values.

The catalog and saved-settings pages support GET and HEAD; writes to those
routes return 405. Generation forms support GET, HEAD, POST previews, and SDXL/FLUX/WAN/MiniMax
queue submissions. Browsing never loads weights or queues jobs. Catalog browsing keeps workflow
imports lazy; generation forms import only lightweight definitions and validation.
Catalog loading creates missing defaults/user JSON files using
the [initialization rules](Catalog.md#automatic-manifest-initialization), without
editing existing files. The library must be writable when manifests are missing;
initialization errors appear on the catalog diagnostic page.
LoRA family counts are inventory counts,
not claims of checkpoint compatibility. Only explicitly linked adapters appear
on model pages. A checkpoint-specific strength takes precedence over the
adapter's resolved strength; recommendations are displayed separately and
never applied to model settings.

The catalog is discovered once per request. Refresh after editing JSON files
to see the latest saved configuration. Unknown models return 404. Invalid
catalog configuration returns a diagnostic page with status 503, naming each
source file and field. Fix the files and refresh to recover without restarting.

## Application configuration

`create_app()` defaults to `Models/Model_Families` in the source checkout.
Supply another catalog root through application configuration:

```python
from image_generator import create_app

app = create_app({"CATALOG_ROOT": "/path/to/Model_Families"})
```

Or set the environment variable before starting the app:

```bash
IMAGE_GENERATOR_CATALOG_ROOT=/path/to/Model_Families .venv/bin/python main.py
```

Flask's prefixed configuration loader reads `IMAGE_GENERATOR_*` variables;
explicit factory configuration takes precedence. Relative catalog roots resolve
against the working directory. When installed outside an editable checkout,
configure the path to your external model library explicitly.

The factory is in `src/image_generator/__init__.py`. Catalog views live in the
`catalog` blueprint under `web/web.py`; the `generation` blueprint is in
`web/generation_routes.py`. Shared templates, CSS, and JavaScript are under
`web/templates/` and `web/static/`. These assets are included in Python package builds. Flask
imports are deferred until the factory is called, preserving the catalog
inspection script's ability to run without installed web or GPU dependencies.

The `jobs` blueprint lives in `web/job_routes.py`. Database/output configuration,
worker startup, and real GPU testing are described in [SDXL execution](SDXL_UI_Execution.md).
See [WAN execution](WAN_UI_Execution.md) for source storage, video playback,
upload limits, and WAN-specific GPU checks.
See [MiniMax execution](MiniMax_UI_Execution.md) for native ComfyUI progress,
H3 restrictions, silent video results, and runtime startup.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_web.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

Flask test-client checks cover navigation, resolved values, adapter strengths,
read-only behavior, manifest refresh, HTML escaping, error pages, static assets,
and rendering every model in the repository catalog.

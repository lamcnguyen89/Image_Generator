# Application environment setup

## Web development without ML dependencies

For web application work on macOS (Apple Silicon or Intel) or Linux (including
x86 Ubuntu), use the lightweight installer with an existing Python 3.11:

```bash
bash scripts/setup_web.sh
```

Select an interpreter explicitly if needed:

```bash
bash scripts/setup_web.sh --python /path/to/python3.11
```

`PYTHON_BIN` also selects the interpreter. The installer creates a separate
`.venv-web`, installs `requirements-web.txt`, checks dependency consistency,
and renders the fixture catalog, model settings, and every generation form.
It also checks static asset serving and the memory API. Verification uses
temporary copies of the fixtures. Rerunning setup reuses `.venv-web`.

The only direct dependencies are Flask, python-dotenv, Pillow (image upload
validation), and psutil (memory monitoring). Setup does not use sudo or install
system packages, PyTorch, Diffusers, Transformers, ComfyUI, or model weights.
It leaves `.venv` and the full generation installer's requirements untouched.
It requires Python's `venv` and pip support to be available already.

From the project root, start only the web server with the weight-free fixtures:

```bash
IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families .venv-web/bin/python main.py
```

Open <http://127.0.0.1:5000>. You can browse catalogs, edit settings, validate
image uploads, and preview requests. Saved settings update the selected catalog's
JSON files, including fixtures when selected. Actual image/video generation needs
the full runtime and real assets; this installer does not start a worker.
`scripts/start.sh` and `scripts/setup.sh --dev` remain full generation workflows.

For Python and template live reload, use Flask's local development server:

```bash
PYTHONPATH=src IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families \
  .venv-web/bin/python -m flask --app image_generator:create_app run --debug
```

The web environment runs from source: `main.py` and the tests add `src` to the
Python import path. Avoid `pip install -e .` in this environment because the
package metadata currently declares the full machine learning dependency set.

Run the web and request preview tests without ML dependencies:

```bash
.venv-web/bin/python -m unittest discover -s tests -p 'test_web.py'
.venv-web/bin/python -m unittest discover -s tests -p 'test_generation_ui.py'
```

## Full generation environment

The application uses one project-local `.venv` for its SDXL, FLUX, WAN, and MiniMax
workflows. MiniMax H3 imports native ComfyUI Python nodes from `build/ComfyUI` in
that same environment; no separate environment or ComfyUI server is needed.
The notebooks remain independent experiments. All four families have
reusable backends and command-line runners: [SDXL](SDXL_Workflow.md),
[FLUX](FLUX_Workflow.md), [WAN](WAN_Workflow.md), and [MiniMax](MiniMax_Workflow.md).

## Current status

Setup provisions all four families. The original runtime pins were read from the
existing `huggingface` Conda environment, except Flask. MiniMax additions were
resolved with pip's dry-run mode on Python 3.11 / Linux aarch64 without installing
packages. The baseline is Python 3.11 on Linux
aarch64 with `torch==2.14.1+cu132` and `torchvision==0.29.1+cu132`.
The pinned Transformers version uses torchvision for WAN's CLIP image processor.
These packages have not been validated together in a fresh environment. Keep the
existing Conda environment during migration.

`requirements.txt` currently pins direct dependencies only. It is **not** a
complete generated lock of transitive dependencies. The default PyTorch wheel
URLs are listed in the official CUDA 13.2 index. Generate the complete lock after
resolving and validating the runtime together on the target machine.

## Files

- `requirements.in`: direct application dependencies and compatibility ranges.
- `requirements.txt`: initial exact direct pins; later the generated runtime lock.
- `requirements-dev.txt`: optional pip-tools and notebook kernel tooling.
- `.python-version`: the selected Python minor version.
- `pyproject.toml`: installs the package from `src/`; reads dependencies from
  `requirements.in` so package metadata shares the same declarations.
- `scripts/setup.sh`: installs the system `python3.11-dev` package when missing,
  then creates `.venv`, downloads ComfyUI if absent, installs all family
  dependencies and the local package, and runs environment checks.
- `scripts/check_environment.py`: checks imports, FFmpeg, CUDA, BF16, small matrix
  multiplications, PyTorch SDPA, and MiniMax's native ComfyUI node imports and
  sampler support. It does not download or load model weights.

## First setup

Use a standalone Python 3.11 interpreter compatible with the selected PyTorch
wheel. Setup rejects Conda interpreters and venvs created from them: a venv keeps
a dependency on its base Python installation. Activating `huggingface` or passing
its Python path does not produce an environment independent of Conda.

If standalone Python 3.11 is not installed, one option is uv's
[standalone installer](https://docs.astral.sh/uv/getting-started/installation/),
which does not require Python or Conda:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal so `uv` is on `PATH`, then install
[uv-managed Python](https://docs.astral.sh/uv/guides/install-python/):

```bash
uv python install 3.11
uv python find --managed-python 3.11
```

Use the printed path as `--python`, or use the command substitution below. Keep
that standalone Python installation: `.venv` depends on it. An independently
installed system Python 3.11 also works; pass its path explicitly if Conda comes
first on `PATH`. Setup does not install Python automatically.

Setup downloads these pinned GPU packages automatically from PyTorch's official
CUDA 13.2 index. Both wheels target CPython 3.11 and Linux ARM64
(`cp311-cp311-manylinux_2_28_aarch64`):

| Package | Version | Official wheel listing |
| --- | --- | --- |
| torch | `2.14.1+cu132` | [PyTorch CUDA 13.2](https://download.pytorch.org/whl/cu132/torch/) |
| torchvision | `0.29.1+cu132` | [torchvision CUDA 13.2](https://download.pytorch.org/whl/cu132/torchvision/) |

This baseline targets the Jetson Thor CUDA 13.2 stack. The development machine
has L4T 39.2.1 and CUDA 13.2.2. Download availability is verified; GPU compatibility
is checked by `check_environment.py` after installation. A successful small GPU
check still needs to be followed by real workflow validation.

No manual wheel download or wheel arguments are needed:

```bash
bash scripts/setup.sh --python "$(uv python find --managed-python 3.11)"
```

If standalone `python3.11` is already on `PATH`, simply run:

```bash
bash scripts/setup.sh
```

For local wheel files or another verified source, `--torch-wheel` and
`--torchvision-wheel` override the defaults independently. Both versions must
still match `requirements.txt`; conflicting wheels fail dependency resolution.
For example, with local copies of the same builds:

```bash
bash scripts/setup.sh \
  --python /usr/bin/python3.11 \
  --torch-wheel /path/to/torch-2.14.1+cu132-cp311-cp311-manylinux_2_28_aarch64.whl \
  --torchvision-wheel /path/to/torchvision-0.29.1+cu132-cp311-cp311-manylinux_2_28_aarch64.whl
```

Replace those paths with your installed interpreter and wheel files.
`PYTHON_BIN`, `TORCH_WHEEL`, and `TORCHVISION_WHEEL` can also supply the overrides.

Run setup as your normal user, without `sudo`; it refuses to install as root so
project files remain owned by your user. Setup installs `python3.11-dev` through
apt when it is missing, using `sudo` for that system package step only. The
package manager must be available, and you may be prompted for your password.
This package supplies development headers; a standalone Python 3.11 interpreter
must already be installed as described above. To install optional development
tools too, append `--dev`. Python packages install into the project's `.venv`; ComfyUI source
installs into `build/ComfyUI`. Setup does not modify the original Conda
environment. Git, JetPack, drivers, and other required system libraries must
already be available.

When `build/ComfyUI` is absent, setup fetches the official ComfyUI repository at
revision `fd7639f477050e5c25b68fc38a9c2a6b6d988211`, which includes native H3
support. Existing checkouts are preserved, including local changes. A checkout
missing `requirements.txt` or `comfy_extras/nodes_minimax_h3.py` stops setup with
instructions to update or move it aside; setup never resets or pulls it.

The project requirements include the pinned ComfyUI baseline. Setup also resolves
the checkout's own `requirements.txt` under the project constraints in the same
pip install as the CUDA wheels. An incompatible existing checkout fails dependency
resolution instead of silently changing the pinned torch/torchvision builds.
No model weights are downloaded by setup. See [MiniMax](MiniMax_Workflow.md) for
its separate checkpoint, VAE, and text encoder assets.

If `.venv` was previously created from Conda, move it aside before rerunning
setup with standalone Python. Setup will refuse to reuse it, including when
`--python` selects a standalone interpreter. Keep the old environment until the
replacement has been validated.

After successful setup:

```bash
source .venv/bin/activate
python scripts/check_environment.py
```

Subsequent setup runs also use the default wheel URLs unless overridden. Pip
handles installed packages and cached downloads. All runtime installs continue
to use the same requirements file.

## Generate the complete runtime lock

Install the optional development requirements. Run compilation using the target
Python 3.11 environment on the target platform, with both official wheel listings
available to the resolver:

```bash
.venv/bin/python -m piptools compile \
  --find-links https://download.pytorch.org/whl/cu132/torch/ \
  --find-links https://download.pytorch.org/whl/cu132/torchvision/ \
  --output-file requirements.txt \
  requirements.in
```

Keep the verified wheel URLs in `scripts/setup.sh` aligned with the requirements
pins when updating this baseline. For another CUDA stack, Python version, or
machine architecture, validate the builds and generate a separate lock.

Commit the requirements files together after reviewing the resolved versions.
Check the environment and perform short real generations for SDXL, FLUX, WAN, and MiniMax,
including their checkpoint/LoRA loading paths, before treating a lock as validated.

## Attention and optional packages

The baseline uses PyTorch's native SDPA. Standalone `flash-attn`, xformers,
bitsandbytes, TensorFlow, and evaluation packages are not included. If an
optimization later proves necessary, install it through a separate documented
optional step and test it against the exact Python/PyTorch/CUDA stack.

Do not install dependencies when a workflow is imported or a model is selected.
Model manifests contain model configuration; package installation stays in setup.

## Checks without installing dependencies

```bash
bash -n scripts/setup.sh
python -m unittest discover -s tests -v
```

The checker supports `--allow-cpu` for dependency inspection on a machine without
CUDA. Passing that mode does not establish that GPU generation works.
Use `--comfyui-dir /path/to/ComfyUI` on the checker to inspect a nondefault
MiniMax checkout; the standard setup and worker use `build/ComfyUI`.

References: [pip-tools](https://pip-tools.pypa.io/en/stable/),
[Flask releases](https://pypi.org/project/Flask/), and
[Diffusers attention backends](https://huggingface.co/docs/diffusers/optimization/attention_backends).

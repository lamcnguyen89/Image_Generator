#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
venv_dir="$project_root/.venv-web"
venv_python="$venv_dir/bin/python"
python_bin="${PYTHON_BIN:-}"

usage() {
    cat <<'USAGE'
Usage: bash scripts/setup_web.sh [options]

  --python PATH  Python 3.11 interpreter (or set PYTHON_BIN).
  -h, --help     Show this help without changing the environment.

Creates .venv-web on macOS or Linux for catalog, settings, forms, and previews.
Installs only Flask, python-dotenv, Pillow, psutil, and their dependencies.
Requires an existing Python 3.11 installation with venv/pip support.
No system packages, model weights, or generation runtimes are installed.
The separate .venv generation environment is left untouched.
USAGE
}

fail() {
    printf 'Web setup error: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --python)
            (($# >= 2)) && [[ "$2" != --* && -n "$2" ]] \
                || fail "--python requires a value."
            python_bin="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1. Use --help for usage." ;;
    esac
done

check_python() {
    "$1" - "$project_root/.python-version" <<'PY'
from pathlib import Path
import sys

expected = Path(sys.argv[1]).read_text().strip()
actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if actual != expected:
    raise SystemExit(f"Web setup requires Python {expected}; got {actual}. Use --python /path/to/python{expected}.")
PY
}

if [[ -z "$python_bin" ]]; then
    if [[ -x "$venv_python" ]]; then
        python_bin="$venv_python"
    else
        python_bin=python3.11
    fi
fi
command -v "$python_bin" >/dev/null 2>&1 \
    || fail "Python executable not found: $python_bin. Install Python 3.11, then pass --python /path/to/python3.11."
check_python "$python_bin"
if [[ -x "$venv_python" ]]; then
    check_python "$venv_python"
else
    "$python_bin" -m venv "$venv_dir"
fi

"$venv_python" -m pip install -r "$project_root/requirements-web.txt"
"$venv_python" -m pip check

# main.py and the tests load src directly. Installing the package here would
# declare the full ML dependencies from pyproject.toml, even with --no-deps.
# Render copied fixtures so this check never changes the user's model library.
"$venv_python" - "$project_root" <<'PY'
from pathlib import Path
import shutil
import sys
import tempfile

import dotenv
from PIL import Image
import psutil

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
from image_generator import create_app
from image_generator.catalog import discover_catalog

with tempfile.TemporaryDirectory() as temporary:
    catalog_root = Path(temporary) / "model_families"
    shutil.copytree(root / "tests/fixtures/model_families", catalog_root)
    app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": catalog_root,
                      "JOB_DATABASE": Path(temporary) / "jobs.sqlite3"})
    catalog = discover_catalog(catalog_root)
    urls = ["/", "/api/system/memory", "/catalog-assets/catalog.css"]
    for model_id in catalog.models:
        urls.extend([f"/models/{model_id}", f"/generate/{model_id}"])
    with app.test_client() as client:
        for url in urls:
            with client.get(url) as response:
                if response.status_code != 200:
                    raise SystemExit(f"Web environment check failed: {url} returned {response.status_code}")
print("Web environment checks passed (catalog, settings, all fixture forms, assets, and memory API).")
PY

printf '\nWeb setup complete. From the project root, run:\n'
printf '  IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families .venv-web/bin/python main.py\n'
printf 'Open http://127.0.0.1:5000 to browse the catalog and preview requests.\n'
printf 'Generation requires the separate full runtime and model weights.\n'

#!/usr/bin/env python3
"""Run the catalog-backed Qwen-Image 2.1 workflow from any working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_generator.cli import run_family


if __name__ == "__main__":
    raise SystemExit(run_family("qwen_2_1"))

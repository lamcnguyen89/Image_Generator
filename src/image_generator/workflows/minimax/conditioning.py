"""Workflow-owned source preprocessing."""

import hashlib
import math
from io import BytesIO


def prepare_input(request, report_progress):
    """Resolve the canvas and optional image without loading GPU components."""
    settings = request.settings
    width, height = settings["width"], settings["height"]
    if request.mode == "txt2vid":
        return width, height, None, {}

    from PIL import Image, ImageOps

    report_progress("conditioning_image")
    path = request.inputs["input_image"]
    payload = path.read_bytes()
    with Image.open(BytesIO(payload)) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
    input_hash = hashlib.sha256(payload).hexdigest()
    if settings["preserve_aspect_ratio"]:
        ratio = source.width / source.height
        multiple = math.lcm(32, request.model.defaults["requirements"].get("dimension_multiple", 32))
        width = max(multiple, int(math.sqrt(width * height * ratio)) // multiple * multiple)
        height = max(multiple, int(math.sqrt(settings["width"] * settings["height"] / ratio)) // multiple * multiple)
        conditioning = source.resize((width, height), Image.Resampling.LANCZOS)
    else:
        conditioning = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
    return (
        width,
        height,
        conditioning,
        {
            "input_image": {
                "path": str(path),
                "sha256": input_hash,
                "source_size": list(source.size),
                "conditioning_size": list(conditioning.size),
            },
        },
    )

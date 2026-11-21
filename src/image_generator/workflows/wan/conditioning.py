"""Workflow-owned source preprocessing."""

import math


def prepare_conditioning(source, request, multiples):
    from PIL import Image, ImageOps

    height_multiple, width_multiple = multiples
    settings = request.settings
    height, width = settings["height"], settings["width"]
    if settings["preserve_aspect_ratio"]:
        ratio = source.height / source.width
        height = max(height_multiple, int(math.sqrt(height * width * ratio)) // height_multiple * height_multiple)
        width = max(
            width_multiple,
            int(math.sqrt(settings["height"] * settings["width"] / ratio)) // width_multiple * width_multiple,
        )
        conditioning = source.resize((width, height), Image.Resampling.LANCZOS)
    else:
        if height % height_multiple or width % width_multiple:
            raise ValueError(f"Loaded WAN model requires height/width multiples of {height_multiple}/{width_multiple}")
        conditioning = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
    return conditioning

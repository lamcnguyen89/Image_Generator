"""Resolve example media without exposing arbitrary files from asset folders."""

MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".avif": "image/avif",
    ".mp4": "video/mp4", ".webm": "video/webm",
    ".mov": "video/quicktime", ".qt": "video/quicktime",
}


def example_media(asset):
    """Prefer the configured preview, then an existing example.* in this asset.

    Resolve symlinks before checking containment. Only supported media within
    this asset's directory may be served, never weights or unrelated files.
    """
    directory = asset.manifest_path.parent.resolve()
    configured = asset.defaults.get("preview_image")
    candidates = [directory / configured] if configured else []
    candidates.extend(sorted(
        (path for path in directory.iterdir() if path.name.lower().startswith("example.")),
        key=lambda path: path.name.lower(),
    ))
    for candidate in candidates:
        path = candidate.resolve()
        if (path.is_relative_to(directory) and path.is_file()
                and path.suffix.lower() in MEDIA_TYPES):
            return path
    return None

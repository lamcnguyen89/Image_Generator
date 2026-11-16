"""Find local example media without opening weights or importing image libraries."""

MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".avif": "image/avif",
    ".mp4": "video/mp4", ".webm": "video/webm",
    ".mov": "video/quicktime", ".qt": "video/quicktime",
}


def example_media(asset):
    """List the configured preview first, then local media in filename order.

    Resolve symlinks before checking containment. Only supported media within
    this asset's directory may be served, never weights or files outside it.
    Directly copied media can have any filename. Ignore hidden files and do not
    scan subfolders; an explicitly configured preview can live in a subfolder.
    """
    directory = asset.manifest_path.parent.resolve()
    configured = asset.defaults.get("preview_image")
    candidates = [directory / configured] if configured else []
    candidates.extend(sorted(
        (path for path in directory.iterdir() if not path.name.startswith(".")),
        key=lambda path: (path.name.casefold(), path.name),
    ))
    media = []
    seen = set()
    for candidate in candidates:
        path = candidate.resolve()
        if (path.is_relative_to(directory) and path.is_file()
                and path.suffix.lower() in MEDIA_TYPES and path not in seen):
            media.append(path)
            seen.add(path)
    return media

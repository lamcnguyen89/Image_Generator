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
    Scan examples/ and, for older catalogs, the asset directory itself. Media
    can have any filename. Ignore hidden files and other subfolders; an
    explicitly configured preview can live in any asset subfolder.
    """
    directory = asset.manifest_path.parent.resolve()
    configured = asset.defaults.get("preview_image")
    candidates = [directory / configured] if configured else []
    folders = [directory / "examples", directory]
    candidates.extend(sorted(
        (path for folder in folders
         if folder.resolve().is_relative_to(directory) and folder.is_dir()
         for path in folder.iterdir() if not path.name.startswith(".")),
        key=lambda path: (path.name.casefold(), path.name, path.as_posix()),
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

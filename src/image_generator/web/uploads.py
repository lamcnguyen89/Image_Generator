"""Bounded image validation and immutable, retry-stable job input storage."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from io import BytesIO
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import warnings

from flask import current_app

from image_generator.jobs import SubmissionConflict


IMAGE_EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "TIFF": "tiff", "GIF": "gif"}


def validate_image_upload(upload):
    """Decode for validation only; keep original bytes, orientation, and colors."""
    from PIL import Image, UnidentifiedImageError

    payload = upload.stream.read(current_app.config["INPUT_MAX_BYTES"] + 1)
    if len(payload) > current_app.config["INPUT_MAX_BYTES"]:
        raise ValueError("Source image exceeds the upload size limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                extension = IMAGE_EXTENSIONS.get(image.format)
                if extension is None:
                    raise ValueError("Choose a PNG, JPEG, WebP, BMP, TIFF, or GIF image.")
                if image.width * image.height > current_app.config["INPUT_MAX_PIXELS"]:
                    raise ValueError("Source image exceeds the decoded pixel limit.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Choose a single-frame source image.")
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Source image is invalid, damaged, or too large to decode.") from error
    return payload, extension


def validate_upload(spec, upload):
    if spec.kind != "image":
        raise ValueError(f"Unsupported input kind {spec.kind!r}")
    return validate_image_upload(upload)


@dataclass
class StagedSources:
    paths: dict[str, Path]
    retained: bool = False

    @property
    def path(self):
        return self.paths["input_image"]

    def retain(self):
        self.retained = True


StagedSource = StagedSources


def _source_name(name, extension):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or extension not in IMAGE_EXTENSIONS.values():
        raise ValueError("Invalid source name or extension")
    return f"source.{extension}" if name == "input_image" else f"input-{name}.{extension}"


@contextmanager
def stage_sources(submission_key, validated):
    """Stage all named inputs atomically under one retry-stable submission lock."""
    if not re.fullmatch(r"[0-9a-f]{32}", submission_key):
        raise ValueError("Invalid submission key")
    names = {name: _source_name(name, value[1]) for name, value in validated.items()}
    root = Path(current_app.config["JOB_INPUT_ROOT"]).expanduser().resolve()
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    if locks.is_symlink():
        raise ValueError("Input lock directory must not be a symlink")
    with (locks / submission_key).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory = root / submission_key
        if directory.is_symlink():
            raise ValueError("Input directory must not be a symlink")
        directory.mkdir(exist_ok=True)
        staged = StagedSources({name: directory / filename for name, filename in names.items()})
        created = []
        try:
            existing = set(directory.iterdir())
            if existing:
                if existing != set(staged.paths.values()) or any(
                    path.is_symlink() or path.read_bytes() != validated[name][0]
                    for name, path in staged.paths.items()
                ):
                    raise SubmissionConflict("This form already has different source files. Reload to create another job.")
            else:
                for name, path in staged.paths.items():
                    with NamedTemporaryFile(dir=directory, prefix=".source-", delete=False) as stream:
                        temporary = Path(stream.name)
                        try:
                            stream.write(validated[name][0])
                            stream.flush()
                            temporary.chmod(0o400)
                            temporary.replace(path)
                            created.append(path)
                        finally:
                            temporary.unlink(missing_ok=True)
            yield staged
        finally:
            if not staged.retained:
                for path in created:
                    path.unlink(missing_ok=True)
            if not any(directory.iterdir()):
                directory.rmdir()


@contextmanager
def stage_source(submission_key, validated):
    """Compatibility entry point for callers staging a single source image."""
    with stage_sources(submission_key, {"input_image": validated}) as staged:
        yield staged


def managed_source_path(job, name="input_image"):
    """Only a registered source inside the configured input root is public."""
    raw = job.request.get("inputs", {}).get(name)
    if not raw:
        return None
    root = Path(current_app.config["JOB_INPUT_ROOT"]).expanduser().resolve()
    path = Path(raw)
    try:
        allowed = {_source_name(name, extension) for extension in IMAGE_EXTENSIONS.values()}
    except ValueError:
        return None
    if (path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(root)
            or path.parent.parent != root or not re.fullmatch(r"[0-9a-f]{32}", path.parent.name)
            or path.name not in allowed or not path.is_file()):
        return None
    return path

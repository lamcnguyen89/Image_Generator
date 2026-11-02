"""Remove finished jobs and their managed files without following symlinks."""

from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import re
import shutil

from .jobs import JobDeletionConflict, TERMINAL_STATES


def _remove_directory(directory):
    # Never resolve the job directory itself: a replaced directory or nested
    # symlink must not turn deletion into a traversal outside the managed root.
    if directory.is_symlink():
        directory.unlink()
    elif directory.exists():
        if not directory.is_dir():
            raise ValueError(f"Expected a job directory: {directory}")
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            # A concurrent deletion of the same finished job is harmless.
            pass


def delete_job(store, identifier, *, output_roots, input_root):
    """Delete owned files first, retaining history for retry if cleanup fails.

    Terminal jobs cannot be claimed again. File I/O stays outside SQLite write
    transactions. Per-submission upload locks serialize cleanup with form retries.
    Original external input files and model assets are never removed.
    """
    job = store.get(identifier)
    if job.status not in TERMINAL_STATES:
        raise JobDeletionConflict("Wait for this job to finish before deleting it.")
    if not re.fullmatch(r"[0-9a-f]{32}", job.id):
        raise ValueError("Invalid job directory identifier.")
    roots = {Path(root).expanduser().resolve() for root in output_roots}
    input_root = Path(input_root).expanduser().resolve()
    keys = store.submission_keys(identifier)
    source_directories = {
        input_root / key for key in keys if re.fullmatch(r"[0-9a-f]{32}", key)
        and any(Path(path).parent == input_root / key for path in job.request.get("inputs", {}).values())
    }
    with ExitStack() as stack:
        if source_directories:
            locks = input_root / ".locks"
            if locks.is_symlink():
                raise ValueError("Input lock directory must not be a symlink.")
            locks.mkdir(parents=True, exist_ok=True)
            for directory in sorted(source_directories):
                descriptor = os.open(locks / directory.name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                lock = stack.enter_context(os.fdopen(descriptor, "a+b"))
                fcntl.flock(lock, fcntl.LOCK_EX)
        # Recheck after any upload-lock wait. Another request may have deleted it.
        job = store.get(identifier)
        if job.status not in TERMINAL_STATES:
            raise JobDeletionConflict("Wait for this job to finish before deleting it.")
        for directory in sorted({root / job.id for root in roots} | source_directories):
            _remove_directory(directory)
        store.delete_finished(identifier)

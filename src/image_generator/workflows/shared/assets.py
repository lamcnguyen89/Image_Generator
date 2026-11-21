"""Shared sampling helpers."""


def file_fingerprint(path):
    if path is None:
        return None
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns

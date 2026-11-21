"""Shared seeds helpers."""

import secrets


def image_seeds(settings, count):
    seed = settings["seed"]
    return [secrets.randbelow(2**63) if seed is None else (seed + index) % (2**63) for index in range(count)]

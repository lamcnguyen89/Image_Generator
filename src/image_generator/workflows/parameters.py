"""Typed parameter schema used by forms, CLI, and validation; no Flask imports."""

import math
import re
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Parameter:
    name: str
    label: str
    type: Literal["text", "textarea", "integer", "number", "boolean", "select", "string_list", "number_list", "file"]
    group: str
    help: str = ""
    required: bool = False
    nullable: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    scope: Literal["settings", "inputs"] = "settings"
    read_only: bool = False
    cli_flags: tuple[str, ...] = ()
    validation_message: str = ""
    inherit_when_absent: bool = False
    choice_source: str = ""
    choices_by_source: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def validate(self, value: Any) -> str | None:
        if value is None and self.nullable:
            return None
        if self.type in {"text", "textarea", "select"}:
            if not isinstance(value, str):
                return "Enter text."
            if self.required and not value.strip():
                return "This field is required."
            if self.type == "select" and value not in dict(self.choices):
                return "Choose one of the available options."
        elif self.type == "boolean":
            if type(value) is not bool:
                return "Choose a boolean value."
        elif self.type == "string_list":
            if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                return "Enter one nonempty value per line."
        elif self.type == "number_list":
            if not isinstance(value, list) or not value:
                return "Enter a nonempty list of numbers, or leave blank for the default."
            try:
                if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
                    return "Every list entry must be a finite number."
            except OverflowError:
                return "Number is too large."
        elif self.type in {"integer", "number"}:
            if self.type == "integer" and type(value) is not int:
                return "Enter a whole number."
            if type(value) not in (int, float):
                return "Enter a number."
            try:
                if not math.isfinite(value):
                    return "Enter a finite number."
            except OverflowError:
                return "Number is too large."
            if self.minimum is not None and value < self.minimum:
                return f"Minimum value is {self.minimum}."
            if self.maximum is not None and value > self.maximum:
                return f"Maximum value is {self.maximum}."
            if self.step is not None and (value - (self.minimum or 0)) % self.step:
                return f"Use increments of {self.step} starting at {self.minimum or 0}."
        return None

    def parse(self, raw):
        if self.type == "boolean":
            if raw not in {"", "on"}:
                raise ValueError("Invalid checkbox value")
            return raw == "on"
        if not raw.strip() and self.nullable:
            return None
        if self.type == "integer":
            return int(raw)
        if self.type == "number":
            return float(raw)
        if self.type == "string_list":
            return [line.strip() for line in raw.splitlines() if line.strip()]
        if self.type == "number_list":
            return [float(item) for item in re.split(r"[\s,]+", raw.strip())]
        return raw

    def display_value(self, value):
        if self.type == "string_list":
            return "\n".join(value or [])
        if self.type == "number_list":
            return "\n".join(map(str, value)) if value is not None else ""
        return "" if value is None else value

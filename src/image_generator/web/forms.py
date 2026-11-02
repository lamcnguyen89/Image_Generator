"""Translate HTML form values to workflow settings, without running generation."""

from copy import deepcopy
import math
import re

from image_generator.catalog.persistence import adapter_unavailable
from image_generator.catalog.settings import lora_strength_minimum
from image_generator.workflows.ui import set_setting, setting_value


def display_values(ui, settings, form=None):
    values = {}
    for field in ui.fields:
        if field.scope != "settings":
            continue
        if form is not None:
            value = form.get(field.name, "") if field.type != "boolean" else field.name in form
        else:
            value = setting_value(settings, field.name)
            if field.type == "string_list":
                value = "\n".join(value or [])
            elif field.type == "number_list":
                value = "\n".join(map(str, value)) if value is not None else ""
        values[field.name] = "" if value is None else value
    return values


def parse_settings(ui, settings, form, *, allow_empty_prompt=False):
    result, errors = deepcopy(settings), {}
    for field in ui.fields:
        if field.scope != "settings":
            continue
        raw = form.get(field.name, "")
        try:
            if field.type == "boolean":
                if raw not in {"", "on"}:
                    raise ValueError("Invalid checkbox value")
                value = raw == "on"
            elif not raw.strip() and field.nullable:
                value = None
            elif field.type == "integer":
                value = int(raw)
            elif field.type == "number":
                value = float(raw)
            elif field.type == "string_list":
                value = [line.strip() for line in raw.splitlines() if line.strip()]
            elif field.type == "number_list":
                value = [float(item) for item in re.split(r"[\s,]+", raw.strip())]
            else:
                value = raw
            error = field.validate(value)
            if allow_empty_prompt and field.name == "prompt" and value == "":
                error = None
            if error:
                errors[field.name] = error
            else:
                set_setting(result, field.name, value)
        except (ValueError, OverflowError):
            errors[field.name] = "Enter a whole number." if field.type == "integer" else "Enter a valid value."
    return result, errors


def lora_options(catalog, model, settings, form=None):
    links = {link["id"]: link for link in settings.get("loras", [])}
    selected = set(form.getlist("lora")) if form is not None else None
    configured = set(form.getlist("configured_lora")) if form is not None else set(links)
    options = []
    for asset in sorted(catalog.loras.values(), key=lambda item: (item.display_name.casefold(), item.id)):
        compatibility = asset.defaults["compatibility"]
        target = compatibility.get("expert_target")
        reason = adapter_unavailable(model, asset)
        if reason and asset.id not in links and not (asset.family == model.family and model.family in {"wan", "minimax"}):
            continue
        resolved = asset.resolve_settings()
        link = links.get(asset.id, {})
        strength = link.get("strength", resolved.get("strength", 0.8))
        strength_source = "override" if "strength" in link or not link else "default"
        options.append({
            "asset": asset, "settings": resolved, "eligible": not reason, "reason": reason,
            "enabled": (asset.id in selected if selected is not None else bool(link.get("enabled"))) and not reason,
            "configured": asset.id in configured or (selected is not None and asset.id in selected),
            "strength_source": form.get(f"strength_source:{asset.id}", strength_source) if form is not None else strength_source,
            "strength": form.get(f"strength:{asset.id}", strength) if form is not None else strength,
            "verified": model.id in compatibility.get("verified_checkpoint_ids", []),
            "minimum_strength": lora_strength_minimum(asset.family),
            "expert": target if target in {"high_noise", "low_noise"} and not reason else "unknown",
        })
    return options


def parse_loras(options, form, *, preserve_disabled=False):
    enabled = form.getlist("lora")
    configured = form.getlist("configured_lora")
    selected = configured + [item for item in enabled if item not in configured] if preserve_disabled else enabled
    errors, links = {}, []
    if len(selected) != len(set(selected)) or len(enabled) != len(set(enabled)):
        errors["loras"] = "Select each adapter only once."
    by_id = {item["asset"].id: item for item in options}
    for identifier in dict.fromkeys(selected):
        item = by_id.get(identifier)
        if item is None or (not item["eligible"] and (not preserve_disabled or identifier in enabled)):
            errors["loras"] = "A selected adapter is unavailable for this model."
            continue
        source = form.get(f"strength_source:{identifier}", "override")
        if source not in {"default", "override"}:
            errors[f"strength:{identifier}"] = "Choose adapter default or checkpoint override."
            continue
        try:
            strength = float(form.get(f"strength:{identifier}", ""))
            minimum = item["minimum_strength"]
            if not math.isfinite(strength) or (minimum is not None and strength < minimum):
                raise ValueError
        except (ValueError, OverflowError):
            errors[f"strength:{identifier}"] = (
                "Strength must be a finite number." if item["minimum_strength"] is None
                else "Strength must be a finite number at least zero."
            )
            continue
        link = {"id": identifier, "enabled": identifier in enabled}
        if source == "override":
            link["strength"] = strength
        elif not preserve_disabled:
            link["strength"] = item["settings"].get("strength", 0.8)
        links.append(link)
    return links, errors

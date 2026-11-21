"""Translate HTML form values to workflow settings, without running generation."""

from copy import deepcopy
from dataclasses import replace
import math

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
            value = field.display_value(value)
        values[field.name] = "" if value is None else value
    return values


def parse_settings(ui, settings, form, *, allow_empty_prompt=False):
    result, errors = deepcopy(settings), {}
    for field in ui.fields:
        if field.scope != "settings":
            continue
        raw = form.get(field.name, "")
        try:
            value = field.parse(raw)
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
    from image_generator.workflows.adapters import policy_for_model
    policy = policy_for_model(model)
    links = {link["id"]: link for link in settings.get("loras", [])}
    selected = set(form.getlist("lora")) if form is not None else None
    configured = set(form.getlist("configured_lora")) if form is not None else set(links)
    options = []
    for asset in sorted(catalog.loras.values(), key=lambda item: (item.display_name.casefold(), item.id)):
        compatibility = asset.defaults["compatibility"]
        resolved = asset.resolve_settings()
        link = links.get(asset.id, {})
        link_values = {field.name: (form.get(f"{field.name}:{asset.id}", policy.field_value(asset, link, field.name))
                                   if form is not None else policy.field_value(asset, link, field.name))
                       for field in policy.link_fields}
        reason = policy.unavailable(model, asset, link=link_values, require_complete=False)
        if reason and asset.id not in links and not (asset.family == model.family and policy.retain_unavailable):
            continue
        strength = link.get("strength", resolved.get("strength", 0.8))
        strength_source = "override" if "strength" in link or not link else "default"
        group = link_values.get(policy.group_field)
        options.append({
            "asset": asset, "settings": resolved, "eligible": not reason, "reason": reason,
            "enabled": (asset.id in selected if selected is not None else bool(link.get("enabled"))) and not reason,
            "configured": asset.id in configured or (selected is not None and asset.id in selected),
            "strength_source": form.get(f"strength_source:{asset.id}", strength_source) if form is not None else strength_source,
            "strength": form.get(f"strength:{asset.id}", strength) if form is not None else strength,
            "verified": model.id in compatibility.get("verified_checkpoint_ids", []),
            "minimum_strength": policy.minimum_strength,
            "group": group if group in dict(policy.groups) and not reason else "unknown",
            "link_fields": policy.link_fields,
            "link_controls": [(replace(field, name=f"{field.name}:{asset.id}", required=False),
                               {f"{field.name}:{asset.id}": field.display_value(link_values[field.name])})
                              for field in policy.link_fields],
            "link_values": link_values,
            "link_defaults": {field.name: policy.field_value(asset, {}, field.name) for field in policy.link_fields},
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
        for field in item["link_fields"]:
            raw = form.get(f"{field.name}:{identifier}", "" if field.type == "boolean" else item["link_values"][field.name])
            if raw in (None, "") and not link["enabled"]:
                continue
            try:
                value = field.parse(str(raw) if raw is not None else "")
                error = field.validate(value)
            except (ValueError, OverflowError):
                value, error = None, "Enter a valid value."
            if error:
                errors[f"{field.name}:{identifier}"] = field.validation_message or error
            elif value != item["link_defaults"][field.name]:
                link[field.name] = value
        if source == "override":
            link["strength"] = strength
        elif not preserve_disabled:
            link["strength"] = item["settings"].get("strength", 0.8)
        links.append(link)
    return links, errors

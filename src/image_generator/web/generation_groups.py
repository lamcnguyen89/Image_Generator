"""Present related checkpoints as one card while keeping requests model-specific."""


def _group_key(model):
    group = model.defaults.get("generation_group")
    return (model.family, model.defaults["workflow"], group["id"]) if group else (model.id,)


def generation_title(model):
    group = model.defaults.get("generation_group")
    return group["display_name"] if group else model.display_name


def group_members(catalog, model):
    key = _group_key(model)
    return sorted((item for item in catalog.models.values() if _group_key(item) == key),
                  key=lambda item: (item.display_name.casefold(), item.id))


def generation_targets(catalog, model, implemented_modes):
    """Resolve every offered mode to exactly one concrete catalog checkpoint."""
    members = group_members(catalog, model)
    targets = {}
    for mode in implemented_modes:
        candidates = [item for item in members if mode in item.defaults["supported_modes"]]
        if len(candidates) > 1:
            raise ValueError(f"This model group has multiple checkpoints for {mode}; give them separate generation groups.")
        if candidates:
            targets[mode] = candidates[0]
    return targets


def catalog_cards(catalog, ordered_models):
    seen = set()
    for model in ordered_models:
        key = _group_key(model)
        if key in seen:
            continue
        seen.add(key)
        group = model.defaults.get("generation_group") or {}
        yield {
            "model": model, "title": generation_title(model),
            "description": group.get("description", model.defaults.get("description", "")),
            "modes": tuple(dict.fromkeys(mode for item in group_members(catalog, model)
                                        for mode in item.defaults["supported_modes"])),
        }

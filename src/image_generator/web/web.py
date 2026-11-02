"""Read-only catalog routes. Browsing never imports generation workflows."""

from flask import Blueprint, abort, current_app, g, render_template

from image_generator.catalog import CatalogValidationError, discover_catalog


catalog_bp = Blueprint(
    "catalog", __name__, template_folder="templates",
    static_folder="static", static_url_path="/catalog-assets",
)

FAMILY_ORDER = {"flux": 0, "qwen_2_1": 1, "krea2": 2, "sdxl": 3, "minimax": 4, "wan": 5}


@catalog_bp.app_template_filter("family_label")
def family_label(family):
    return "QWEN 2.1" if family == "qwen_2_1" else family.upper()


def get_catalog():
    """Discover once per request so the next refresh sees manifest edits."""
    if "model_catalog" not in g:
        g.model_catalog = discover_catalog(current_app.config["CATALOG_ROOT"])
    return g.model_catalog


@catalog_bp.app_template_filter("mode_label")
def mode_label(mode):
    return {
        "txt2img": "Text to image", "img2img": "Image to image",
        "txt2vid": "Text to video", "img2vid": "Image to video",
    }.get(mode, mode)


@catalog_bp.get("/")
def index():
    catalog = get_catalog()
    groups = {}
    for model in sorted(catalog.models.values(), key=lambda item: (
        FAMILY_ORDER.get(item.family, len(FAMILY_ORDER)),
        item.family, item.display_name.casefold(), item.id,
    )):
        group = groups.setdefault(model.family, {"name": model.family, "models": [], "lora_count": 0})
        group["models"].append(model)
    for lora in catalog.loras.values():
        if lora.family in groups:
            groups[lora.family]["lora_count"] += 1
    return render_template(
        "catalog.html", families=list(groups.values()),
        model_count=len(catalog.models), lora_count=len(catalog.loras),
    )


@catalog_bp.get("/models/<model_id>")
def model_settings(model_id):
    catalog = get_catalog()
    try:
        model = catalog.get_model(model_id)
    except KeyError:
        abort(404)
    settings = model.resolve_settings()
    linked_loras = []
    for link in settings.get("loras", []):
        lora = catalog.get_lora(link["id"])
        lora_settings = lora.resolve_settings()
        linked_loras.append({
            "asset": lora, "enabled": link["enabled"],
            "strength": link.get("strength", lora_settings.get("strength")),
            "settings": lora_settings,
            "verified": model.id in lora.defaults["compatibility"].get("verified_checkpoint_ids", []),
        })
    return render_template(
        "model_settings.html", model=model, settings=settings, linked_loras=linked_loras,
    )


@catalog_bp.app_errorhandler(CatalogValidationError)
def invalid_catalog(error):
    return render_template(
        "error.html", title="Catalog unavailable",
        message="The catalog could not be read. Correct the configuration errors below, then refresh this page.",
        issues=error.issues,
    ), 503


@catalog_bp.app_errorhandler(404)
def page_not_found(error):
    return render_template(
        "error.html", title="Page not found",
        message="This page or model is not in the catalog. Return to the catalog to choose a model.",
    ), 404

"""Load trusted integration templates and serve package-local static assets."""

from flask import Blueprint, abort, send_from_directory
from jinja2 import BaseLoader, ChoiceLoader, TemplateNotFound

from image_generator.workflows.registry import get_definition

workflow_assets_bp = Blueprint("workflow_assets", __name__)


class WorkflowTemplateLoader(BaseLoader):
    def get_source(self, environment, template):
        name, separator, relative = template.partition("/")
        if not separator:
            raise TemplateNotFound(template)
        try:
            definition = get_definition(name)
        except ValueError:
            raise TemplateNotFound(template) from None
        if definition.asset_root is None:
            raise TemplateNotFound(template)
        root = (definition.asset_root / "templates").resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise TemplateNotFound(template)
        source = path.read_text(encoding="utf-8")
        mtime = path.stat().st_mtime_ns
        return source, str(path), lambda: path.is_file() and path.stat().st_mtime_ns == mtime


def install_workflow_assets(app):
    app.jinja_loader = ChoiceLoader([app.jinja_loader, WorkflowTemplateLoader()])
    app.jinja_env.globals["workflow_template"] = lambda name, path: f"{name}/{path}"
    app.register_blueprint(workflow_assets_bp)


@workflow_assets_bp.get("/workflows/<workflow>/static/<path:filename>")
def static(workflow, filename):
    try:
        definition = get_definition(workflow)
    except ValueError:
        abort(404)
    if definition.asset_root is None:
        abort(404)
    root = (definition.asset_root / "static").resolve()
    if not (root / filename).resolve().is_relative_to(root):
        abort(404)
    return send_from_directory(root, filename)

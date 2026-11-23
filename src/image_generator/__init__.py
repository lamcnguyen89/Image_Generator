"""Local image and video generation application."""


def create_app(config=None):
    """Create the catalog, generation forms, and job UI without GPU imports."""
    from pathlib import Path
    import os
    from secrets import token_hex

    from flask import Flask

    from .web.web import catalog_bp
    from .web.generation_routes import generation_bp
    from .web.job_routes import jobs_bp
    from .web.settings_routes import settings_bp
    from .web.system_routes import system_bp
    from .web.auth_routes import auth_bp

    app = Flask(__name__, static_folder=None)
    app.config.from_mapping(
        CATALOG_ROOT=Path(__file__).resolve().parents[2] / "Models/Model_Families",
        JOB_DATABASE=Path(__file__).resolve().parents[2] / "data/jobs.sqlite3",
        JOB_OUTPUT_ROOT=Path(__file__).resolve().parents[2] / "Images",
        JOB_VIDEO_OUTPUT_ROOT=Path(__file__).resolve().parents[2] / "Videos",
        JOB_INPUT_ROOT=Path(__file__).resolve().parents[2] / "data/inputs",
        MAX_CONTENT_LENGTH=24 * 1024 * 1024,
        INPUT_MAX_BYTES=20 * 1024 * 1024,
        INPUT_MAX_PIXELS=25_000_000,
        SESSION_COOKIE_SAMESITE="Lax",
        # This local UI is edited in place while the long-running worker runs.
        TEMPLATES_AUTO_RELOAD=True,
    )
    app.config.from_prefixed_env(prefix="IMAGE_GENERATOR")
    # Passwords such as "1234" or "false" must stay strings, not parsed JSON.
    app.config["ACCESS_PASSWORD"] = os.environ.get("IMAGE_GENERATOR_ACCESS_PASSWORD", "")
    if config is not None:
        app.config.update(config)
    from .diagnostics import configure_logging, install_request_logging
    if app.config.get("LOG_ENABLED", not app.testing):
        configure_logging(app.config.get("LOG_DIR"), app.config.get("LOG_LEVEL"))
        install_request_logging(app)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = token_hex(32)
    if not isinstance(app.config["ACCESS_PASSWORD"], str):
        raise ValueError("ACCESS_PASSWORD must be a string")
    app.register_blueprint(auth_bp)
    app.register_blueprint(catalog_bp)
    app.register_blueprint(generation_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(system_bp)
    from .web.workflow_assets import install_workflow_assets
    install_workflow_assets(app)
    return app

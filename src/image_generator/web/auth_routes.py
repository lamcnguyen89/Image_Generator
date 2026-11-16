"""Optional shared-password gate for the entire web service."""

from hmac import compare_digest, digest
from urllib.parse import urlsplit

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for

from .security import csrf_token, valid_csrf


auth_bp = Blueprint("auth", __name__)


def password_stamp():
    """Bind the login to this password without putting the password in the cookie."""
    key = current_app.secret_key
    if isinstance(key, str):
        key = key.encode("utf-8")
    message = b"access-login\0" + current_app.config["ACCESS_PASSWORD"].encode("utf-8")
    return digest(key, message, "sha256").hex()


def signed_in():
    stamp = session.get("access_login")
    return isinstance(stamp, str) and compare_digest(stamp, password_stamp())


def next_page(value):
    # Only redirect within this service, including when the link was supplied by a visitor.
    if (not value or not value.startswith("/") or value.startswith("//")
            or "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        return url_for("catalog.index")
    parts = urlsplit(value)
    if parts.scheme or parts.netloc or parts.path in {"/login", "/logout"}:
        return url_for("catalog.index")
    return value


PUBLIC_ASSETS = {"catalog.css", "favicon.ico"}


def public_asset():
    return (request.endpoint == "catalog.static"
            and request.view_args.get("filename") in PUBLIC_ASSETS)


@auth_bp.before_app_request
def require_password():
    if not current_app.config["ACCESS_PASSWORD"]:
        return None
    if request.endpoint == "auth.login":
        return None
    # Only the sign-in page's stylesheet and favicon are public. Media and API routes stay gated.
    if public_asset():
        return None
    if signed_in():
        return None
    destination = request.full_path.rstrip("?") if request.method in {"GET", "HEAD"} else request.path
    login_url = url_for("auth.login", next=destination)
    if request.path.startswith("/api/"):
        return jsonify(error="Sign in to access Midnight Generator.", login_url=login_url), 401
    return redirect(login_url, code=303)


@auth_bp.after_app_request
def private_response(response):
    if current_app.config["ACCESS_PASSWORD"] and not public_asset():
        response.cache_control.no_store = True
        response.cache_control.private = True
        response.vary.add("Cookie")
    return response


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    destination = next_page(request.args.get("next", ""))
    if not current_app.config["ACCESS_PASSWORD"] or signed_in():
        return redirect(destination, code=303)
    error = None
    status = 200
    if request.method == "POST":
        if not valid_csrf(request.form.get("csrf_token")):
            error, status = "Session expired. Please try again.", 400
        elif compare_digest(request.form.get("password", "").encode("utf-8"),
                            current_app.config["ACCESS_PASSWORD"].encode("utf-8")):
            session.clear()
            session["access_login"] = password_stamp()
            csrf_token()
            return redirect(destination, code=303)
        else:
            error, status = "Incorrect password.", 401
    return render_template("login.html", error=error, next_page=destination), status


@auth_bp.post("/logout")
def logout():
    if not valid_csrf(request.form.get("csrf_token")):
        return "Session expired. Refresh the page and try again.", 400
    session.clear()
    return redirect(url_for("auth.login"), code=303)

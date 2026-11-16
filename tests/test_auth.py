"""Exercise the shared-password gate, including direct service and media access."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app


class PasswordTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.password = "test café password #42"
        self.config = {"TESTING": True, "ACCESS_PASSWORD": self.password,
                       "SECRET_KEY": "auth-tests-only",
                       "CATALOG_ROOT": ROOT / "tests/fixtures/model_families",
                       "JOB_DATABASE": self.root / "jobs.sqlite3"}
        self.app = create_app(self.config)
        self.client = self.app.test_client()

    def token(self, client=None):
        client = client or self.client
        client.get("/login")
        with client.session_transaction() as session:
            return session["generation_csrf"]

    def login(self, password=None, next_page="/", client=None):
        client = client or self.client
        token = self.token(client)
        return client.post("/login", query_string={"next": next_page},
                           data={"password": self.password if password is None else password,
                                 "csrf_token": token})

    def test_pages_media_and_unknown_routes_require_login(self):
        for path in ("/", "/models/fixture-sdxl", "/generate/fixture-sdxl", "/jobs",
                     "/assets/fixture-sdxl/preview",
                     "/jobs/example", "/jobs/example/source",
                     "/jobs/example/outputs/1/media?download=1",
                     "/jobs/example/outputs/1/metadata", "/missing", "/catalog-assets/memory.js"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 303)
                target = urlsplit(response.location)
                self.assertEqual(target.path, "/login")
                self.assertEqual(parse_qs(target.query)["next"], [path])
                self.assertTrue(response.cache_control.no_store)
        self.assertFalse((self.root / "jobs.sqlite3").exists())

    def test_api_returns_401_json_without_touching_service(self):
        with patch("image_generator.web.system_routes.psutil.virtual_memory") as memory:
            for method, path in (("get", "/api/system/memory"),
                                 ("get", "/api/jobs/example"),
                                 ("post", "/api/system/models/unload"),
                                 ("post", "/api/assets/fixture-sdxl/examples"),
                                 ("post", "/api/models/fixture-sdxl/settings/field"),
                                 ("get", "/api/system/models/unload/example")):
                response = getattr(self.client, method)(path)
                self.assertEqual(response.status_code, 401)
                self.assertIn("error", response.json)
                self.assertTrue(response.json["login_url"].startswith("/login?"))
                self.assertTrue(response.cache_control.no_store)
            memory.assert_not_called()
        self.assertFalse((self.root / "jobs.sqlite3").exists())

    def test_protected_posts_do_not_modify_settings_or_enqueue(self):
        for path in ("/generate/fixture-sdxl", "/settings/models/fixture-sdxl",
                     "/settings/adapters/fixture-style"):
            response = self.client.post(path, data={"action": "generate"})
            self.assertEqual(response.status_code, 303)
            self.assertTrue(response.location.startswith("/login?"))
        self.assertFalse((self.root / "jobs.sqlite3").exists())

    def test_login_and_stylesheet_are_public_without_controls_or_secrets(self):
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn('type="password"', response.text)
        self.assertNotIn("Shared Memory", response.text)
        self.assertNotIn(self.password, response.text)
        self.assertTrue(response.cache_control.no_store)
        cookie = response.headers["Set-Cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        response = self.client.get("/catalog-assets/catalog.css")
        self.assertEqual(response.status_code, 200)
        response.close()
        response = self.client.get("/catalog-assets/favicon.ico")
        self.assertEqual(response.status_code, 200)
        response.close()

    def test_incorrect_and_missing_passwords_stay_locked(self):
        for password in ("wrong", "", self.password + " "):
            response = self.login(password)
            self.assertEqual(response.status_code, 401)
            self.assertIn("Incorrect password", response.text)
            self.assertNotIn(password if password else self.password, response.text)
            self.assertEqual(self.client.get("/api/system/memory").status_code, 401)

    def test_login_requires_same_session_csrf_and_handles_unicode_input(self):
        token = self.token()
        for invalid in (None, "wrong", "☃"):
            data = {"password": self.password}
            if invalid is not None:
                data["csrf_token"] = invalid
            self.assertEqual(self.client.post("/login", data=data).status_code, 400)
        other_client = self.app.test_client()
        response = other_client.post("/login", data={"password": self.password, "csrf_token": token})
        self.assertEqual(response.status_code, 400)

    def test_login_unlocks_pages_api_and_rotates_session(self):
        old_token = self.token()
        with self.client.session_transaction() as session:
            session["old_state"] = "discard"
        response = self.login(next_page="/jobs?view=recent")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.location, "/jobs?view=recent")
        with self.client.session_transaction() as session:
            self.assertIn("access_login", session)
            self.assertNotIn(self.password, str(dict(session)))
            self.assertNotEqual(session["generation_csrf"], old_token)
            self.assertNotIn("old_state", session)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Sign out", response.text)
        self.assertTrue(response.cache_control.no_store)
        self.assertEqual(self.client.get("/api/system/memory").status_code, 200)
        self.assertEqual(self.app.test_client().get("/").status_code, 303)

    def test_redirects_cannot_leave_the_service(self):
        for destination in ("https://example.com", "//example.com", "///example.com",
                            "/\\example.com", "/\n/evil", "/login", "/logout"):
            with self.subTest(destination=destination):
                client = self.app.test_client()
                response = self.login(next_page=destination, client=client)
                self.assertEqual(response.location, "/")

    def test_logout_requires_post_and_csrf_then_locks_every_route(self):
        self.login()
        self.assertEqual(self.client.get("/logout").status_code, 405)
        self.assertEqual(self.client.post("/logout").status_code, 400)
        self.assertEqual(self.client.get("/").status_code, 200)
        with self.client.session_transaction() as session:
            token = session["generation_csrf"]
        response = self.client.post("/logout", data={"csrf_token": token})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.location, "/login")
        self.assertEqual(self.client.get("/").status_code, 303)
        self.assertEqual(self.client.get("/api/system/memory").status_code, 401)

    def test_password_change_invalidates_sessions_even_with_same_secret(self):
        self.login()
        changed = create_app({**self.config, "ACCESS_PASSWORD": "new-password"})
        client = changed.test_client()
        client.set_cookie("session", self.client.get_cookie("session").value)
        self.assertEqual(client.get("/").status_code, 303)
        self.assertEqual(self.login(password="new-password", client=client).status_code, 303)
        self.assertEqual(client.get("/").status_code, 200)

    def test_unset_or_empty_password_preserves_open_access(self):
        for password in (None, ""):
            with patch.dict("os.environ", {}, clear=True):
                config = {**self.config}
                if password is None:
                    config.pop("ACCESS_PASSWORD")
                else:
                    config["ACCESS_PASSWORD"] = password
                client = create_app(config).test_client()
                response = client.get("/")
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("Sign out", response.text)
                self.assertEqual(client.get("/api/system/memory").status_code, 200)
                self.assertEqual(client.get("/login").location, "/")

    def test_env_password_is_always_a_string_and_explicit_config_wins(self):
        config = {key: value for key, value in self.config.items() if key != "ACCESS_PASSWORD"}
        for password in ("1234", "false", "null", '"quoted"', " leading and trailing "):
            with patch.dict("os.environ", {"IMAGE_GENERATOR_ACCESS_PASSWORD": password}):
                app = create_app(config)
                self.assertEqual(app.config["ACCESS_PASSWORD"], password)
                self.assertEqual(self.login(password=password, client=app.test_client()).status_code, 303)
                self.assertEqual(create_app(self.config).config["ACCESS_PASSWORD"], self.password)


if __name__ == "__main__":
    unittest.main()

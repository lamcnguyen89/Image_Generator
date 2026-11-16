"""Optional Firefox check for inline settings; use CAROUSEL_FIREFOX as in carousel tests."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

from flask import request
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from test_carousel_browser import QuietHandler


HARNESS = r"""
<script>
window.addEventListener("load", async () => {
  const results = [];
  const wait = () => new Promise((resolve) => setTimeout(resolve, 40));
  const until = async (condition) => {
    const end = performance.now() + 4000;
    while (!condition() && performance.now() < end) await wait();
    if (!condition()) throw new Error("Timed out waiting for inline settings update");
  };
  const check = (condition, message) => {
    if (!condition) throw new Error(message);
    results.push(message);
  };
  const row = (field) => document.querySelector(`[data-setting-row][data-field="${field}"]`);
  const editor = (field) => row(field).querySelector("[data-setting-editor]");
  const input = (field) => row(field).querySelector("[data-setting-input]");
  const display = (field) => row(field).querySelector("[data-setting-display]");
  const edit = (field) => row(field).querySelector("[data-setting-edit]").click();
  try {
    const fields = [...document.querySelectorAll("[data-setting-edit]")];
    check(fields.length > 10 && fields.every((button) => !button.hidden), "Every setting has a working Edit button");
    const original = display("negative_prompt").textContent;
    edit("negative_prompt");
    check(!editor("negative_prompt").hidden && display("negative_prompt").hidden, "Edit opens an inline editor");
    check(document.activeElement === input("negative_prompt"), "Editor receives keyboard focus");
    check(row("width").querySelector("[data-setting-edit]").disabled, "Other row editors wait for save or cancel");
    input("negative_prompt").value = "Unsaved text";
    editor("negative_prompt").querySelector("[data-setting-cancel]").click();
    check(editor("negative_prompt").hidden && display("negative_prompt").textContent === original, "Cancel preserves the saved value");
    const text = '<img src=x onerror="window.inlineInjected=true">\nblurry, café';
    edit("negative_prompt"); input("negative_prompt").value = text;
    editor("negative_prompt").requestSubmit();
    await until(() => editor("negative_prompt").hidden);
    check(JSON.parse(display("negative_prompt").textContent) === text, "Negative prompt saves exact plain text and line breaks");
    check(!window.inlineInjected && !display("negative_prompt").querySelector("img"), "Saved prompt text cannot inject HTML");
    check(JSON.parse(document.querySelector("[data-user-overrides]").textContent).negative_prompt === text,
          "Saved user overrides refresh after a row save");
    const originalWidth = display("width").textContent;
    edit("width"); input("width").value = "513"; editor("width").requestSubmit();
    await until(() => row("width").querySelector("[data-setting-error]").textContent);
    check(!editor("width").hidden && input("width").value === "513", "Validation errors preserve the entered value");
    check(display("width").textContent === originalWidth, "Invalid width does not replace the saved value");
    input("width").value = "512"; editor("width").requestSubmit();
    await until(() => editor("width").hidden);
    check(JSON.parse(display("width").textContent) === 512, "A corrected value saves using the updated revision");
    edit("seed"); input("seed").value = "9223372036854775807"; editor("seed").requestSubmit();
    await until(() => editor("seed").hidden);
    check(display("seed").textContent === "9223372036854775807", "Large integer seed retains exact precision");
    check(document.querySelector("[data-user-overrides]").textContent.includes("9223372036854775807"),
          "Source settings also retain exact seed precision");
    edit("negative_prompt");
    input("negative_prompt").value = "Discard with Escape";
    input("negative_prompt").dispatchEvent(new KeyboardEvent("keydown", {key:"Escape", bubbles:true, cancelable:true}));
    check(editor("negative_prompt").hidden && input("negative_prompt").value === text, "Escape discards unsaved text");
    edit("negative_prompt"); input("negative_prompt").value = "Keep this unsaved edit";
    await fetch("/browser-external-change", {method:"POST"});
    editor("negative_prompt").requestSubmit();
    await until(() => row("negative_prompt").querySelector("[data-setting-error]").textContent);
    check(input("negative_prompt").value === "Keep this unsaved edit" && !editor("negative_prompt").hidden,
          "Conflicting editor changes preserve unsaved input");
    check(row("negative_prompt").querySelector("[data-setting-error]").textContent.includes("changed"),
          "Conflicts explain that saved settings changed");
  } catch (error) { results.push("FAIL: " + error.message); }
  await fetch("/browser-results", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(results)});
});
</script>
"""


@unittest.skipUnless(os.environ.get("CAROUSEL_FIREFOX"), "Set CAROUSEL_FIREFOX to run real-browser checks")
class InlineSettingsBrowserTests(unittest.TestCase):
    def test_edit_save_cancel_errors_and_revisions_in_browser(self):
        with tempfile.TemporaryDirectory(prefix="inline-settings-browser-") as temporary:
            root = Path(temporary)
            catalog = root / "families"
            shutil.copytree(ROOT / "tests/fixtures/model_families", catalog)
            app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": catalog})
            results = []
            done = threading.Event()

            @app.after_request
            def inject_harness(response):
                if request.path == "/models/fixture-sdxl" and response.mimetype == "text/html":
                    response.set_data(response.get_data(as_text=True).replace("</body>", HARNESS + "</body>"))
                return response

            @app.post("/browser-external-change")
            def external_change():
                path = catalog / "SDXL/checkpoints/demo/model.user.json"
                document = json.loads(path.read_text())
                document["settings"]["seed"] = 100
                path.write_text(json.dumps(document))
                return {"ok": True}

            @app.post("/browser-results")
            def browser_results():
                results.extend(request.get_json())
                done.set()
                return {"ok": True}

            server = make_server("127.0.0.1", 0, app, threaded=True, request_handler=QuietHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            profile = root / "profile"
            profile.mkdir()
            with (root / "firefox.log").open("w+") as log:
                browser = None
                try:
                    browser = subprocess.Popen([
                        os.environ["CAROUSEL_FIREFOX"], "--headless", "--no-remote", "--profile", str(profile),
                        f"http://127.0.0.1:{server.server_port}/models/fixture-sdxl",
                    ], stdout=log, stderr=log)
                    completed = done.wait(30)
                    log.flush()
                    log.seek(0)
                    self.assertTrue(completed, log.read()[-3000:])
                    self.assertFalse([item for item in results if item.startswith("FAIL:")], json.dumps(results, indent=2))
                    self.assertGreater(len(results), 15)
                finally:
                    if browser is not None:
                        browser.terminate()
                        try:
                            browser.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            browser.kill()
                            browser.wait()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()

"""Optional Firefox check for dependent sampler/schedule controls."""

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
  const check = (condition, message) => {
    if (!condition) throw new Error(message);
    results.push(message);
  };
  try {
    const sampler = document.getElementById("field-sampling-sampler");
    const schedule = document.getElementById("field-sampling-noise_schedule");
    const karras = Array.from(schedule.options).find(option => option.value === "karras");
    check(karras.disabled && karras.hidden, "Euler hides Karras");
    sampler.value = "dpmpp_2m_sde";
    sampler.dispatchEvent(new Event("change", {bubbles: true}));
    check(!karras.disabled && !karras.hidden, "DPM++ enables Karras");
    check(["normal", "exponential", "beta"].every(name => Array.from(schedule.options).some(option => option.value === name && !option.disabled)),
          "DPM++ exposes registered Normal, Exponential and Beta schedules");
    schedule.value = "normal";
    sampler.dispatchEvent(new Event("input", {bubbles: true}));
    check(schedule.value === "normal", "A valid schedule stays selected");
    const data = new FormData(sampler.form);
    data.set("prompt", "A mountain lake");
    data.set("action", "preview");
    const response = await fetch(sampler.form.getAttribute("action"), {method: "POST", body: data});
    const page = new DOMParser().parseFromString(await response.text(), "text/html");
    check(response.ok && page.getElementById("request-preview"), "The preview request succeeds");
    const preview = JSON.parse(page.getElementById("request-preview").textContent);
    check(response.ok && preview.overrides.sampling.noise_schedule === "normal", "The browser submits the Normal pair");
    sampler.value = "euler_ancestral";
    sampler.dispatchEvent(new Event("change", {bubbles: true}));
    check(schedule.value === "default" && karras.disabled && karras.hidden, "Changing to Euler A resets an incompatible schedule");
  } catch (error) { results.push("FAIL: " + error.message); }
  await fetch("/browser-results", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(results)});
});
</script>
"""


@unittest.skipUnless(os.environ.get("CAROUSEL_FIREFOX"), "Set CAROUSEL_FIREFOX to run real-browser checks")
class SamplingBrowserTests(unittest.TestCase):
    def test_schedule_choices_follow_sampler_and_submit_valid_pair(self):
        browser_binary = Path(os.environ["CAROUSEL_FIREFOX"])
        native_firefox = Path("/snap/firefox/current/usr/lib/firefox/firefox")
        if browser_binary == Path("/usr/bin/firefox") and native_firefox.is_file():
            # Bypass Snap's launcher so the test owns and can stop its child.
            browser_binary = native_firefox
        # Snap Firefox has a private /tmp; use a visible workspace profile.
        with tempfile.TemporaryDirectory(prefix="sampling-browser-", dir=ROOT) as temporary:
            root = Path(temporary)
            catalog = root / "families"
            shutil.copytree(ROOT / "tests/fixtures/model_families", catalog)
            manifest = catalog / "SDXL/checkpoints/demo/model.defaults.json"
            document = json.loads(manifest.read_text())
            document["requirements"]["allowed_sampling"].extend([
                {"sampler": "dpmpp_2m_sde", "noise_schedule": "default"},
                {"sampler": "dpmpp_2m_sde", "noise_schedule": "normal"},
                {"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"},
                {"sampler": "dpmpp_2m_sde", "noise_schedule": "exponential"},
                {"sampler": "dpmpp_2m_sde", "noise_schedule": "beta"},
            ])
            manifest.write_text(json.dumps(document))
            app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": catalog})
            results, done = [], threading.Event()

            @app.after_request
            def inject_harness(response):
                if request.path == "/generate/fixture-sdxl":
                    response.set_data(response.get_data(as_text=True).replace("</body>", HARNESS + "</body>"))
                return response

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
            browser = None
            with (root / "firefox.log").open("w+") as log:
                try:
                    browser = subprocess.Popen([
                        str(browser_binary), "--headless", "--no-remote", "--profile", str(profile),
                        f"http://127.0.0.1:{server.server_port}/generate/fixture-sdxl",
                    ], stdout=log, stderr=log)
                    completed = done.wait(30)
                    log.seek(0)
                    self.assertTrue(completed, log.read()[-3000:])
                    self.assertFalse([item for item in results if item.startswith("FAIL:")], json.dumps(results, indent=2))
                    self.assertEqual(len(results), 7)
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

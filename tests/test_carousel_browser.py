"""Optional real-browser checks; set CAROUSEL_FIREFOX to a Firefox executable.

Run: PYTHONPATH=src:tests CAROUSEL_FIREFOX=/path/to/firefox python -m unittest test_carousel_browser
The test uses temporary fixtures and a loopback HTTP server, never real assets.
"""

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

from flask import render_template_string, request
from werkzeug.serving import WSGIRequestHandler, make_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.web.web import get_catalog


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Carousel memory check</title>
<link rel="stylesheet" href="{{ url_for('catalog.static', filename='catalog.css') }}">
<script src="{{ url_for('catalog.static', filename='carousel.js') }}" defer></script>
</head><body>
{% from 'controls/asset_preview.html' import asset_preview %}
<form id="cards" style="display: grid; grid-template-columns: 260px 260px; gap: 20px">
  <article class="card">{{ asset_preview(model) }}</article>
  <article class="card">{{ asset_preview(lora) }}</article>
</form>
<div style="margin-top: 3000px; width: 260px">{{ asset_preview(offscreen) }}</div>
<script>{{ harness | safe }}</script>
</body></html>"""

HARNESS = r"""
window.addEventListener("load", async () => {
  const results = [];
  const wait = (ms = 50) => new Promise((resolve) => setTimeout(resolve, ms));
  const check = (ok, message) => {
    if (!ok) throw new Error(message);
    results.push(message);
  };
  const until = async (condition) => {
    const end = performance.now() + 4000;
    while (!condition() && performance.now() < end) await wait();
    if (!condition()) throw new Error("Timed out waiting for browser media update");
  };
  const sources = (card) => [...card.querySelectorAll("img[src], source[src]")];
  const requests = () => fetch("/browser-requests").then((response) => response.json());
  try {
    let submitted = false;
    document.getElementById("cards").addEventListener("submit", (event) => {
      event.preventDefault(); submitted = true;
    });
    const [first, second, offscreen] = document.querySelectorAll("[data-asset-carousel]");
    const track = first.querySelector("[data-carousel-track]");
    const slides = [...track.children];
    const next = first.querySelector("[data-carousel-next]");
    const previous = first.querySelector("[data-carousel-previous]");
    const counter = first.querySelector("[data-carousel-counter]");
    const at = (index) => sources(first).length === 1 && sources(slides[index]).length === 1;
    const key = (value) => track.dispatchEvent(new KeyboardEvent("keydown", {
      key: value, bubbles: true, cancelable: true,
    }));
    await until(() => at(0) && sources(second).length === 1);
    await wait(200);
    check(sources(offscreen).length === 0, "Offscreen carousel has no active sources");
    const initial = await requests();
    check(initial.length === 2 && initial.every((item) => item.index === 0),
          "Only first examples of visible cards are requested; other examples are never prefetched");
    next.click(); await until(() => at(1));
    check(!slides[0].querySelector("img").hasAttribute("src"), "Previous image source is removed");
    await until(() => counter.textContent.trim() === "2 / 13");
    check(sources(second).length === 1, "Other visible card keeps its own active example");
    for (let index = 2; index < 12; index++) {
      next.click(); await until(() => at(index));
      check(sources(first).length === 1, `Only one source remains after visiting image ${index + 1}`);
    }
    check((await requests()).every((item) => item.asset !== "fixture-flux-detail"),
          "Offscreen examples still have no network requests after browsing");
    key("End"); await until(() => at(12));
    const video = slides[12].querySelector("video");
    check(video.preload === "metadata", "Only active video loads metadata");
    let pauses = 0, resets = 0;
    const pauseVideo = video.pause.bind(video), loadVideo = video.load.bind(video);
    video.pause = () => { pauses++; pauseVideo(); };
    video.load = () => { resets++; loadVideo(); };
    next.click(); await until(() => at(0));
    check(pauses === 1 && resets === 1 && !video.querySelector("source").hasAttribute("src"),
          "Leaving video pauses, removes source, and resets playback buffers");
    check(video.buffered.length === 0 && video.preload === "none", "Inactive video has no playback buffers or preloading");
    previous.click(); await until(() => at(12));
    check(counter.textContent.trim() === "13 / 13", "Previous wraps to last slide");
    key("Home"); await until(() => at(0));
    key("ArrowRight"); await until(() => at(1));
    key("ArrowLeft"); await until(() => at(0));
    track.scrollTo({left: track.clientWidth * 3, behavior: "instant"});
    await until(() => at(3));
    check(counter.textContent.trim() === "4 / 13", "Scrolling selects a slide and releases the old image");
    first.style.width = "220px";
    await wait(150);
    check(at(3), "Resizing preserves selection with a single active source");
    first.style.display = "none";
    await until(() => sources(first).length === 0);
    first.style.display = "";
    await until(() => at(3));
    check(true, "Collapsing releases media and expanding restores the selection");
    offscreen.scrollIntoView({block: "center"});
    await until(() => sources(offscreen).length === 1 && !sources(first).length && !sources(second).length);
    check(true, "Scrolling away releases both old carousels and loads only the newly visible carousel");
    window.scrollTo(0, 0);
    await until(() => at(3) && sources(second).length === 1 && !sources(offscreen).length);
    check(true, "Returning to a carousel restores its selected example");
    Object.defineProperty(document, "hidden", {configurable: true, value: true});
    document.dispatchEvent(new Event("visibilitychange"));
    check(!sources(first).length && !sources(second).length, "Hiding the tab detaches all carousel media");
    delete document.hidden;
    document.dispatchEvent(new Event("visibilitychange"));
    await until(() => at(3) && sources(second).length === 1);
    window.dispatchEvent(new PageTransitionEvent("pagehide"));
    check(!sources(first).length && !sources(second).length, "Leaving the page detaches all carousel media");
    window.dispatchEvent(new PageTransitionEvent("pageshow"));
    await until(() => at(3) && sources(second).length === 1);
    check(true, "Restoring the page restores selected media");
    check(!submitted, "Carousel controls do not submit the generation form");
  } catch (error) { results.push("FAIL: " + error.message); }
  await fetch("/browser-results", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(results)});
});
"""


class QuietHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs):
        pass


@unittest.skipUnless(os.environ.get("CAROUSEL_FIREFOX"), "Set CAROUSEL_FIREFOX to run real-browser checks")
class CarouselBrowserTests(unittest.TestCase):
    def test_only_visible_selected_media_is_loaded_and_old_sources_are_released(self):
        with tempfile.TemporaryDirectory(prefix="carousel-browser-") as temporary:
            root = Path(temporary)
            catalog = root / "families"
            shutil.copytree(ROOT / "tests/fixtures/model_families", catalog)
            pixel = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
            for folder in ("FLUX/checkpoints/demo", "FLUX/loras/style", "FLUX/loras/detail"):
                directory = catalog / folder
                for index in range(12):
                    (directory / f"photo-{index:02}.png").write_bytes(pixel)
                (directory / "video.webm").write_bytes(b"test video")
            app = create_app({"TESTING": True, "ACCESS_PASSWORD": "", "CATALOG_ROOT": catalog})
            requested, results = [], []
            done = threading.Event()

            @app.before_request
            def record_media_request():
                if request.endpoint == "catalog.preview":
                    requested.append({"asset": request.view_args["asset_id"], "index": request.view_args["media_index"]})

            @app.get("/browser-test")
            def browser_page():
                current = get_catalog()
                return render_template_string(PAGE, model=current.get_model("fixture-flux"),
                                              lora=current.get_lora("fixture-flux-style"),
                                              offscreen=current.get_lora("fixture-flux-detail"), harness=HARNESS)

            @app.get("/browser-requests")
            def browser_requests():
                return requested

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
                        f"http://127.0.0.1:{server.server_port}/browser-test",
                    ], stdout=log, stderr=log)
                    completed = done.wait(30)
                    log.flush()
                    log.seek(0)
                    self.assertTrue(completed, f"Browser check timed out: {log.read()[-3000:]}")
                    self.assertFalse([item for item in results if item.startswith("FAIL:")], json.dumps(results, indent=2))
                    self.assertGreater(len(results), 20)
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

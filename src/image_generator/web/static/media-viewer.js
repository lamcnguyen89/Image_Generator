"use strict";

// Any [data-media-gallery] can opt in with [data-media-item] links whose href
// points to original media, plus data-media-kind and data-media-type. Delegation
// supports additional galleries without initializing a viewer for every card.
(() => {
  let viewer;
  let gallery;
  let items = [];
  let current = 0;
  let opener;
  let stage, title, counter, previous, next, zoom, status;
  let pageActive = true;

  const release = () => {
    const video = stage.querySelector("video");
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.querySelectorAll("source").forEach((source) => source.removeAttribute("src"));
      video.load();
    }
    stage.querySelectorAll("img").forEach((img) => img.removeAttribute("src"));
    stage.replaceChildren();
  };

  const show = (index) => {
    current = (index + items.length) % items.length;
    release();
    stage.removeAttribute("data-actual-size");
    stage.scrollTo(0, 0);
    zoom.setAttribute("aria-pressed", "false");
    zoom.textContent = "Actual size";
    const item = items[current];
    const isVideo = item.dataset.mediaKind === "video";
    title.textContent = gallery.dataset.mediaLabel || "Example media";
    counter.textContent = `${current + 1} / ${items.length}`;
    previous.disabled = next.disabled = items.length < 2;
    zoom.hidden = isVideo;
    status.textContent = "";
    if (document.hidden || !pageActive) return;

    const media = document.createElement(isVideo ? "video" : "img");
    media.addEventListener("error", () => {
      if (stage.contains(media)) status.textContent = "This media could not be loaded. You can still browse other examples.";
    });
    if (isVideo) {
      media.controls = true;
      media.muted = true; // Muting permits autoplay under browser policies.
      media.autoplay = true;
      media.loop = true;
      media.playsInline = true;
      media.setAttribute("aria-label", item.getAttribute("aria-label") || "Example video");
      const source = document.createElement("source");
      source.src = item.href;
      source.type = item.dataset.mediaType || "";
      source.addEventListener("error", () => media.dispatchEvent(new Event("error")));
      media.append(source);
    } else {
      media.alt = item.querySelector("img")?.alt || "Example image";
      media.src = item.href;
    }
    stage.append(media);
    if (isVideo) media.play().catch(() => {
      if (stage.contains(media) && !media.error && media.paused) {
        status.textContent = "Press play to start this video.";
      }
    });
  };

  const createViewer = () => {
    viewer = document.createElement("dialog");
    viewer.className = "media-viewer";
    viewer.setAttribute("aria-labelledby", "media-viewer-title");
    viewer.innerHTML = `
      <div class="media-viewer-header">
        <h2 class="media-viewer-title" id="media-viewer-title"></h2>
        <button type="button" data-viewer-close autofocus aria-label="Exit focus mode">×</button>
      </div>
      <div class="media-viewer-body">
        <button type="button" data-viewer-previous aria-label="Previous example">←</button>
        <div class="media-viewer-stage" data-viewer-stage tabindex="0" aria-label="Example media; scroll to inspect actual-size images"></div>
        <button type="button" data-viewer-next aria-label="Next example">→</button>
      </div>
      <div class="media-viewer-footer">
        <span data-viewer-counter role="status" aria-live="polite" aria-atomic="true"></span>
        <button type="button" data-viewer-zoom aria-pressed="false">Actual size</button>
        <p class="media-viewer-status" data-viewer-status role="status"></p>
      </div>`;
    document.body.append(viewer);
    stage = viewer.querySelector("[data-viewer-stage]");
    title = viewer.querySelector(".media-viewer-title");
    counter = viewer.querySelector("[data-viewer-counter]");
    previous = viewer.querySelector("[data-viewer-previous]");
    next = viewer.querySelector("[data-viewer-next]");
    zoom = viewer.querySelector("[data-viewer-zoom]");
    status = viewer.querySelector("[data-viewer-status]");
    viewer.querySelector("[data-viewer-close]").addEventListener("click", () => viewer.close());
    previous.addEventListener("click", () => show(current - 1));
    next.addEventListener("click", () => show(current + 1));
    zoom.addEventListener("click", () => {
      const actualSize = stage.toggleAttribute("data-actual-size");
      zoom.setAttribute("aria-pressed", String(actualSize));
      zoom.textContent = actualSize ? "Fit to screen" : "Actual size";
    });
    viewer.addEventListener("keydown", (event) => {
      // Native video controls and actual-size panning retain their arrow keys.
      if (event.target.closest("video") || (stage.hasAttribute("data-actual-size") && stage.contains(event.target)) ||
          event.altKey || event.ctrlKey || event.metaKey) return;
      const targets = { ArrowLeft: current - 1, ArrowRight: current + 1, Home: 0, End: items.length - 1 };
      if (!(event.key in targets)) return;
      event.preventDefault();
      show(targets[event.key]);
    });
    viewer.addEventListener("close", () => {
      release();
      document.documentElement.classList.remove("media-viewer-open");
      if (opener?.isConnected) opener.focus({ preventScroll: true });
      items = [];
      gallery = opener = null;
    });
  };

  document.addEventListener("click", (event) => {
    const item = event.target.closest("[data-media-item]");
    const group = item?.closest("[data-media-gallery]");
    if (!group || event.defaultPrevented || event.button !== 0 ||
        event.altKey || event.ctrlKey || event.metaKey || event.shiftKey ||
        typeof HTMLDialogElement === "undefined") return;
    event.preventDefault();
    if (!viewer) createViewer();
    gallery = group;
    items = Array.from(group.querySelectorAll("[data-media-item]"));
    opener = item;
    // Avoid simultaneous playback of an inline video and its focused copy.
    group.querySelectorAll("video").forEach((video) => video.pause());
    viewer.showModal();
    document.documentElement.classList.add("media-viewer-open");
    show(items.indexOf(item));
  });

  // Keep a single active item and discard video buffers when hidden or closed.
  const refresh = () => { if (viewer?.open) show(current); };
  document.addEventListener("visibilitychange", refresh);
  window.addEventListener("pagehide", () => { pageActive = false; refresh(); });
  window.addEventListener("pageshow", () => { pageActive = true; refresh(); });
})();

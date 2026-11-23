"use strict";

(() => {
  const carousels = document.querySelectorAll("[data-asset-carousel]");
  if (!carousels.length) return;
  const refreshers = [];
  let pageActive = true;
  const visibilityCallbacks = new WeakMap();
  const visibilityObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => visibilityCallbacks.get(entry.target)(entry.isIntersecting && entry.intersectionRatio > 0));
  });
  const refreshAll = () => refreshers.forEach((refresh) => refresh());
  document.addEventListener("visibilitychange", refreshAll);
  window.addEventListener("pagehide", () => { pageActive = false; refreshAll(); });
  window.addEventListener("pageshow", () => { pageActive = true; refreshAll(); });
  carousels.forEach((carousel) => {
    const track = carousel.querySelector("[data-carousel-track]");
    const slides = Array.from(track.children);
    const counter = carousel.querySelector("[data-carousel-counter]");
    let current = 0;
    let visible = false;
    let loaded = null;
    const syncMedia = () => {
      const target = visible && pageActive && !document.hidden ? current : null;
      if (target === loaded) return;
      if (loaded !== null) {
        const previous = slides[loaded];
        previous.querySelectorAll("video").forEach((video) => {
          video.pause();
          video.preload = "none";
          video.removeAttribute("src");
          video.querySelectorAll("source").forEach((source) => source.removeAttribute("src"));
          // Reset the media element to discard its buffered playback data.
          video.load();
        });
        previous.querySelectorAll("img").forEach((img) => img.removeAttribute("src"));
      }
      loaded = target;
      if (target === null) return;
      const slide = slides[target];
      slide.querySelectorAll("[data-media-src]").forEach((media) => {
        media.setAttribute("src", media.dataset.mediaSrc);
      });
      slide.querySelectorAll("video").forEach((video) => {
        video.preload = "metadata";
        video.load();
      });
    };
    refreshers.push(syncMedia);
    visibilityCallbacks.set(track, (inViewport) => { visible = inViewport; syncMedia(); });
    visibilityObserver.observe(track);
    const update = () => {
      if (!track.clientWidth) return;
      const index = Math.max(0, Math.min(slides.length - 1, Math.round(track.scrollLeft / track.clientWidth)));
      if (index === current) return;
      current = index;
      syncMedia();
      counter.textContent = `${current + 1} / ${slides.length}`;
    };
    const go = (index) => {
      const target = (index + slides.length) % slides.length;
      track.scrollTo({ left: target * track.clientWidth, behavior: "instant" });
      update();
    };
    carousel.querySelector("[data-carousel-previous]").addEventListener("click", () => go(current - 1));
    carousel.querySelector("[data-carousel-next]").addEventListener("click", () => go(current + 1));
    track.addEventListener("scroll", update, { passive: true });
    track.addEventListener("keydown", (event) => {
      // Leave links and native video controls to handle their own keyboard input.
      if (event.target !== track || event.altKey || event.ctrlKey || event.metaKey) return;
      const targets = { ArrowLeft: current - 1, ArrowRight: current + 1, Home: 0, End: slides.length - 1 };
      if (!(event.key in targets)) return;
      event.preventDefault();
      go(targets[event.key]);
    });
    // Preserve the selected example when the card changes width, including
    // LoRA cards that are initially inside a collapsed group.
    new ResizeObserver(() => go(current)).observe(track);
    carousel.querySelector("[data-carousel-controls]").hidden = false;
  });
})();

"use strict";

(() => {
const field = (name) => document.getElementById(`field-${name.replaceAll(".", "-")}`);
const frames = field("num_frames");
const fps = field("fps");
const duration = document.getElementById("video-duration");
if (frames && fps && duration) {
  const updateDuration = () => {
    const count = Number(frames.value);
    const rate = Number(fps.value);
    duration.textContent = count > 0 && rate > 0 && Number.isFinite(count / rate)
      ? `Estimated playback: ${(count / rate).toFixed(2)} seconds (${count} frames at ${rate} FPS).`
      : "Enter a positive frame count and FPS to estimate duration.";
  };
  frames.addEventListener("input", updateDuration);
  fps.addEventListener("input", updateDuration);
  updateDuration();
}

const source = card.querySelector('select[name^="strength_source:"]');
  enabled.addEventListener("change", () => { if (enabled.checked) configured.checked = true; });
  strength.addEventListener("input", () => { source.value = "override"; });
  card.querySelector("[data-remove-lora]").addEventListener("click", () => {
    enabled.checked = false;
    configured.checked = false;
  });
  const apply = card.querySelector("[data-apply-recommendations]");
  if (apply) apply.addEventListener("click", () => {
    let applied = 0;
    const skipped = [];
    const visit = (settings, prefix = "") => Object.entries(settings).forEach(([key, value]) => {
      const name = prefix + key;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        visit(value, `${name}.`);
        return;
      }
      const control = field(name);
      if (!control || control.type === "file" || control.readOnly || control.disabled) {
        skipped.push(name);
        return;
      }
      if (control.tagName === "SELECT" && !Array.from(control.options).some((option) => option.value === String(value) && !option.disabled)) {
        skipped.push(name);
        return;
      }
      if (control.type === "checkbox") control.checked = Boolean(value);
      else control.value = Array.isArray(value) ? value.join("\n") : (value ?? "");
      control.dispatchEvent(new Event("input", { bubbles: true }));
      applied += 1;
    });
    visit(JSON.parse(apply.dataset.applyRecommendations));
    card.querySelector("[data-recommendation-status]").textContent = `${applied} recommended fields applied.${skipped.length ? ` Unavailable fields skipped: ${skipped.join(", ")}.` : ""}`;
  });
});
const frames = field("num_frames");
const fps = field("fps");
const duration = document.getElementById("video-duration");
if (frames && fps && duration) {
  const updateDuration = () => {
    const count = Number(frames.value);
    const rate = Number(fps.value);
    duration.textContent = count > 0 && rate > 0 && Number.isFinite(count / rate)
      ? `Estimated playback: ${(count / rate).toFixed(2)} seconds (${count} frames at ${rate} FPS).`
      : "Enter a positive frame count and FPS to estimate duration.";
  };
  frames.addEventListener("input", updateDuration);
  fps.addEventListener("input", updateDuration);
  updateDuration();
}

const cfg = field("true_cfg_scale");
const negative = field("negative_prompt");
const cfgStatus = document.getElementById("cfg-status");
if (cfg && negative && cfgStatus) {
  const updateCFG = () => {
    cfgStatus.textContent = Number(cfg.value) > 1 && negative.value.trim()
      ? "Negative prompting is active with these settings."
      : "Negative prompting is inactive. Set true CFG above 1 and enter a negative prompt to enable it.";
  };
  cfg.addEventListener("input", updateCFG);
  negative.addEventListener("input", updateCFG);
  updateCFG();
}

const source = field("input_image");
const image = document.getElementById("source-preview");
if (source && image) {
  let imageURL;
  source.addEventListener("change", () => {
    if (imageURL) URL.revokeObjectURL(imageURL);
    const file = source.files[0];
    image.hidden = true;
    image.removeAttribute("src");
    if (file && file.type.startsWith("image/")) {
      imageURL = URL.createObjectURL(file);
      image.src = imageURL;
      image.hidden = false;
    }
  });
  image.addEventListener("error", () => { image.hidden = true; });
  window.addEventListener("pagehide", () => { if (imageURL) URL.revokeObjectURL(imageURL); });
}


})();

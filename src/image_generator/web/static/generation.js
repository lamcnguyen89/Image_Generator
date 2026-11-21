"use strict";

// Optional hints only. Python validates the submitted values independently.
const field = (name) => document.getElementById(`field-${name.replaceAll(".", "-")}`);
const generationMode = document.getElementById("generation-mode");
const changeMode = document.getElementById("change-generation-mode");
if (generationMode && changeMode) {
  generationMode.addEventListener("change", () => {
    changeMode.form.requestSubmit(changeMode);
  });
}
document.querySelectorAll("[data-lora-id]").forEach((card) => {
  const enabled = card.querySelector('input[name="lora"]');
  const configured = card.querySelector('input[name="configured_lora"]');
  const strength = card.querySelector('input[name^="strength:"]');
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

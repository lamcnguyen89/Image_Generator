"use strict";

(() => {
  const panel = document.querySelector("[data-settings-panel]");
  if (!panel) return;
  const status = panel.querySelector("[data-setting-status]");
  const rows = [...panel.querySelectorAll("[data-setting-row]")];
  let active = null;
  let saving = false;
  const updateButtons = () => rows.forEach((row) => {
    row.querySelector("[data-setting-edit]").disabled = saving || (active !== null && active !== row);
  });
  rows.forEach((row) => {
    const display = row.querySelector("[data-setting-display]");
    const form = row.querySelector("[data-setting-editor]");
    const input = form.querySelector("[data-setting-input]");
    const error = form.querySelector("[data-setting-error]");
    const edit = row.querySelector("[data-setting-edit]");
    let savedValue = input.value;
    const close = () => {
      input.value = savedValue;
      error.textContent = "";
      input.removeAttribute("aria-invalid");
      form.hidden = true;
      display.hidden = false;
      edit.hidden = false;
      edit.setAttribute("aria-expanded", "false");
      active = null;
      updateButtons();
      edit.focus();
    };
    edit.hidden = false;
    edit.addEventListener("click", () => {
      active = row;
      status.textContent = "";
      form.hidden = false;
      display.hidden = true;
      edit.hidden = true;
      edit.setAttribute("aria-expanded", "true");
      updateButtons();
      input.focus();
    });
    form.querySelector("[data-setting-cancel]").addEventListener("click", close);
    form.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !saving) { event.preventDefault(); close(); }
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (saving) return;
      saving = true;
      error.textContent = "";
      input.removeAttribute("aria-invalid");
      updateButtons();
      const controls = [...form.querySelectorAll("button, input, select, textarea")];
      controls.forEach((control) => { control.disabled = true; });
      status.textContent = "Saving…";
      try {
        const response = await fetch(panel.dataset.saveUrl, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csrf_token: panel.dataset.csrfToken,
            settings_revision: panel.dataset.settingsRevision,
            field: row.dataset.field, value_text: input.value }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "The setting could not be saved.");
        panel.dataset.settingsRevision = data.settings_revision;
        savedValue = data.editor_value;
        display.textContent = data.display_value;
        document.querySelector("[data-user-overrides]").textContent = data.overrides_json;
        close();
        status.textContent = `${row.dataset.field} saved.`;
        if (data.removed) {
          const index = rows.indexOf(row);
          rows.splice(index, 1);
          row.remove();
        }
        if (row.dataset.field === "loras") window.location.reload();
      } catch (failure) {
        status.textContent = "";
        error.textContent = failure instanceof SyntaxError ? "The server response could not be read. Reload the page and sign in again if needed." : failure.message;
        input.setAttribute("aria-invalid", "true");
      } finally {
        saving = false;
        controls.forEach((control) => { control.disabled = false; });
        updateButtons();
        (active === row ? input : edit).focus();
      }
    });
  });
})();

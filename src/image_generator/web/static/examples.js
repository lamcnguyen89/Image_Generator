"use strict";

(() => {
  const form = document.getElementById("generation-form");
  if (!form) return;
  const revision = form.querySelector('input[name="settings_revision"]');
  const csrf = form.querySelector('input[name="csrf_token"]');
  let saving = false;
  const editors = Array.from(form.querySelectorAll("[data-example-editor]"), (element) => {
    const select = element.querySelector("[data-example-select]");
    const text = element.querySelector("textarea[data-example-text]");
    const use = element.querySelector("[data-example-use]");
    const add = element.querySelector("[data-example-add]");
    const remove = element.querySelector("[data-example-remove]");
    const status = element.querySelector("[data-example-status]");
    const target = document.getElementById(`field-${element.dataset.promptTarget}`);
    const update = () => {
      select.disabled = saving;
      text.disabled = saving;
      use.disabled = saving || !text.value.trim() || !target || target.disabled || target.readOnly;
      add.disabled = saving || !text.value.trim();
      remove.disabled = saving || select.value === "";
    };
    select.addEventListener("change", () => {
      text.value = select.selectedOptions[0]?.dataset.exampleText || "";
      status.textContent = "";
      update();
    });
    text.addEventListener("input", update);
    use.addEventListener("click", () => {
      if (use.disabled) return;
      target.value = text.value;
      target.dispatchEvent(new Event("input", { bubbles: true }));
      target.dispatchEvent(new Event("change", { bubbles: true }));
      target.focus();
      status.dataset.error = "false";
      status.textContent = element.dataset.promptTarget === "prompt" ? "Copied to prompt." : "Copied to negative prompt.";
    });
    const save = async (action) => {
      if (saving) return;
      const payload = {
        csrf_token: csrf.value, settings_revision: revision.value,
        field: element.dataset.exampleField, action,
        text: text.value, index: select.value === "" ? null : Number(select.value),
      };
      saving = true;
      editors.forEach((editor) => editor.update());
      const submitButtons = Array.from(form.querySelectorAll('button[type="submit"]'));
      const previouslyDisabled = submitButtons.map((button) => button.disabled);
      submitButtons.forEach((button) => { button.disabled = true; });
      status.dataset.error = "false";
      status.textContent = "Saving examples…";
      try {
        const response = await fetch(element.dataset.examplesUrl, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Examples could not be saved.");
        revision.value = data.settings_revision;
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = data.examples.length ? "Choose an example" : "No saved examples";
        select.replaceChildren(placeholder);
        data.examples.forEach((example, index) => {
          const option = document.createElement("option");
          option.value = String(index);
          option.textContent = example;
          option.dataset.exampleText = example;
          select.append(option);
        });
        select.value = action === "add" ? String(data.examples.length - 1) : "";
        text.value = action === "add" ? data.examples.at(-1) : "";
        element.querySelector("[data-example-count]").textContent = `(${data.examples.length})`;
        status.textContent = action === "add" ? "Example added." : "Example removed.";
      } catch (error) {
        status.dataset.error = "true";
        status.textContent = error instanceof SyntaxError ? "The server response could not be read. Reload the page and sign in again if needed." : error.message;
      } finally {
        saving = false;
        editors.forEach((editor) => editor.update());
        submitButtons.forEach((button, index) => { button.disabled = previouslyDisabled[index]; });
      }
    };
    add.addEventListener("click", () => save("add"));
    remove.addEventListener("click", () => save("remove"));
    update();
    return { update };
  });
})();

"use strict";

(() => {
const field = (name) => document.getElementById(`field-${name.replaceAll(".", "-")}`);
const negative = field("negative_prompt");
const krea2Guidance = field("guidance_scale");
const krea2Status = document.getElementById("krea2-cfg-status");
if (krea2Guidance && negative && krea2Status) {
  const updateKrea2CFG = () => {
    krea2Status.textContent = Number(krea2Guidance.value) > 0 && negative.value.trim()
      ? "Negative prompting is active with these settings."
      : "Negative prompting is inactive. Set guidance above 0 and enter a negative prompt to enable it.";
  };
  krea2Guidance.addEventListener("input", updateKrea2CFG);
  negative.addEventListener("input", updateKrea2CFG);
  updateKrea2CFG();
}

})();

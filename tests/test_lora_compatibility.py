"""Declarations used consistently by WAN UI, preferences, and generation."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog.compatibility import lora_model_variants
from image_generator.catalog.validation import validate_defaults, validate_settings


class LoRACompatibilityTests(unittest.TestCase):
    def test_legacy_single_variant_and_explicit_list(self):
        self.assertEqual(lora_model_variants({"model_variant": "i2v"}), ("i2v",))
        self.assertEqual(lora_model_variants({"model_variant": None}), ())
        self.assertEqual(lora_model_variants({"model_variant": "i2v", "model_variants": ["i2v", "t2v"]}),
                         ("i2v", "t2v"))
        self.assertEqual(lora_model_variants({"model_variant": "i2v", "model_variants": ["t2v"]}), ("t2v",))

    def test_variant_lists_reject_invalid_and_duplicate_entries(self):
        for variants in (None, "i2v", [], [""], ["i2v", "i2v"], [{"variant": "i2v"}]):
            issues = validate_defaults({"compatibility": {"model_variants": variants}}, Path("lora.json"), "lora")
            self.assertIn("compatibility.model_variants", [issue.field for issue in issues])

    def test_expert_link_override_accepts_only_high_or_low_noise(self):
        for target in ("high_noise", "low_noise", "both", "", None, ["high_noise"]):
            settings = {"loras": [{"id": "adapter", "enabled": True, "expert_target": target}]}
            issues = validate_settings(settings, Path("model.user.json"))
            self.assertEqual(not issues, target in ("high_noise", "low_noise"))


if __name__ == "__main__":
    unittest.main()

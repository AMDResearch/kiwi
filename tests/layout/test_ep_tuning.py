# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Host-only checks for EP runtime tuning configuration lookup."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "apps/ep/benchmarks/python/kiwi_ep_tuning.py"
SPEC = importlib.util.spec_from_file_location("kiwi_ep_tuning", MODULE_PATH)
TUNING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TUNING)


class EpTuningTests(unittest.TestCase):
    def test_gfx950_pollara_rules_are_monotonic_and_select_hybrid(self):
        nic_family = TUNING.nic_family_for("ionic_0")
        self.assertEqual(nic_family, "pollara")
        rules = TUNING.load_rules("gfx950", nic_family)
        self.assertTrue(rules)
        for world_size in (8, 16, 32):
            previous = 0
            for num_tokens in (1, 2, 4, 8, 16, 32, 64):
                defaults = TUNING.lookup_runtime_defaults(
                    rules, world_size, num_tokens
                )
                self.assertIsNotNone(defaults)
                self.assertGreaterEqual(defaults["num_blocks"], previous)
                self.assertEqual(defaults["rcomp_backend"], "hybrid")
                previous = defaults["num_blocks"]

    def test_lookup_uses_token_ceiling_and_clamps_above_table(self):
        rules = TUNING.load_rules("gfx950", "pollara")
        self.assertEqual(
            TUNING.lookup_runtime_defaults(rules, 8, 3)["num_blocks"], 16
        )
        self.assertEqual(
            TUNING.lookup_runtime_defaults(rules, 8, 4096)["num_blocks"], 64
        )

    def test_unknown_hardware_or_world_has_no_tuned_result(self):
        self.assertEqual(TUNING.load_rules("gfx942", "pollara"), [])
        rules = TUNING.load_rules("gfx950", "pollara")
        self.assertIsNone(TUNING.lookup_runtime_defaults(rules, 64, 8))

    def test_connectx7_is_identified_by_pci_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            device_dir = root / "mlx5_0/device"
            device_dir.mkdir(parents=True)
            (device_dir / "vendor").write_text("0x15b3\n")
            (device_dir / "device").write_text("0x1021\n")
            self.assertEqual(TUNING.nic_family_for("mlx5_0", root), "cx7")
            (device_dir / "device").write_text("0x101d\n")
            self.assertEqual(TUNING.nic_family_for("mlx5_0", root), "mlx5")

    def test_gfx942_cx7_rules_are_available(self):
        rules = TUNING.load_rules("gfx942", "cx7")
        self.assertTrue(rules)
        self.assertEqual(
            TUNING.lookup_runtime_defaults(rules, 64, 64),
            {"num_blocks": 64, "rcomp_backend": "hybrid"},
        )


if __name__ == "__main__":
    unittest.main()

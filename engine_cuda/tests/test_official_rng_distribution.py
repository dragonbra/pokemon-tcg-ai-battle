from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "engine_cuda" / "tools" / "run_official_rng_distribution.py"
SPEC = importlib.util.spec_from_file_location("run_official_rng_distribution", PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class OfficialRngDistributionTest(unittest.TestCase):
    def test_analyzer_rejects_shift_and_accepts_large_equivalent_counts(self):
        rows = []
        for batch in (1, 8, 256, 512):
            for backend, shift in (("official_cpu", 0), ("cuda", 10)):
                rows.append({"backend": backend, "logical_batch_size": batch,
                             "samples": 1_000_000, "coin_heads": 500_000 + shift,
                             "opening_inclusion": 116_667, "prize_inclusion": 100_000,
                             "card_position_counts": [16_667] * 40 + [16_666] * 20,
                             "random_target_counts": [125_000] * 8,
                             "adjacent_fingerprint_duplicates": 0, "coin_lag1": 0.0})
        self.assertEqual(module.analyze(rows)["status"], "PASS")
        rows[-1]["coin_heads"] = 550_000
        self.assertEqual(module.analyze(rows)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()

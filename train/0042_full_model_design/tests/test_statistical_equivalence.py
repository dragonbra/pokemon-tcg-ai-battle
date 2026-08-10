from __future__ import annotations

import importlib
import random
import unittest


stats = importlib.import_module("train.0042_full_model_design.semantic_parity.statistical_equivalence")


class StatisticalEquivalenceTest(unittest.TestCase):
    def test_binary_interval_requires_containment_not_p_value(self):
        passed = stats.binary_difference_interval(50_000, 100_000, 50_020, 100_000, margin=0.01)
        self.assertTrue(passed.passed)
        failed = stats.binary_difference_interval(51_200, 100_000, 50_000, 100_000, margin=0.01)
        self.assertFalse(failed.passed)

    def test_total_variation_has_conservative_upper_bound(self):
        same = stats.multinomial_total_variation({0: 50_000, 1: 50_000}, {0: 50_010, 1: 49_990}, margin=0.02)
        self.assertTrue(same.passed)
        shifted = stats.multinomial_total_variation({0: 60_000, 1: 40_000}, {0: 50_000, 1: 50_000}, margin=0.02)
        self.assertFalse(shifted.passed)

    def test_lag_correlation_detects_duplicated_stream(self):
        rng = random.Random(7)
        independent = [rng.randrange(2) for _ in range(100_000)]
        self.assertTrue(stats.lag1_binary_correlation(independent).passed)
        duplicated = [value for value in independent[:50_000] for _ in (0, 1)]
        self.assertFalse(stats.lag1_binary_correlation(duplicated).passed)


if __name__ == "__main__":
    unittest.main()

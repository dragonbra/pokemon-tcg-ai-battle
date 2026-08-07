from __future__ import annotations

import importlib
import unittest

import torch


BENCHMARK = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.benchmark_prototype_cache"
)


class PrototypeCacheBenchmarkProtocolTests(unittest.TestCase):
    def test_paired_orders_alternate_leading_arm(self) -> None:
        self.assertEqual(
            BENCHMARK.paired_orders(5),
            [
                ("uncached", "cached"),
                ("cached", "uncached"),
                ("uncached", "cached"),
                ("cached", "uncached"),
                ("uncached", "cached"),
            ],
        )

    def test_timing_summary_reports_median_mad_and_p95(self) -> None:
        self.assertEqual(
            BENCHMARK.timing_summary([1.0, 2.0, 10.0]),
            {"median_ms": 2.0, "mad_ms": 1.0, "p95_ms": 10.0},
        )

    def test_model_dtype_alignment_only_casts_floating_inputs(self) -> None:
        values = {
            "numeric": torch.ones(2, dtype=torch.float32),
            "categorical": torch.ones(2, dtype=torch.long),
            "mask": torch.ones(2, dtype=torch.bool),
        }
        aligned = BENCHMARK.align_model_dtype(values, torch.float16)
        self.assertEqual(aligned["numeric"].dtype, torch.float16)
        self.assertEqual(aligned["categorical"].dtype, torch.long)
        self.assertEqual(aligned["mask"].dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()

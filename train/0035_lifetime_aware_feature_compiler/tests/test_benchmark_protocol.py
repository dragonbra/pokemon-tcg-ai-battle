from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest


BENCHMARK = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.benchmark_incremental_features"
)


class BenchmarkProtocolTests(unittest.TestCase):
    def test_timing_summary_reports_within_repetition_mad(self) -> None:
        summary = BENCHMARK._timing_summary([1_000_000, 2_000_000, 10_000_000])
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["mad"], 1.0)
        self.assertEqual(summary["p95"], 10.0)
        self.assertEqual(summary["total"], 13.0)

    def test_compare_repetitions_are_paired_and_alternate_leading_arm(self) -> None:
        payload = BENCHMARK.run_benchmark(
            mode="compare",
            decisions=3,
            warmup_decisions=0,
            repetitions=3,
        )

        self.assertEqual(payload["repetition_count"], 3)
        self.assertEqual(
            [item["order"] for item in payload["paired_repetitions"]],
            [
                ["full", "incremental"],
                ["incremental", "full"],
                ["full", "incremental"],
            ],
        )
        for arm in ("full", "incremental"):
            result = payload["results"][arm]
            self.assertEqual(len(result["repetitions"]), 3)
            self.assertEqual(result["compiled_decisions"], 3)
            self.assertTrue(math.isfinite(result["decisions_per_second"]))
            for stage in ("knowledge", "compile", "collate", "total"):
                self.assertIn("mad", result["milliseconds_per_decision"][stage])
                across = result["across_repetitions"][
                    "milliseconds_per_decision"
                ][stage]
                self.assertIn("mad", across["median"])

    def test_incremental_arm_exposes_available_compiler_counters(self) -> None:
        payload = BENCHMARK.run_benchmark(
            mode="incremental",
            decisions=4,
            warmup_decisions=0,
        )
        result = payload["results"]["incremental"]
        diagnostics = result["compiler_diagnostics_per_repetition"][0]
        self.assertGreaterEqual(diagnostics["sessions"], 1)
        self.assertTrue(diagnostics["stats"])
        if "decisions" in diagnostics["stats"]:
            self.assertEqual(diagnostics["stats"]["decisions"], 4)

    def test_persistent_compare_is_paired_and_checks_tensor_parity(self) -> None:
        payload = BENCHMARK.run_benchmark(
            mode="compare_persistent",
            decisions=3,
            warmup_decisions=0,
            repetitions=2,
        )
        self.assertEqual(
            [item["order"] for item in payload["paired_repetitions"]],
            [["full", "persistent"], ["persistent", "full"]],
        )
        self.assertEqual(payload["incremental_parity_decisions"], 35)
        diagnostics = payload["results"]["persistent"][
            "compiler_diagnostics_per_repetition"
        ][0]
        self.assertEqual(diagnostics["tensor_bank"]["collates"], 3)

    def test_optional_stats_and_work_counter_surfaces_are_both_collected(self) -> None:
        class Counters:
            def __init__(self, values):
                self.values = values

            def snapshot(self):
                return self.values

        class Compiler:
            stats = Counters({"decisions": 7, "derived_rate": 0.5})
            work_counters = {"event_payload_encodes": 3}

        self.assertEqual(
            BENCHMARK._compiler_counters(Compiler()),
            {
                "stats": {"decisions": 7, "derived_rate": 0.5},
                "work_counters": {"event_payload_encodes": 3},
            },
        )

    def test_old_cli_arguments_remain_valid_and_repetitions_are_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark.json"
            exit_code = BENCHMARK.main(
                [
                    "--mode",
                    "full",
                    "--decisions",
                    "1",
                    "--warmup-decisions",
                    "0",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["repetition_count"], 1)
            self.assertEqual(payload["compiled_decisions"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import math
from pathlib import Path
import unittest


BENCHMARK = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining."
    "benchmark_incremental_features"
)
FIELDS = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.contracts.fields"
)


class IncrementalFeatureBenchmarkTests(unittest.TestCase):
    def test_fixture_is_small_chronological_and_representative(self) -> None:
        path = BENCHMARK.FIXTURE_PATH
        self.assertTrue(path.is_file())
        self.assertLess(path.stat().st_size, 256 * 1024)
        trajectories = BENCHMARK.load_parity_trajectories(path)
        self.assertEqual(len(trajectories), 1)
        trajectory = trajectories[0]
        self.assertEqual((trajectory.provenance["episode_id"], trajectory.actor), (89418098, 1))
        self.assertEqual(len(trajectory.deck), 60)
        self.assertEqual(len(trajectory.decisions), 35)
        self.assertEqual(
            [item.event_cursor["actor_decision_index"] for item in trajectory.decisions],
            list(range(len(trajectory.decisions))),
        )
        self.assertTrue(
            BENCHMARK.REQUIRED_COVERAGE.issubset(
                BENCHMARK.trajectory_coverage(trajectory)
            )
        )

    def test_full_rebuild_sequence_preserves_batch_contract(self) -> None:
        trajectory = BENCHMARK.load_parity_trajectories()[0]
        batches = BENCHMARK.full_rebuild_sequence(trajectory)
        self.assertEqual(len(batches), len(trajectory.decisions))
        for batch in batches:
            self.assertEqual(set(batch), FIELDS.EXPECTED_BATCH_KEYS)
            self.assertTrue(all(value.shape[0] == 1 for value in batch.values()))

    def test_full_benchmark_emits_finite_stage_metrics(self) -> None:
        payload = BENCHMARK.run_benchmark(
            mode="full",
            decisions=7,
            warmup_decisions=2,
        )
        self.assertEqual(payload["schema_version"], BENCHMARK.BENCHMARK_SCHEMA_VERSION)
        self.assertEqual(payload["actor_schema_version"], FIELDS.SCHEMA_VERSION)
        self.assertEqual(payload["compiled_decisions"], 7)
        self.assertEqual(payload["results"]["full"]["compiled_decisions"], 7)
        self.assertEqual(payload["fixture"]["trajectory_count"], 1)
        self.assertEqual(payload["fixture"]["decisions_per_cycle"], 35)
        self.assertGreater(payload["results"]["full"]["decisions_per_second"], 0.0)
        stages = payload["results"]["full"]["milliseconds_per_decision"]
        self.assertEqual(set(stages), {"knowledge", "compile", "collate", "total"})
        for stage in stages.values():
            for value in stage.values():
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)

    def test_default_fixture_path_is_repository_relative_in_report(self) -> None:
        payload = BENCHMARK.run_benchmark(
            mode="full",
            decisions=1,
            warmup_decisions=0,
        )
        reported = Path(payload["fixture"]["path"])
        self.assertFalse(reported.is_absolute())
        self.assertEqual(reported.name, "incremental_feature_trajectory.json.gz")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import importlib
from pathlib import Path
import tempfile
import unittest

from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.runner.batch import _game_jobs
from evaluation.traces.store import TraceStore
series = importlib.import_module(
    "train.0037_dragapult_value_initialized_rl.evaluate_seeded_checkpoint_series"
)
COMPARISON_UPDATES = series.COMPARISON_UPDATES
comparison_config = series.comparison_config
schedule_commitment = series.schedule_commitment
select_comparison_arms = series.select_comparison_arms
wilson_interval = series.wilson_interval


class SeededCheckpointSeriesTest(unittest.TestCase):
    def test_current_zero_shot_schema_is_exportable(self) -> None:
        exporter = importlib.import_module(
            "train.0037_dragapult_value_initialized_rl.export_full_semantic_candidate"
        )
        payload = __import__("torch").load(
            series.SOURCE_CANDIDATE / "strategy/model.bin",
            map_location="cpu",
            weights_only=True,
        )
        self.assertIn(payload["schema_version"], exporter.SOURCE_SCHEMAS)

    def test_selects_candidate_screening_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.jsonl"
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            rows = []
            for update, rate in ((0, .57), (10, .63), (20, .48), (60, .56),
                                 (80, .625), (100, .613), (120, .566)):
                row = {"trainer/update": update, "eval/checkpoint_update": update,
                       "eval/win_rate": rate}
                rows.append(row)
            rows.append({"trainer/update": 123, "rollout/source_policy_update": 122,
                         "rollout/win_rate": .4922})
            metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            for update in COMPARISON_UPDATES:
                (checkpoint / f"update-{update:06d}.pt").write_bytes(str(update).encode())

            arms = select_comparison_arms(metrics, checkpoint)

            self.assertEqual(tuple(arm.update for arm in arms), COMPARISON_UPDATES)
            self.assertEqual(arms[3].update, 80)
            self.assertEqual(arms[-1].prior_evidence, "sampled_rollout_context")
            self.assertEqual(arms[-1].source_policy_update, 122)

    def test_schedule_is_2048_independent_seed_games(self) -> None:
        catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
        config = comparison_config(catalog, output_root=Path("/tmp/unused"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = TraceStore(root / "traces", root / "reports")
            try:
                jobs = _game_jobs(config, "fixed-run", store)
                commitment = schedule_commitment(jobs)
            finally:
                store.cleanup()

        self.assertEqual(len(jobs), 2048)
        self.assertEqual(sum(config.games_by_opponent or ()), 2048)
        self.assertEqual(commitment["engine_seed_pairs"], 2048)
        self.assertEqual(commitment["candidate_first"], 1024)
        self.assertEqual(commitment["candidate_second"], 1024)
        self.assertEqual(len({request.seed for request, _ in jobs}), 2048)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(1280, 2048)
        self.assertLess(low, 1280 / 2048)
        self.assertGreater(high, 1280 / 2048)


if __name__ == "__main__":
    unittest.main()

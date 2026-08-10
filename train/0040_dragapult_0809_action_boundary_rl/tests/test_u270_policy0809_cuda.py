from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest


exporter = importlib.import_module(
    "train.0040_dragapult_0809_action_boundary_rl.export_full_semantic_candidate"
)
driver = importlib.import_module(
    "train.0040_dragapult_0809_action_boundary_rl.evaluate_u270_policy0809_cuda"
)


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / (
    "rl_runs/0040_dragapult_0809_action_boundary_rl/versions/"
    "V2_snapshot_loader_fix_long_run/checkpoint/update-000270.pt"
)
SOURCE = ROOT / (
    "archive/submission/0031_zero_shot_0809_007_dragapult_ex_"
    "fp16_storage_fp32_runtime"
)
DECK_002 = ROOT / (
    "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks/"
    "002_alakazam_dudunsparce/deck.csv"
)


class U270Policy0809CudaTest(unittest.TestCase):
    def test_schedule_is_eight_policy_bound_256_game_units(self) -> None:
        catalog, candidates = driver._catalog()
        focal = next(
            item for item in candidates
            if item.package_manifest["frozen_deck_number"] == "007"
        )
        schedule = driver.build_schedule(
            focal_deck_id=focal.name, entries=catalog.pool.schedule
        )
        jobs = schedule["jobs"]
        self.assertEqual(len(jobs), 2048)
        self.assertEqual({row["replica"] for row in jobs}, set(range(8)))
        self.assertEqual(
            {replica: sum(row["replica"] == replica for row in jobs) for replica in range(8)},
            {replica: 256 for replica in range(8)},
        )
        self.assertEqual(len({row["engine_seed"] for row in jobs}), 2048)
        self.assertEqual(len({row["search_seed"] for row in jobs}), 2048)
        self.assertTrue(all(row["focal_policy_id"] == "0040-U270" for row in jobs))
        self.assertTrue(all(row["opponent_policy_id"] == "Policy-0809" for row in jobs))
        self.assertTrue(all(type(row["focal_won_toss"]) is bool for row in jobs))

    def test_rollout_jobs_are_grouped_by_exact_opponent_deck(self) -> None:
        catalog, candidates = driver._catalog()
        focal = next(
            item for item in candidates
            if item.package_manifest["frozen_deck_number"] == "007"
        )
        schedule = driver.build_schedule(
            focal_deck_id=focal.name, entries=catalog.pool.schedule
        )
        jobs = driver._rollout_jobs(focal, schedule, candidates)
        groups = driver._group_jobs_by_opponent_deck(jobs)

        self.assertEqual(len(groups), 55)
        self.assertEqual(sum(len(group) for group in groups), 2048)
        self.assertTrue(
            all(len({job.opponent_deck for job in group}) == 1 for group in groups)
        )
        self.assertTrue(
            all(len({job.opponent_id for job in group}) == 1 for group in groups)
        )
        self.assertEqual(
            {job.game_id for group in groups for job in group},
            {job.game_id for job in jobs},
        )

    def test_export_accepts_explicit_non_007_exact_deck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate-002"
            manifest = exporter.export_candidate(
                source=SOURCE,
                checkpoint=CHECKPOINT,
                output=output,
                require_frozen_selection=False,
                deck_path=DECK_002,
                deck_id="alakazam_dudunsparce_3f4515092dc5",
                deck_display_name="002 · Alakazam / Dudunsparce",
                expected_deck_sha256=(
                    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
                ),
            )
            self.assertEqual(manifest["checkpoint_update"], 270)
            self.assertEqual(manifest["deck_id"], "alakazam_dudunsparce_3f4515092dc5")
            self.assertEqual(manifest["deck_display_name"], "002 · Alakazam / Dudunsparce")
            self.assertEqual(len((output / "deck.csv").read_text().splitlines()), 60)
            self.assertNotEqual(manifest["deck_sha256"], exporter.FOCAL_DECK_SHA256)


if __name__ == "__main__":
    unittest.main()

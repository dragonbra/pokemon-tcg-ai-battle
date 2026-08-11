from __future__ import annotations

import importlib
from collections import Counter
import unittest

scaling = importlib.import_module("train.0042_full_model_design.training.scaling")
metrics = importlib.import_module("train.0042_full_model_design.training.metric_frequency")
panel = importlib.import_module("train.0042_full_model_design.evaluation.frozen_panel")
frozen_jobs = importlib.import_module(
    "train.0042_full_model_design.evaluation.frozen_jobs"
)
runner = importlib.import_module(
    "train.0042_full_model_design.training.run_full_semantic"
)
league = importlib.import_module("train.0042_full_model_design.league")


class ScalingAndPanelTest(unittest.TestCase):
    def test_panel_is_2048_unique_and_disjoint(self):
        rows = panel.build_panel([("deck_a", "a"), ("deck_b", "b")])
        self.assertEqual(len(rows), 2048)
        self.assertEqual(len({row.seed for row in rows}), 2048)
        shards = [{row.seed for row in rows if row.shard_id == i} for i in range(8)]
        self.assertTrue(all(len(item) == 256 for item in shards))
        self.assertEqual(len(set.union(*shards)), 2048)
        self.assertTrue(all(type(row.focal_won_toss) is bool for row in rows))

    def test_formal_frozen_jobs_match_canonical_007_contract(self):
        jobs, schedule_sha = frozen_jobs.build_frozen_jobs(
            focal_deck_id=runner.FOCAL_DECK_ID,
            focal_deck=runner.focal_deck(),
            runtime_root=runner.runtime_root(),
            source_policy_update=0,
            focal_deployment_identity=(
                frozen_jobs.EXPECTED_007_U0_DEPLOYMENT_SHA256
            ),
            opponent_effective_policy_sha256=(
                "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
            ),
        )
        expected = {
            item.deck_id: item.games * 8 for item in league.load_frozen_catalog()
        }

        self.assertEqual(len(jobs), 2048)
        self.assertEqual(schedule_sha, frozen_jobs.EXPECTED_007_SCHEDULE_SHA256)
        self.assertEqual(Counter(job.opponent_id for job in jobs), expected)
        self.assertTrue(all(type(job.focal_won_toss) is bool for job in jobs))
        self.assertEqual(sum(bool(job.focal_won_toss) for job in jobs), 1044)
        self.assertEqual(len({job.seed for job in jobs}), 2048)
        self.assertEqual({job.full_round_draw_limit for job in jobs}, {50})

    def test_scaling_has_no_512_constraint(self):
        config = scaling.RolloutScalingConfig(rollout_games_per_update=1537)
        config.validate()
        self.assertEqual(config.metadata(policy_transitions=9911, optimizer_steps=32)
                         ["effective_minibatch_size"], 256)

    def test_accelerated_backend_fails_closed(self):
        with self.assertRaises(RuntimeError):
            scaling.assert_backend_parity({"observation": True})

    def test_sparse_frequency(self):
        hits = [i for i in range(31) if metrics.is_sparse_diagnostic_update(i)]
        self.assertEqual(hits, [0, 5, 10, 20, 30])

    def test_paired_results_require_same_seeds(self):
        with self.assertRaises(ValueError):
            panel.paired_summary({1: -1}, {2: 1})


if __name__ == "__main__":
    unittest.main()

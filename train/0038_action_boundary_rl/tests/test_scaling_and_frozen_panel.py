from __future__ import annotations

import importlib
from collections import Counter
import unittest

scaling = importlib.import_module("train.0038_action_boundary_rl.training.scaling")
metrics = importlib.import_module("train.0038_action_boundary_rl.training.metric_frequency")
panel = importlib.import_module("train.0038_action_boundary_rl.evaluation.frozen_panel")
frozen_jobs = importlib.import_module(
    "train.0038_action_boundary_rl.evaluation.frozen_jobs"
)
runner = importlib.import_module(
    "train.0038_action_boundary_rl.training.run_full_semantic"
)
league = importlib.import_module("train.0038_action_boundary_rl.league")


class ScalingAndPanelTest(unittest.TestCase):
    def test_panel_is_2048_unique_and_disjoint(self):
        rows = panel.build_panel([("deck_a", "a"), ("deck_b", "b")])
        self.assertEqual(len(rows), 2048)
        self.assertEqual(len({row.seed for row in rows}), 2048)
        shards = [{row.seed for row in rows if row.shard_id == i} for i in range(8)]
        self.assertTrue(all(len(item) == 256 for item in shards))
        self.assertEqual(len(set.union(*shards)), 2048)
        self.assertEqual(sum(row.focal_first for row in rows), 1024)

    def test_formal_frozen_jobs_match_canonical_007_contract(self):
        jobs, schedule_sha = frozen_jobs.build_frozen_jobs(
            focal_deck_id=runner.FOCAL_DECK_ID,
            focal_deck=runner.focal_deck(),
            runtime_root=runner.runtime_root(),
            source_policy_update=0,
        )
        expected = {
            item.deck_id: item.games * 8 for item in league.load_frozen_catalog()
        }

        self.assertEqual(len(jobs), 2048)
        self.assertEqual(schedule_sha, "98b58bced460c1a2e622ae4b39bf506294fcb0230bb42e4117aaa6efc73c9ce9")
        self.assertEqual(Counter(job.opponent_id for job in jobs), expected)
        self.assertEqual(sum(job.focal_first for job in jobs), 1024)
        self.assertEqual(len({job.seed for job in jobs}), 2048)

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

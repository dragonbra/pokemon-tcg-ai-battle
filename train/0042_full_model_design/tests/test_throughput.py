from __future__ import annotations

import importlib
import unittest

throughput = importlib.import_module("train.0042_full_model_design.training.throughput")


class ThroughputTest(unittest.TestCase):
    def test_one_scalar_aggregate_per_stage(self):
        item = throughput.UpdateThroughput(games=10, official_selects=100,
                                           strategic_decisions=20, policy_transitions=20,
                                           ppo_samples=40)
        item.add_stage("environment_engine", 2.0)
        item.add_stage("model_inference", 1.0)
        item.add_stage("ppo_forward_backward", 2.0)
        metrics = item.metrics()
        self.assertAlmostEqual(metrics["throughput/games_per_sec"], 10 / 3)
        self.assertEqual(metrics["throughput/ppo_samples_per_sec"], 20)
        self.assertEqual(len([key for key in metrics if key.startswith("throughput/seconds/")]), 8)


if __name__ == "__main__":
    unittest.main()

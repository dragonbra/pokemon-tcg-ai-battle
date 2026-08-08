from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch


batch = importlib.import_module("train.0037_dragapult_value_initialized_rl.training.batch")


class RolloutBatchTest(unittest.TestCase):
    def test_training_schedules_are_aggregated_without_overwrite(self) -> None:
        run = importlib.import_module(
            "train.0037_dragapult_value_initialized_rl.training.run_full_semantic"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_schedules.json"
            for update in (0, 1):
                payload = {
                    "schema": "0037_seeded512_rollout_schedule_v1",
                    "jobs": [{"source_policy_update": update}],
                    "schedule_sha256": str(update),
                }
                run._record_training_schedule(path, payload)
            aggregate = json.loads(path.read_text())
            self.assertEqual(
                [row["source_policy_update"] for row in aggregate["schedules"]],
                [0, 1],
            )
            with self.assertRaises(FileExistsError):
                run._record_training_schedule(path, payload)

    def test_result_reward_contract(self) -> None:
        self.assertEqual(batch.result_reward(1), 1.0)
        self.assertEqual(batch.result_reward(2), -1.0)
        self.assertEqual(batch.result_reward(3), 0.0)
        with self.assertRaises(ValueError):
            batch.result_reward(0)

    def test_terminal_and_bootstrapped_gae(self) -> None:
        values = torch.tensor([0.2, 0.4])
        rewards = torch.tensor([0.0, 1.0])
        dones = torch.tensor([False, True])
        advantage, returns = batch.generalized_advantage_estimate(
            values, rewards, dones, gamma=1.0, gae_lambda=1.0
        )
        torch.testing.assert_close(returns, torch.ones(2))
        truncated_advantage, truncated_returns = batch.generalized_advantage_estimate(
            values[:1], torch.zeros(1), torch.zeros(1, dtype=torch.bool),
            bootstrap_value=0.7, gamma=1.0, gae_lambda=1.0
        )
        torch.testing.assert_close(truncated_returns, torch.tensor([0.7]))
        self.assertAlmostEqual(float(truncated_advantage[0]), 0.5, places=6)

    def test_frozen_environment_hash_ignores_only_provenance(self) -> None:
        protocol = importlib.import_module(
            "train.0037_dragapult_value_initialized_rl.rollout.protocol"
        )
        run = importlib.import_module(
            "train.0037_dragapult_value_initialized_rl.training.run_full_semantic"
        )
        job = protocol.RolloutJob(
            "eval-u0000-0000", "opponent", True, 7, 0,
            tuple(range(1, 61)), tuple(range(1, 61)), Path("."),
            policy_seed=11, search_seed=13,
        )
        baseline = run._environment_schedule_sha256([job])
        self.assertEqual(
            baseline,
            run._environment_schedule_sha256([
                replace(job, game_id="eval-u0010-0000", source_policy_update=10)
            ]),
        )
        for changed in (
            replace(job, opponent_id="other"), replace(job, focal_first=False),
            replace(job, seed=8), replace(job, policy_seed=12),
            replace(job, search_seed=14),
        ):
            self.assertNotEqual(baseline, run._environment_schedule_sha256([changed]))


if __name__ == "__main__":
    unittest.main()

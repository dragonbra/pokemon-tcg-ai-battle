from __future__ import annotations

import importlib
import unittest

import torch


batch = importlib.import_module("train.0033_dragapult_third_ptcg_club_rl.training.batch")


class RolloutBatchTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

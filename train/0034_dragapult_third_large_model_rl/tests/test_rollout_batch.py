from __future__ import annotations

import importlib
import unittest

import torch


batch = importlib.import_module("train.0034_dragapult_third_large_model_rl.training.batch")


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

    def test_complete_episode_preparation_stays_on_source_device(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                steps, lanes = 4, 2
                storage = {
                    name: torch.arange(
                        steps * lanes, dtype=torch.float32, device=device
                    ).reshape(steps, lanes, 1)
                    for name in batch.FEATURE_NAMES
                }
                storage.update(
                    {
                        "sequences": torch.full(
                            (steps, lanes, 64), -1, dtype=torch.long, device=device
                        ),
                        "lengths": torch.ones(
                            (steps, lanes), dtype=torch.long, device=device
                        ),
                        "logprob": torch.zeros((steps, lanes), device=device),
                        "value": torch.tensor(
                            [[0.2, 0.1], [0.0, 0.3], [0.4, 0.0], [0.0, 0.5]],
                            device=device,
                        ),
                        "focal_mask": torch.tensor(
                            [[True, True], [False, True], [True, False], [False, True]],
                            device=device,
                        ),
                        "episode_id": torch.tensor(
                            [[0, 1], [0, 1], [2, 1], [2, 3]], device=device
                        ),
                        "terminal_episode_id": torch.tensor(
                            [[-1, -1], [-1, -1], [0, -1], [-1, 1]], device=device
                        ),
                        "terminal_result": torch.tensor(
                            [[0, 0], [0, 0], [1, 0], [0, 2]], device=device
                        ),
                    }
                )
                prepared = batch.prepare_complete_episodes(
                    storage, source_policy_update=7, gamma=1.0, gae_lambda=1.0
                )
                self.assertEqual(prepared.completed_episodes, 2)
                self.assertEqual(prepared.discarded_incomplete_decisions, 2)
                self.assertEqual(prepared.source_policy_update, 7)
                self.assertEqual(prepared.episode_id.tolist(), [0, 1, 1])
                torch.testing.assert_close(
                    prepared.gae_return,
                    torch.tensor([1.0, -1.0, -1.0], device=device),
                )
                torch.testing.assert_close(
                    prepared.episode_weight,
                    torch.tensor([1.0, 0.5, 0.5], device=device),
                )
                self.assertTrue(
                    all(
                        value.device == storage["episode_id"].device
                        for value in prepared.features.values()
                    )
                )


if __name__ == "__main__":
    unittest.main()

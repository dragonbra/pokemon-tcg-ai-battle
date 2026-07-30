from __future__ import annotations

from importlib import import_module
from pathlib import Path
import tempfile
import unittest

import torch


CHECKPOINT = import_module("train.0020_pluggable_deck_rl.rl.checkpoint")
PROTOCOL = import_module("train.0020_pluggable_deck_rl.rl.rollout.protocol")
BATCH = import_module("train.0020_pluggable_deck_rl.rl.training.batch")
RUN = import_module("train.0020_pluggable_deck_rl.rl.run")


def _episode(name: str, reward: float, length: int):
    decision = PROTOCOL.TrajectoryDecision(
        features={"dummy": torch.tensor([[1.0]])},
        action=(0,),
        stopped=False,
        old_log_prob=-0.5,
        old_value=0.0,
        entropy=0.5,
        turn=1,
    )
    return PROTOCOL.Episode(
        episode_id=name,
        opponent="opponent",
        opponent_hash="hash",
        candidate_first=True,
        seed=1,
        valid=True,
        reward=reward,
        winner=0 if reward > 0 else 1 if reward < 0 else None,
        status="finished",
        error=None,
        engine_selections=10,
        final_turn=4,
        complete_rounds=2,
        decisions=[decision for _ in range(length)],
    )


class RLTrainingContractTests(unittest.TestCase):
    def test_terminal_returns_and_episode_weights(self) -> None:
        batch = BATCH.prepare_episodes(
            [_episode("win", 1.0, 2), _episode("loss", -1.0, 4)]
        )
        self.assertTrue(torch.equal(batch.terminal_return[:2], torch.ones(2)))
        self.assertTrue(torch.equal(batch.terminal_return[2:], -torch.ones(4)))
        self.assertAlmostEqual(float(batch.episode_weight[:2].sum()), 1.0)
        self.assertAlmostEqual(float(batch.episode_weight[2:].sum()), 1.0)

    def test_model_only_checkpoint_excludes_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update-000001.pt"
            model = torch.nn.Linear(2, 1)
            CHECKPOINT.save_model_only(
                path, model, update=1, metadata={"test": True}
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertFalse(
            {
                "optimizer",
                "scheduler",
                "scaler",
                "rng_state",
                "rollout",
            }.intersection(payload)
        )

    def test_ppo_parser_freezes_initial_experiment_contract(self) -> None:
        args = RUN.build_parser().parse_args(
            [
                "ppo",
                "--version",
                "V9_dragapult_ppo_initial",
                "--warm-start",
                "checkpoint.pt",
                "--updates",
                "100",
                "--episodes-per-update",
                "512",
                "--gae-lambda",
                "1",
                "--epochs",
                "2",
                "--actor-learning-rate",
                "1e-5",
                "--entropy-coefficient",
                "0",
                "--target-behavior-kl",
                "0.01",
            ]
        )
        self.assertEqual(args.updates, 100)
        self.assertEqual(args.episodes_per_update, 512)
        self.assertEqual(args.gae_lambda, 1.0)
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.actor_learning_rate, 1e-5)
        self.assertEqual(args.entropy_coefficient, 0.0)
        self.assertEqual(args.target_behavior_kl, 0.01)


if __name__ == "__main__":
    unittest.main()

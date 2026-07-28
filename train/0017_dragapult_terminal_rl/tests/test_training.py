from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ..checkpoint import checkpoint_metadata, save_model_only
from ..observability.metrics import OutcomeTracker
from ..rollout.collector import RolloutCollector
from ..rollout.protocol import Episode, TrajectoryDecision
from ..run import build_parser
from ..storage import storage_guard
from ..training.batch import prepare_episodes, training_batch_metrics


def _decision(value: float = 0.0) -> TrajectoryDecision:
    return TrajectoryDecision(
        features={"dummy": torch.tensor([[1.0]])},
        action=(0,),
        stopped=False,
        old_log_prob=-0.5,
        old_value=value,
        entropy=0.5,
        turn=1,
    )


def _episode(name: str, reward: float, length: int, first: bool = True) -> Episode:
    return Episode(
        episode_id=name,
        opponent="opponent",
        opponent_hash="hash",
        candidate_first=first,
        seed=1,
        valid=True,
        reward=reward,
        winner=0 if reward > 0 else 1 if reward < 0 else None,
        status="finished",
        error=None,
        engine_selections=10,
        final_turn=4,
        complete_rounds=2,
        decisions=[_decision() for _ in range(length)],
    )


class TrainingContractTest(unittest.TestCase):
    def test_rollout_collector_rejects_train_mode_policy(self) -> None:
        model = torch.nn.Linear(2, 1)
        with self.assertRaisesRegex(ValueError, "eval mode"):
            RolloutCollector(  # type: ignore[arg-type]
                model, device=torch.device("cpu"), workers=1
            )

    def test_ppo_parser_exposes_gae_lambda(self) -> None:
        args = build_parser().parse_args(
            [
                "ppo",
                "--version",
                "V_test",
                "--warm-start",
                "checkpoint.pt",
                "--gae-lambda",
                "1.0",
            ]
        )
        self.assertEqual(args.gae_lambda, 1.0)

    def test_terminal_returns_and_episode_weights(self) -> None:
        batch = prepare_episodes(
            [_episode("win", 1.0, 2), _episode("loss", -1.0, 4)]
        )
        self.assertEqual(batch.decisions, 6)
        self.assertTrue(torch.equal(batch.terminal_return[:2], torch.ones(2)))
        self.assertTrue(torch.equal(batch.terminal_return[2:], -torch.ones(4)))
        self.assertAlmostEqual(float(batch.episode_weight[:2].sum()), 1.0)
        self.assertAlmostEqual(float(batch.episode_weight[2:].sum()), 1.0)
        self.assertAlmostEqual(float((batch.advantage * batch.episode_weight).sum()), 0.0, places=5)

    def test_training_batch_metrics_are_episode_balanced_and_finite(self) -> None:
        batch = prepare_episodes(
            [_episode("win", 1.0, 2), _episode("loss", -1.0, 8)]
        )
        metrics = training_batch_metrics(batch)
        self.assertAlmostEqual(metrics["ppo/advantage_mean"], 0.0, places=5)
        self.assertAlmostEqual(metrics["ppo/advantage_std"], 1.0, places=5)
        self.assertGreaterEqual(metrics["ppo/return_mean"], -1.0)
        self.assertLessEqual(metrics["ppo/return_mean"], 1.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_outcome_tracker_has_rolling_wilson_and_seats(self) -> None:
        tracker = OutcomeTracker()
        tracker.add(_episode("win", 1.0, 1, True))
        tracker.add(_episode("loss", -1.0, 1, False))
        metrics = tracker.metrics()
        self.assertEqual(metrics["rollout/rolling_100/games"], 2.0)
        self.assertEqual(metrics["rollout/rolling_100/win_rate"], 0.5)
        self.assertLess(metrics["rollout/rolling_100/wilson_low"], 0.5)
        self.assertGreater(metrics["rollout/rolling_100/wilson_high"], 0.5)
        self.assertEqual(metrics["rollout/seat/first/win_rate"], 1.0)

    def test_model_only_checkpoint_excludes_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update-000001.pt"
            model = torch.nn.Linear(2, 1)
            save_model_only(  # type: ignore[arg-type]
                path, model, update=1, metadata={"test": True}
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertFalse(
                {
                    "optimizer",
                    "scheduler",
                    "scaler",
                    "rng_state",
                    "rollout",
                }.intersection(payload)
            )
            self.assertEqual(checkpoint_metadata(path)["update"], 1)

    def test_storage_guard_reports_both_filesystems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = storage_guard(
                Path(directory),
                warning_free_gib=10_000.0,
                hard_stop_free_gib=0.0,
                version_cap_gib=1.0,
            )
            self.assertGreater(snapshot.root_free_gib, 0.0)
            self.assertGreater(snapshot.windows_free_gib, 0.0)
            self.assertTrue(snapshot.warning)


if __name__ == "__main__":
    unittest.main()

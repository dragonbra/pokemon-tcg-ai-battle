from __future__ import annotations

import unittest
from pathlib import Path

import torch

from ..rollout import (
    EpisodeTrajectory,
    LeaguePolicyView,
    RolloutJob,
    TrajectoryDecision,
)
from ..training.batch import _episode_gae, prepare_episodes
from ..training.run import LeagueTrainingConfig, compose_training_config, schedule_jobs
from ..decks import load_deck_plugins
from ..league import DEFAULT_DECK_ROOT


def _episode(update: int, reward: float) -> EpisodeTrajectory:
    deck = tuple(range(1, 61))
    job = RolloutJob("g", "dragapult_ex_001", "opponent", LeaguePolicyView.FROZEN, True, 1, update, deck, deck, Path("runtime"))
    episode = EpisodeTrajectory(job)
    episode.decisions = [
        TrajectoryDecision({}, (0,), True, -0.2, 0.1, 0.0),
        TrajectoryDecision({}, (1,), True, -0.3, 0.2, 0.0),
    ]
    episode.finish(reward, 4)
    return episode


class LeagueTrainingTest(unittest.TestCase):
    def test_formal_config_preserves_initialized_decoder_assets(self) -> None:
        initial = {"checkpoints": {"deck": {"decoder_ref": "x"}}, "catalog": {"deck_count": 48}}
        result = compose_training_config(
            initial, LeagueTrainingConfig("V1_test"), identity={"weights_sha256": "a"},
            catalog_count=48, runtime_root=Path("runtime"),
        )
        self.assertEqual(result["checkpoints"], initial["checkpoints"])
        self.assertFalse(result["live_opponent_updates_enabled"])
        self.assertEqual(result["run"]["games_per_update"], 512)
        self.assertEqual(result["schema_version"], "0022_focal_league_ppo_v1")

    def test_schedule_balances_seats_and_preserves_policy_update(self) -> None:
        plugins = load_deck_plugins(DEFAULT_DECK_ROOT)
        jobs = schedule_jobs(plugins, count=512, update=9, seed=3, runtime_root=Path("runtime"))
        self.assertEqual(len(jobs), 512)
        self.assertEqual(sum(job.focal_first for job in jobs), 256)
        self.assertEqual({job.source_policy_update for job in jobs}, {9})
        self.assertEqual({job.opponent_view for job in jobs}, {LeaguePolicyView.FROZEN, LeaguePolicyView.LIVE})

    def test_frozen_evaluation_covers_every_deck_and_both_seats(self) -> None:
        plugins = load_deck_plugins(DEFAULT_DECK_ROOT)
        jobs = schedule_jobs(plugins, count=0, update=0, seed=3, runtime_root=Path("runtime"), evaluation=True)
        self.assertEqual(len(jobs), 96)
        self.assertEqual({job.opponent_deck_id for job in jobs}, {item.deck_id for item in plugins})
        self.assertTrue(all(job.opponent_view is LeaguePolicyView.FROZEN for job in jobs))
    def test_terminal_gae_gamma_one(self) -> None:
        advantages, returns = _episode_gae([0.1, 0.2], 1.0, gamma=1.0, gae_lambda=1.0)
        self.assertTrue(torch.allclose(torch.tensor(advantages), torch.tensor([0.9, 0.8])))
        self.assertTrue(torch.allclose(torch.tensor(returns), torch.ones(2)))

    def test_prepare_rejects_mixed_behavior_policy_updates(self) -> None:
        with self.assertRaisesRegex(ValueError, "mixes source_policy_update"):
            prepare_episodes([_episode(0, 1.0), _episode(1, -1.0)])

    def test_prepare_records_single_behavior_policy_update(self) -> None:
        batch = prepare_episodes([_episode(7, 1.0)])
        self.assertEqual(batch.source_policy_update, 7)
        self.assertEqual(batch.decisions, 2)


if __name__ == "__main__":
    unittest.main()

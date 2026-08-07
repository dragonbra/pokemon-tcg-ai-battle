from __future__ import annotations

import importlib
from pathlib import Path
import unittest


PROJECT = "train.0034_dragapult_third_large_model_rl"


class FullTerminalCreditTest(unittest.TestCase):
    def _episode(self, steps: int, reward: float, decision_turns=None, *, zero_values=False):
        protocol = importlib.import_module(f"{PROJECT}.rollout.protocol")
        job = protocol.RolloutJob(
            game_id=f"credit-{steps}",
            opponent_id="test",
            focal_first=True,
            seed=1,
            source_policy_update=0,
            focal_deck=tuple(range(1, 61)),
            opponent_deck=tuple(range(1, 61)),
            runtime_root=Path("."),
        )
        decision_turns = decision_turns or list(range(steps))
        decisions = [
            protocol.TrajectoryDecision(
                {},
                (),
                True,
                0.0,
                0.0,
                0.0 if zero_values else index / 100.0,
                0,
                turn=decision_turns[index],
            )
            for index in range(steps)
        ]
        episode = protocol.EpisodeTrajectory(job, decisions=decisions)
        episode.finish(reward, turns=steps)
        return episode

    def test_first_and_last_action_receive_same_terminal_return(self) -> None:
        batch_module = importlib.import_module(f"{PROJECT}.training.batch_full_semantic")
        for steps in (2, 50):
            batch = batch_module.prepare_episodes([self._episode(steps, 1.0)])
            self.assertEqual(batch.gae_return.tolist(), [1.0] * steps)
            self.assertEqual(batch.terminal_return.tolist(), [1.0] * steps)
            self.assertAlmostEqual(float(batch.episode_weight[0]), 1.0 / steps)
            self.assertAlmostEqual(float(batch.episode_weight[-1]), 1.0 / steps)

    def test_ppo_defaults_restore_0023_selection_credit_configuration(self) -> None:
        ppo = importlib.import_module(f"{PROJECT}.training.ppo_full_semantic")
        self.assertEqual(ppo.PPOConfig().gamma, 1.0)
        self.assertEqual(ppo.PPOConfig().gae_lambda, 0.95)
        self.assertEqual(ppo.PPOConfig().credit_clock, "selection")
        self.assertEqual(ppo.PPOConfig().batch_size, 1024)
        self.assertEqual(ppo.PPOConfig().epochs, 4)

    def test_turn_clock_decays_only_across_official_turn_boundaries(self) -> None:
        batch_module = importlib.import_module(f"{PROJECT}.training.batch_full_semantic")
        episode = self._episode(5, 1.0, [1, 1, 3, 3, 5], zero_values=True)
        batch = batch_module.prepare_episodes(
            [episode], gamma=1.0, gae_lambda=0.97, credit_clock="turn"
        )
        expected = [0.97**2, 0.97**2, 0.97, 0.97, 1.0]
        for actual, target in zip(batch.gae_return.tolist(), expected, strict=True):
            self.assertAlmostEqual(actual, target, places=6)
        self.assertEqual(batch.semantic_boundaries, 2)
        self.assertAlmostEqual(batch.first_decision_credit_mean, 0.97**2, places=6)

    def test_selection_clock_retains_legacy_per_decision_decay(self) -> None:
        batch_module = importlib.import_module(f"{PROJECT}.training.batch_full_semantic")
        episode = self._episode(5, 1.0, [1, 1, 3, 3, 5], zero_values=True)
        batch = batch_module.prepare_episodes(
            [episode], gamma=1.0, gae_lambda=0.97, credit_clock="selection"
        )
        expected = [0.97**4, 0.97**3, 0.97**2, 0.97, 1.0]
        for actual, target in zip(batch.gae_return.tolist(), expected, strict=True):
            self.assertAlmostEqual(actual, target, places=6)

    def test_selection_clock_lambda095_matches_0023_credit(self) -> None:
        batch_module = importlib.import_module(f"{PROJECT}.training.batch_full_semantic")
        episode = self._episode(5, 1.0, [1, 1, 3, 3, 5], zero_values=True)
        batch = batch_module.prepare_episodes(
            [episode], gamma=1.0, gae_lambda=0.95, credit_clock="selection"
        )
        expected = [0.95**4, 0.95**3, 0.95**2, 0.95, 1.0]
        for actual, target in zip(batch.gae_return.tolist(), expected, strict=True):
            self.assertAlmostEqual(actual, target, places=6)

    def test_turn_equal_loss_weighting_gives_each_official_turn_equal_mass(self) -> None:
        batch_module = importlib.import_module(f"{PROJECT}.training.batch_full_semantic")
        episode = self._episode(6, 1.0, [1, 1, 1, 3, 5, 5], zero_values=True)
        batch = batch_module.prepare_episodes(
            [episode],
            gamma=1.0,
            gae_lambda=0.97,
            credit_clock="turn",
            loss_weighting="episode_equal_turns",
        )
        expected = [1.0 / 9.0] * 3 + [1.0 / 3.0] + [1.0 / 6.0] * 2
        for actual, target in zip(batch.episode_weight.tolist(), expected, strict=True):
            self.assertAlmostEqual(actual, target, places=6)
        self.assertAlmostEqual(sum(batch.episode_weight.tolist()), 1.0, places=6)
        self.assertEqual(batch.loss_weighting, "episode_equal_turns")

    def test_loss_weighting_rejects_unknown_contract(self) -> None:
        batch_module = importlib.import_module(f"{PROJECT}.training.batch_full_semantic")
        with self.assertRaisesRegex(ValueError, "loss_weighting"):
            batch_module.prepare_episodes(
                [self._episode(2, 1.0)], loss_weighting="selection_count_bias"
            )

    def test_turn_clock_rejects_missing_turn_metadata(self) -> None:
        protocol = importlib.import_module(f"{PROJECT}.rollout.protocol")
        batch_module = importlib.import_module(f"{PROJECT}.training.batch_full_semantic")
        episode = self._episode(2, 1.0, [1, 1])
        episode.decisions[0] = protocol.TrajectoryDecision(
            {}, (), True, 0.0, 0.0, 0.0, 0
        )
        with self.assertRaisesRegex(ValueError, "turn metadata"):
            batch_module.prepare_episodes(
                [episode], gamma=1.0, gae_lambda=0.97, credit_clock="turn"
            )

    def test_ppo_accepts_turn_clock_and_rejects_discounted_gamma(self) -> None:
        ppo = importlib.import_module(f"{PROJECT}.training.ppo_full_semantic")
        ppo.PPOConfig(
            gamma=1.0,
            gae_lambda=0.97,
            credit_clock="turn",
            loss_weighting="episode_equal_turns",
        ).validate()
        with self.assertRaisesRegex(ValueError, "gamma"):
            ppo.PPOConfig(gamma=0.99, gae_lambda=0.97, credit_clock="turn").validate()
        with self.assertRaisesRegex(ValueError, "credit_clock"):
            ppo.PPOConfig(credit_clock="episode").validate()
        with self.assertRaisesRegex(ValueError, "gae_lambda"):
            ppo.PPOConfig(gae_lambda=0.0).validate()
        with self.assertRaisesRegex(ValueError, "loss_weighting"):
            ppo.PPOConfig(loss_weighting="selection_count_bias").validate()


if __name__ == "__main__":
    unittest.main()

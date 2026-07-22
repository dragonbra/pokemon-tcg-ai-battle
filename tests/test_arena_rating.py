from __future__ import annotations

import unittest
from dataclasses import replace

from arena.rating import (
    EloRatingEngine,
    GaussianRatingEngine,
    RatingState,
    apply_checkpoint_status,
    eligible_for_demotion,
)


class RatingTests(unittest.TestCase):
    def test_gaussian_starts_at_600_and_win_moves_winner_up(self) -> None:
        engine = GaussianRatingEngine()
        update = engine.update(RatingState("a"), RatingState("b"), 1.0)

        self.assertEqual(update.before_a.mu, 600.0)
        self.assertGreater(update.after_a.mu, 600.0)
        self.assertLess(update.after_b.mu, 600.0)
        self.assertLess(update.after_a.sigma, 200.0)

    def test_draw_moves_equal_ratings_toward_each_other_without_prize_margin(self) -> None:
        update = GaussianRatingEngine().update(
            RatingState("a", 700),
            RatingState("b", 500),
            0.5,
        )

        self.assertLess(update.after_a.mu, 700)
        self.assertGreater(update.after_b.mu, 500)

    def test_elo_compat_is_replayable_from_same_result(self) -> None:
        update = EloRatingEngine().update(RatingState("a"), RatingState("b"), 1.0)

        self.assertEqual(update.after_a.elo, 616.0)
        self.assertEqual(update.after_b.elo, 584.0)

    def test_demotion_requires_twenty_games_and_two_checkpoints(self) -> None:
        state = RatingState("bad", 299, 120, games=19)
        self.assertFalse(eligible_for_demotion(state, 300))

        state = replace(state, games=20, below_threshold_checkpoints=2)
        self.assertTrue(eligible_for_demotion(state, 300))

    def test_checkpoint_updates_consecutive_low_status_without_deleting_state(self) -> None:
        state = RatingState("bad", mu=299, games=20, below_threshold_checkpoints=1)

        updated = apply_checkpoint_status((state,))[0]

        self.assertEqual(updated.below_threshold_checkpoints, 2)
        self.assertEqual(updated.status, "demoted")


if __name__ == "__main__":
    unittest.main()

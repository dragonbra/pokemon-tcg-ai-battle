from __future__ import annotations

import unittest
from pathlib import Path

from ..rollout import EpisodeTrajectory, LeaguePolicyView, RolloutJob


class RolloutProtocolTest(unittest.TestCase):
    def test_terminal_reward_and_policy_provenance(self) -> None:
        deck = tuple(range(1, 61))
        job = RolloutJob("g1", "dragapult_ex_001", "deck_b", LeaguePolicyView.LIVE, True, 7, 3, deck, deck, Path("runtime"))
        episode = EpisodeTrajectory(job)
        episode.finish(1.0, 12)
        self.assertTrue(episode.valid)
        self.assertEqual(episode.job.source_policy_update, 3)
        with self.assertRaisesRegex(ValueError, "terminal reward"):
            EpisodeTrajectory(job).finish(0.5, 1)


if __name__ == "__main__":
    unittest.main()

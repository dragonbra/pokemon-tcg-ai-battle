from __future__ import annotations

import unittest
from pathlib import Path

from ..rollout import (
    EpisodeTrajectory,
    LeaguePolicyView,
    RolloutJob,
    TrajectoryDecision,
)


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

    def test_two_sided_decisions_keep_policy_identity_and_reward_sign(self) -> None:
        deck = tuple(range(1, 61))
        job = RolloutJob(
            "g2", "dragapult_ex_001", "alakazam_dudunsparce_001",
            LeaguePolicyView.LIVE, True, 8, 11, deck, deck, Path("runtime")
        )
        episode = EpisodeTrajectory(job)
        episode.decisions.extend([
            TrajectoryDecision({}, (0,), True, -0.2, 0.1, 0.0, "dragapult_ex_001", 11, 1),
            TrajectoryDecision({}, (1,), True, -0.3, 0.2, 0.0, "alakazam_dudunsparce_001", 7, -1),
        ])
        episode.finish(1.0, 6)
        self.assertEqual(episode.policy_updates(), {"dragapult_ex_001": {11}, "alakazam_dudunsparce_001": {7}})
        self.assertEqual(episode.reward_for(episode.decisions[0]), 1.0)
        self.assertEqual(episode.reward_for(episode.decisions[1]), -1.0)


if __name__ == "__main__":
    unittest.main()

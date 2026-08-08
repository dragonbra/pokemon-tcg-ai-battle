from __future__ import annotations

from pathlib import Path
import unittest

from ..rollout.cuda_collector import _build_resident_jobs
from ..rollout.protocol import RolloutJob


def _job(game_id: str, policy_seed: int) -> RolloutJob:
    deck = tuple(range(1, 61))
    return RolloutJob(
        game_id=game_id,
        opponent_id="opponent",
        focal_first=True,
        seed=123,
        source_policy_update=4,
        focal_deck=deck,
        opponent_deck=deck,
        runtime_root=Path("runtime"),
        policy_seed=policy_seed,
    )


class CudaCollectorContractTest(unittest.TestCase):
    def test_each_job_keeps_its_own_policy_seed(self) -> None:
        jobs = [_job("a", 10), _job("b", 11)]

        resident = _build_resident_jobs(jobs)

        self.assertEqual([job.policy_seed for job in resident], [10, 11])
        self.assertEqual([job.schedule_index for job in resident], [0, 1])


if __name__ == "__main__":
    unittest.main()

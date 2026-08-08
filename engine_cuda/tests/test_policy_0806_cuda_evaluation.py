from __future__ import annotations

from dataclasses import dataclass
import unittest

from engine_cuda.tools.evaluate_policy_0806_cuda import (
    build_cuda_schedule,
)


@dataclass(frozen=True)
class Entry:
    deck_id: str
    games: int


class Policy0806CudaEvaluationTest(unittest.TestCase):
    def test_schedule_is_paired_seeded_and_seat_balanced(self) -> None:
        schedule = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=(Entry("opponent_a", 2), Entry("opponent_b", 1)),
            evaluation_seed=341_512_806,
        )

        jobs = schedule["jobs"]
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum(job["focal_first"] for job in jobs), 3)
        for offset in range(0, len(jobs), 2):
            first, second = jobs[offset : offset + 2]
            self.assertTrue(first["focal_first"])
            self.assertFalse(second["focal_first"])
            self.assertEqual(first["opponent_id"], second["opponent_id"])
            self.assertEqual(first["engine_seed"], second["engine_seed"])
            self.assertEqual(first["search_seed"], second["search_seed"])

    def test_schedule_changes_with_focal_identity_but_is_reproducible(self) -> None:
        entries = (Entry("opponent_a", 1),)
        left = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=entries,
            evaluation_seed=341_512_806,
        )
        replay = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=entries,
            evaluation_seed=341_512_806,
        )
        other = build_cuda_schedule(
            focal_deck_id="candidate_b",
            entries=entries,
            evaluation_seed=341_512_806,
        )

        self.assertEqual(left, replay)
        self.assertNotEqual(
            left["jobs"][0]["engine_seed"], other["jobs"][0]["engine_seed"]
        )


if __name__ == "__main__":
    unittest.main()

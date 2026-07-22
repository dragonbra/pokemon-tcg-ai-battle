from __future__ import annotations

import unittest

from arena.rating import RatingState
from arena.scheduler import Pairing, build_rating_pairs, build_smoke_pairs


class SchedulerTests(unittest.TestCase):
    def test_smoke_pairs_cover_each_unordered_pair_once(self) -> None:
        pairs = build_smoke_pairs(("a", "b", "c"))

        self.assertEqual(
            pairs,
            (Pairing("a", "b", 0), Pairing("a", "c", 0), Pairing("b", "c", 0)),
        )

    def test_rating_scheduler_prefers_closest_ratings_and_underplayed_pairs(self) -> None:
        states = (
            RatingState("a", 600),
            RatingState("b", 605),
            RatingState("c", 1000),
        )
        pairs = build_rating_pairs(
            states,
            {("a", "b"): 0, ("a", "c"): 5, ("b", "c"): 5},
            7,
            1,
        )

        self.assertEqual(pairs[0].players, ("a", "b"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.official_rng import (  # noqa: E402
    mt19937_next,
    seed_mt19937,
    uniform_below,
)


class OfficialMt19937Test(unittest.TestCase):
    def test_matches_std_mt19937_reference_vector(self) -> None:
        engine = seed_mt19937(5489)
        self.assertEqual(
            [mt19937_next(engine) for _ in range(10)],
            [
                3499211612,
                581869302,
                3890346734,
                3586334585,
                545404204,
                4161255391,
                3922919429,
                949333985,
                2715962298,
                1323567403,
            ],
        )

    def test_seed_and_sequence_are_deterministic(self) -> None:
        left = seed_mt19937(123456789)
        right = seed_mt19937(123456789)
        self.assertEqual(
            [mt19937_next(left) for _ in range(1000)],
            [mt19937_next(right) for _ in range(1000)],
        )
        self.assertEqual(left.index, right.index)
        self.assertEqual(left.draw_count, 1000)

    def test_uniform_below_stays_in_range_and_counts_draws(self) -> None:
        engine = seed_mt19937(7)
        values = [uniform_below(engine, size) for size in range(1, 1000)]
        self.assertTrue(all(value < size for value, size in zip(values, range(1, 1000))))
        self.assertGreaterEqual(engine.draw_count, len(values))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import dataclass
import unittest

from evaluation.frozen_0806_contract import (
    FROZEN_0806_EVALUATION_GAMES,
    FROZEN_0806_EVALUATION_SEED,
    FROZEN_0806_EVALUATION_UNITS,
    FROZEN_0806_UNIT_GAMES,
    evaluation_counts,
    evaluation_game_seed,
    evaluation_schedule_id,
)


@dataclass(frozen=True)
class Entry:
    games: int


class Frozen0806EvaluationContractTests(unittest.TestCase):
    def test_game_seed_is_backend_independent_and_namespaced(self) -> None:
        arguments = {
            "focal_identity": "candidate",
            "opponent_identity": "opponent",
            "slot": 3,
            "replica": 5,
        }
        engine = evaluation_game_seed(**arguments)

        self.assertEqual(engine, evaluation_game_seed(**arguments))
        self.assertNotEqual(engine, evaluation_game_seed(**arguments, namespace="search"))
        self.assertNotEqual(engine, evaluation_game_seed(**{**arguments, "replica": 6}))

    def test_eight_fixed_units_produce_2048_games(self) -> None:
        self.assertEqual(FROZEN_0806_UNIT_GAMES, 256)
        self.assertEqual(FROZEN_0806_EVALUATION_UNITS, 8)
        self.assertEqual(FROZEN_0806_EVALUATION_GAMES, 2048)
        self.assertEqual(FROZEN_0806_EVALUATION_SEED, 341_512_806)
        self.assertEqual(
            evaluation_counts((Entry(3), Entry(251), Entry(2))),
            (24, 2008, 16),
        )

    def test_schedule_identity_is_stable_and_contract_specific(self) -> None:
        base = "a" * 64
        first = evaluation_schedule_id(base)
        second = evaluation_schedule_id(base)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, base)
        self.assertNotEqual(first, evaluation_schedule_id("b" * 64))

    def test_rejects_schedule_that_is_not_one_256_game_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "256"):
            evaluation_counts((Entry(255),))


if __name__ == "__main__":
    unittest.main()

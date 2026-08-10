from __future__ import annotations

from dataclasses import dataclass
import unittest

from evaluation.frozen_0806_contract import (
    FROZEN_0806_CONTRACT_ID,
    FROZEN_0806_CPU_GAMES,
    FROZEN_0806_FIRST_PLAYER_CONTRACT,
    FROZEN_0806_EVALUATION_GAMES,
    FROZEN_0806_EVALUATION_SEED,
    FROZEN_0806_EVALUATION_UNITS,
    FROZEN_0806_UNIT_GAMES,
    evaluation_counts,
    evaluation_coin_winner,
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
        self.assertEqual(FROZEN_0806_CPU_GAMES, 256)
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
        self.assertNotEqual(first, evaluation_schedule_id(base, evaluation_units=1))
        self.assertIn("agent_first_player", FROZEN_0806_CONTRACT_ID)
        self.assertIn("winning_agent", FROZEN_0806_FIRST_PLAYER_CONTRACT)

    def test_seeded_toss_winner_is_backend_independent(self) -> None:
        arguments = {
            "focal_identity": "candidate",
            "opponent_identity": "opponent",
            "slot": 3,
            "replica": 5,
        }
        self.assertIs(type(evaluation_coin_winner(**arguments)), bool)
        self.assertEqual(
            evaluation_coin_winner(**arguments), evaluation_coin_winner(**arguments)
        )

    def test_cpu_replica_zero_is_exact_cuda_replica_zero_subset(self) -> None:
        common = {
            "focal_identity": "candidate-002",
            "opponent_identity": "opponent-007",
            "slot": 3,
        }
        cpu = {
            "engine_seed": evaluation_game_seed(**common, replica=0),
            "search_seed": evaluation_game_seed(
                **common, replica=0, namespace="search"
            ),
            "focal_won_toss": evaluation_coin_winner(**common, replica=0),
        }
        cuda = [
            {
                "engine_seed": evaluation_game_seed(**common, replica=replica),
                "search_seed": evaluation_game_seed(
                    **common, replica=replica, namespace="search"
                ),
                "focal_won_toss": evaluation_coin_winner(
                    **common, replica=replica
                ),
            }
            for replica in range(FROZEN_0806_EVALUATION_UNITS)
        ]

        self.assertEqual(cpu, cuda[0])
        self.assertEqual(len({row["engine_seed"] for row in cuda}), 8)
        self.assertEqual(len({row["search_seed"] for row in cuda}), 8)

    def test_rejects_schedule_that_is_not_one_256_game_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "256"):
            evaluation_counts((Entry(255),))


if __name__ == "__main__":
    unittest.main()

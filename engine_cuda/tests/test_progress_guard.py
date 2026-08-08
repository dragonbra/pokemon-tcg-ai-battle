from __future__ import annotations

import unittest

import torch

from ptcg_cuda_engine.progress_guard import (
    DeviceRepeatForfeitGuard,
    apply_loop_forfeits,
)


def _decision(*, serial: int = 7, turn: int = 4) -> dict[str, torch.Tensor]:
    option_cat = torch.zeros((1, 2, 19), dtype=torch.long)
    option_cat[0, 0, 0] = 10  # Ability action.
    option_cat[0, 0, 5] = 652
    option_cat[0, 0, 12] = 652
    option_cat[0, 0, 13] = 3
    option_cat[0, 0, 18] = serial
    return {
        "option_cat": option_cat,
        "selection_type": torch.tensor([0]),
        "turn": torch.tensor([turn]),
        "actor": torch.tensor([1]),
        "actions": torch.tensor([[0]]),
        "lengths": torch.tensor([1]),
        "ready": torch.tensor([True]),
    }


class DeviceRepeatForfeitGuardTest(unittest.TestCase):
    def test_twentieth_identical_ability_forfeits_acting_player(self) -> None:
        guard = DeviceRepeatForfeitGuard(batch_size=1, limit=20, device="cpu")
        decision = _decision()

        for _ in range(19):
            self.assertFalse(guard.observe(**decision).item())
        self.assertTrue(guard.observe(**decision).item())

    def test_different_ability_and_new_turn_have_independent_counts(self) -> None:
        guard = DeviceRepeatForfeitGuard(batch_size=1, limit=20, device="cpu")
        for _ in range(19):
            guard.observe(**_decision(serial=7, turn=4))

        self.assertFalse(guard.observe(**_decision(serial=8, turn=4)).item())
        self.assertFalse(guard.observe(**_decision(serial=7, turn=5)).item())

    def test_forfeit_writes_opponent_win_and_terminal_status(self) -> None:
        class Engine:
            def __init__(self) -> None:
                self.results = torch.zeros(3, dtype=torch.uint8)
                self.states = torch.ones(3, dtype=torch.int32)

            def game_results(self) -> torch.Tensor:
                return self.results

            def statuses(self) -> torch.Tensor:
                return self.states

        engine = Engine()
        mask = torch.tensor([True, True, False])
        losing_players = torch.tensor([0, 1, 0])

        apply_loop_forfeits(engine, mask, losing_players)

        self.assertEqual(engine.results.tolist(), [2, 1, 0])
        self.assertEqual(engine.states.tolist(), [2, 2, 1])


if __name__ == "__main__":
    unittest.main()

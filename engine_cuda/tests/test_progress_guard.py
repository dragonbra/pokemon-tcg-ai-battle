from __future__ import annotations

import unittest

import torch

from ptcg_cuda_engine.progress_guard import (
    DeviceRepeatForfeitGuard,
    apply_loop_forfeits,
)


def _decision(
    *, serial: int = 7, turn: int = 4, actor: int = 1,
    option_position: int = 0,
) -> dict[str, torch.Tensor]:
    option_cat = torch.zeros((1, 2, 19), dtype=torch.long)
    option_cat[0, 0, 0] = 10  # Ability action.
    option_cat[0, 0, 5] = 652
    option_cat[0, 0, 12] = 652
    option_cat[0, 0, 13] = option_position + 1
    option_cat[0, 0, 18] = serial
    return {
        "option_cat": option_cat,
        "selection_type": torch.tensor([0]),
        "turn": torch.tensor([turn]),
        "actor": torch.tensor([actor]),
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

    def test_different_ability_has_an_independent_count(self) -> None:
        guard = DeviceRepeatForfeitGuard(batch_size=1, limit=20, device="cpu")
        for _ in range(19):
            guard.observe(**_decision(serial=7, turn=4))

        self.assertFalse(guard.observe(**_decision(serial=8, turn=4)).item())

    def test_same_ability_accumulates_across_turns(self) -> None:
        guard = DeviceRepeatForfeitGuard(batch_size=1, limit=20, device="cpu")
        for turn in range(4, 23):
            self.assertFalse(guard.observe(**_decision(serial=7, turn=turn)).item())

        self.assertTrue(guard.observe(**_decision(serial=7, turn=23)).item())

    def test_each_actor_keeps_its_count_when_the_other_actor_selects(self) -> None:
        guard = DeviceRepeatForfeitGuard(batch_size=1, limit=20, device="cpu")
        for _ in range(19):
            self.assertFalse(guard.observe(**_decision(actor=1)).item())
        self.assertFalse(guard.observe(**_decision(actor=0)).item())
        self.assertTrue(guard.observe(**_decision(actor=1)).item())

    def test_same_ability_survives_option_order_changes(self) -> None:
        guard = DeviceRepeatForfeitGuard(batch_size=1, limit=2, device="cpu")

        self.assertFalse(
            guard.observe(**_decision(option_position=0)).item()
        )
        self.assertTrue(
            guard.observe(**_decision(option_position=7)).item()
        )

    def test_multitoken_ability_choice_is_still_counted(self) -> None:
        guard = DeviceRepeatForfeitGuard(batch_size=1, limit=20, device="cpu")
        decision = _decision()
        decision["actions"] = torch.tensor([[0, 1]])
        decision["lengths"] = torch.tensor([2])

        for _ in range(19):
            self.assertFalse(guard.observe(**decision).item())
        self.assertTrue(guard.observe(**decision).item())

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

    def test_lane_reset_clears_only_reused_episode_state(self) -> None:
        guard = DeviceRepeatForfeitGuard(batch_size=2, limit=3, device="cpu")
        first = _decision()
        paired = {
            name: value.expand(2, *value.shape[1:]).clone()
            for name, value in first.items()
        }
        guard.observe(**paired)
        guard.observe(**paired)

        guard.reset(torch.tensor([True, False]))
        forfeits = guard.observe(**paired)

        self.assertEqual(forfeits.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()

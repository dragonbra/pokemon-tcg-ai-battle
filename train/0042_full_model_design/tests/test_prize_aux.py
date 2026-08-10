from __future__ import annotations

import importlib
import unittest

import torch

prize = importlib.import_module("train.0042_full_model_design.integrated.prize")
protocol = importlib.import_module("train.0042_full_model_design.rollout.protocol")


def transition(*, own=0, opp=0, turn=0, done=False):
    action = protocol.CanonicalMacroAction("root", (0,), False)
    return protocol.PolicyTransition(
        {}, action, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, None, 0.0, done,
        (0, 1), metadata={"focal_prizes_taken": own,
                          "opponent_prizes_taken": opp, "own_turn_index": turn},
    )


class PrizeAuxTest(unittest.TestCase):
    def test_directional_and_terminal_neutral(self):
        rows = [transition(own=1), transition(opp=2, done=True)]
        directional = prize.transition_prize_rewards(rows, mode="directional")
        self.assertTrue(torch.allclose(directional, torch.tensor([1 / 24, -2 / 24])))
        neutral = prize.transition_prize_rewards(rows, mode="terminal_neutral")
        self.assertAlmostEqual(float(neutral.sum()), 0.0, places=7)

    def test_own_turn_clock_not_callback_count(self):
        rows = [transition(turn=1), transition(turn=1), transition(turn=3, done=True)]
        targets = prize.prize_gae(rows, torch.zeros(3), mode="directional", gamma_prize=.97)
        self.assertAlmostEqual(float(targets.own_turn_discounts[0]), 1.0, places=6)
        self.assertAlmostEqual(float(targets.own_turn_discounts[1]), .97 ** 2, places=6)
        self.assertEqual(float(targets.own_turn_discounts[2]), 0.0)

    def test_actor_advantages_are_separately_normalized(self):
        win = torch.tensor([1.0, 2.0, 3.0])
        shaped = prize.combine_actor_advantages(win, win * 100, alpha_prize=.2)
        expected = 1.2 * prize.normalize_advantage(win)
        self.assertTrue(torch.allclose(shaped, expected))

    def test_prize_head_uses_query_three(self):
        head = prize.PrizeAuxHead(8)
        self.assertEqual(head(torch.zeros(2, 8, 8)).shape, (2,))


if __name__ == "__main__":
    unittest.main()

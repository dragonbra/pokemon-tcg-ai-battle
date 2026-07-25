from __future__ import annotations

import unittest

from archive.train_legacy.alakazam_bc_rl.mcts import PUCTSearch


class PUCTSearchTests(unittest.TestCase):
    def test_visit_counts_form_a_policy_target(self) -> None:
        # The left root action leads to a leaf valued +1, while the right one
        # leads to -1. The root policy should therefore concentrate on left.
        transitions = {("root", (0,)): "win", ("root", (1,)): "loss"}

        def step(state: str, selection: list[int]) -> str:
            return transitions[(state, tuple(selection))]

        search = PUCTSearch(
            root_player=0,
            simulations=24,
            cpuct=1.0,
            step=step,
            observation=lambda state: state,
            player_index=lambda _state: 0,
            evaluate=lambda state: 1.0 if state == "win" else -1.0 if state == "loss" else 0.0,
            expand=lambda state: [((0,), 0.5), ((1,), 0.5)] if state == "root" else [],
        )
        root = search.run("root")
        self.assertEqual(sum(child.node.visits for child in root.children if child.node), 24)
        self.assertGreater(root.children[0].node.visits, root.children[1].node.visits)
        policy = search.root_policy(root)
        self.assertGreater(policy[0], policy[1])

    def test_opponent_turn_inverts_root_value_for_selection(self) -> None:
        transitions = {("root", (0,)): "good", ("root", (1,)): "bad"}

        search = PUCTSearch(
            root_player=0,
            simulations=16,
            cpuct=1.0,
            step=lambda state, selection: transitions[(state, tuple(selection))],
            observation=lambda state: state,
            player_index=lambda state: 1 if state == "root" else 0,
            evaluate=lambda state: 1.0 if state == "good" else -1.0,
            expand=lambda state: [((0,), 0.5), ((1,), 0.5)] if state == "root" else [],
        )
        root = search.run("root")
        # On the opponent's turn, a root-good action is less attractive.
        self.assertGreater(root.children[1].node.visits, root.children[0].node.visits)


if __name__ == "__main__":
    unittest.main()

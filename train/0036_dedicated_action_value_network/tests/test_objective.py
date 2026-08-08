from __future__ import annotations
import unittest
import torch
from ..model.value_network import ValueOutputs
from ..training.objective import LossWeights, value_objective


class ObjectiveTests(unittest.TestCase):
    def test_primary_and_auxiliary_losses_are_switchable(self) -> None:
        outputs = ValueOutputs(
            torch.tensor([0.0, 0.0], requires_grad=True),
            torch.zeros(2, 15, requires_grad=True),
            torch.zeros(2, 13, requires_grad=True),
        )
        targets = dict(
            value_target=torch.tensor([1.0, 0.0]), archetype_target=torch.tensor([0, 14]),
            final_diff_target=torch.tensor([12, 0]), episode_weight=torch.tensor([0.25, 0.75]),
        )
        base = value_objective(outputs, **targets)
        auxiliary = value_objective(outputs, **targets, weights=LossWeights(archetype=0.1, final_diff=0.1))
        self.assertAlmostEqual(float(base.total.detach()), float(base.value.detach()), places=6)
        self.assertGreater(float(auxiliary.total.detach()), float(base.total.detach()))

    def test_hidden_hand_loss_fails_closed_while_disabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden-hand"):
            LossWeights(hand=0.1).validate()


if __name__ == "__main__":
    unittest.main()

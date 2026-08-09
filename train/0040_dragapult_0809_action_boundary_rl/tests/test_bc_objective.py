from __future__ import annotations

import importlib
import unittest

import torch


bc = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.training.bc_smoke")
model_module = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.model")
fixtures = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.tests.test_contract")


class BcObjectiveTest(unittest.TestCase):
    def test_ordered_objective_is_finite_and_differentiable(self) -> None:
        model = model_module.PodNativeActorCritic(
            model_module.ModelConfig(
                d_model=32,
                heads=4,
                state_layers=1,
                option_layers=1,
                ffn_multiplier=2,
                dropout=0.0,
                max_action_steps=4,
            )
        )
        batch = fixtures.valid_batch()
        batch["min_count"][:] = 1
        batch["max_count"][:] = 2
        actions = torch.tensor([[2, -1], [1, 3]], dtype=torch.long)
        lengths = torch.tensor([1, 2], dtype=torch.long)
        loss, metrics = bc.ordered_bc_objective(model, batch, actions, lengths)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(int(metrics["tokens"]), 4)
        loss.backward()
        self.assertIsNotNone(model.action_decoder.query.weight.grad)


if __name__ == "__main__":
    unittest.main()

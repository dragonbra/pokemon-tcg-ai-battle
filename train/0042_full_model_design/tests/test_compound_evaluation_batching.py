from __future__ import annotations

import importlib
from types import SimpleNamespace
import unittest

import torch


PROJECT = "train.0042_full_model_design"


class CompoundEvaluationBatchingTest(unittest.TestCase):
    def _fixture(self):
        allocation = importlib.import_module(f"{PROJECT}.policy.allocation_head")
        dragapult = importlib.import_module(
            f"{PROJECT}.action_boundary.dragapult"
        )
        width = 16
        batch = 6
        cards = 3
        model = SimpleNamespace(
            allocation_head=allocation.DragapultAllocationHead(width)
        )
        card_cat = torch.zeros(batch, cards, 8, dtype=torch.long)
        card_num = torch.zeros(batch, cards, 6)
        card_mask = torch.ones(batch, cards, dtype=torch.bool)
        serials = (10, 11, 12)
        for index, serial in enumerate(serials):
            card_cat[:, index, 0] = 100 + index
            card_cat[:, index, 1] = serial + 1
            card_cat[:, index, 2] = 2
            card_cat[:, index, 3] = 6
            card_cat[:, index, 4] = index + 1
            card_num[:, index, 0] = 80 + index * 10
            card_num[:, index, 1] = 100 + index * 10
        validated = SimpleNamespace(
            card_mask=card_mask, card_cat=card_cat, card_num=card_num
        )
        state = SimpleNamespace(
            summary=torch.randn(batch, width),
            cards=torch.randn(batch, cards, width),
        )
        options = torch.randn(batch, 4, width)

        def macro(row: int, target_count: int):
            targets = serials[:target_count]
            identities = tuple(
                dragapult.StableTargetIdentity(1, serial, 100 + index, index)
                for index, serial in enumerate(targets)
            )
            allocations = dragapult.enumerate_allocations(identities)
            chosen = allocations[row % len(allocations)]
            visible = [
                {
                    "serial": serial,
                    "id": 100 + index,
                    "benchSlot": index,
                    "hp": 80 + index * 10,
                    "maxHp": 100 + index * 10,
                    "energyCards": [],
                    "preEvolution": [],
                    "statusBits": 0,
                }
                for index, serial in enumerate(targets)
            ]
            return {
                "targets": list(targets),
                "counters": list(chosen.counters),
                "visible_targets": visible,
                "root_index": row % options.shape[1],
            }

        macros = (macro(0, 2), macro(1, 2), None, macro(3, 3), macro(4, 3), None)
        return model, validated, state, options, macros

    def test_batched_matches_scalar_values_and_gradients(self) -> None:
        evaluation = importlib.import_module(
            f"{PROJECT}.policy.compound_evaluation"
        )
        torch.manual_seed(42042)
        model, validated, state, options, macros = self._fixture()
        scalar_logprob, scalar_entropy = (
            evaluation._evaluate_parameter_actions_scalar(
                model, validated, state, options, macros
            )
        )
        scalar_loss = -(scalar_logprob + 0.01 * scalar_entropy).sum()
        scalar_loss.backward()
        scalar_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.allocation_head.named_parameters()
        }
        model.allocation_head.zero_grad(set_to_none=True)

        batched_logprob, batched_entropy = evaluation.evaluate_parameter_actions(
            model, validated, state, options, macros
        )
        batched_loss = -(batched_logprob + 0.01 * batched_entropy).sum()
        batched_loss.backward()
        torch.testing.assert_close(batched_logprob, scalar_logprob)
        torch.testing.assert_close(batched_entropy, scalar_entropy)
        for name, parameter in model.allocation_head.named_parameters():
            torch.testing.assert_close(parameter.grad, scalar_gradients[name])


if __name__ == "__main__":
    unittest.main()

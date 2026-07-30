from __future__ import annotations

from importlib import import_module
import unittest

import torch


ACTOR_CRITIC = import_module(
    "train.0020_pluggable_deck_rl.rl.policy.actor_critic"
)
SHARED_POOL = import_module(
    "train.0020_pluggable_deck_rl.rl.opponent_inference.shared_foundation_pool"
)


class SharedFoundationPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, _metadata = ACTOR_CRITIC.load_source_actor_critic()
        cls.model.eval()

    def test_decoder_head_is_an_independent_parameter_copy(self) -> None:
        first = SHARED_POOL._DecoderHead(self.model.actor)
        second = SHARED_POOL._DecoderHead(self.model.actor)
        first_parameter = next(first.parameters())
        second_parameter = next(second.parameters())
        self.assertNotEqual(first_parameter.data_ptr(), second_parameter.data_ptr())
        original = second_parameter.detach().clone()
        with torch.no_grad():
            first_parameter.add_(1.0)
        self.assertTrue(torch.equal(second_parameter, original))

    def test_routed_head_matches_foundation_greedy_decoder(self) -> None:
        actor = self.model.actor
        head = SHARED_POOL._DecoderHead(actor)
        batch_size = 3
        option_count = 7
        width = actor.config.d_model
        state = torch.randn(batch_size, width)
        options = torch.randn(batch_size, option_count, width)
        batch = {
            "option_mask": torch.tensor(
                [
                    [True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False],
                    [True, True, True, True, True, True, True],
                ]
            ),
            "min_count": torch.tensor([0, 1, 2]),
            "max_count": torch.tensor([1, 3, 4]),
        }
        with torch.inference_mode():
            expected = actor.deterministic_action_tensors(
                batch,
                encoded=(state, options),
            )
            actual = head.greedy(batch, state, options)
        expected_actions = [
            [
                int(value)
                for value in expected.sequences[index, : expected.lengths[index]].tolist()
            ]
            for index in range(batch_size)
        ]
        self.assertEqual(actual, expected_actions)

    def test_stacked_expert_decoder_matches_foundation_for_each_policy(self) -> None:
        actor = self.model.actor
        stacked = SHARED_POOL._StackedDecoderHeads(
            actor, ["first", "second", "third"]
        )
        batch_size = 3
        option_count = 7
        width = actor.config.d_model
        state = torch.randn(batch_size, width)
        options = torch.randn(batch_size, option_count, width)
        batch = {
            "option_mask": torch.tensor(
                [
                    [True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False],
                    [True, True, True, True, True, True, True],
                ]
            ),
            "min_count": torch.tensor([0, 1, 2]),
            "max_count": torch.tensor([1, 3, 4]),
        }
        with torch.inference_mode():
            expected = actor.deterministic_action_tensors(
                batch,
                encoded=(state, options),
            )
            actual = stacked.decode(
                batch,
                state,
                options,
                torch.tensor([0, 1, 2]),
                torch.zeros(batch_size, dtype=torch.bool),
                torch.zeros(batch_size),
            )
        expected_actions = [
            [
                int(value)
                for value in expected.sequences[index, : expected.lengths[index]].tolist()
            ]
            for index in range(batch_size)
        ]
        self.assertEqual(actual, expected_actions)
        with torch.no_grad():
            stacked.pointer_key_weight[0].add_(1.0)
        self.assertFalse(
            torch.equal(stacked.pointer_key_weight[0], stacked.pointer_key_weight[1])
        )


if __name__ == "__main__":
    unittest.main()

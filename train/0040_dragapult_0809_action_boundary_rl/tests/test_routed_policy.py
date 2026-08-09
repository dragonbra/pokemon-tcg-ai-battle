from __future__ import annotations

import importlib
import unittest

import torch


contract = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.contract")
model = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.model")


def batch_mapping(batch_size: int = 3) -> dict[str, torch.Tensor]:
    option_mask = torch.zeros(batch_size, 128, dtype=torch.bool)
    option_mask[:, :5] = True
    return {
        "global_cat": torch.zeros(batch_size, 8, dtype=torch.long),
        "global_num": torch.zeros(batch_size, 16),
        "entity_cat": torch.zeros(batch_size, 128, 6, dtype=torch.long),
        "entity_num": torch.zeros(batch_size, 128, 10),
        "entity_parent": torch.full((batch_size, 128), -1, dtype=torch.long),
        "entity_mask": torch.zeros(batch_size, 128, dtype=torch.bool),
        "option_cat": torch.zeros(batch_size, 128, 12, dtype=torch.long),
        "option_num": torch.zeros(batch_size, 128, 4),
        "option_equiv": torch.zeros(batch_size, 128, dtype=torch.long),
        "option_mask": option_mask,
        "min_count": torch.ones(batch_size, dtype=torch.long),
        "max_count": torch.full((batch_size,), 3, dtype=torch.long),
    }


class ActionEvaluationTest(unittest.TestCase):
    def test_sample_logprob_replays_exactly(self) -> None:
        torch.manual_seed(9)
        actor = model.PodNativeActorCritic(model.ModelConfig(dropout=0.0)).eval()
        batch = contract.PodNativeBatch.from_mapping(batch_mapping())
        with torch.no_grad():
            encoded = actor.encode(batch)
            sampled = actor.action_decoder.generate(
                batch, encoded.option_tokens, encoded.state_summary, stochastic=True
            )
            evaluated = actor.action_decoder.evaluate(
                batch,
                encoded.option_tokens,
                encoded.state_summary,
                sampled.sequences,
                sampled.lengths,
            )
        torch.testing.assert_close(evaluated.logprob, sampled.logprob)


if __name__ == "__main__":
    unittest.main()

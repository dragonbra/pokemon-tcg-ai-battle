from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from ptcg_cuda_engine.semantic0031_bridge import (
    semantic0031_decode_device,
    semantic0031_greedy_decode_device,
    semantic0031_mean_pool_by_parent_device,
)
from ptcg_cuda_engine.semantic0031_router import branched_option_outputs


class _ActionDecoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.initial = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.option_bias = nn.Linear(width, 1)
        self.query = nn.Linear(width, width)
        self.stop = nn.Linear(width, 1)
        self.recurrent = nn.GRUCell(width, width)


class Semantic0031RouterTest(unittest.TestCase):
    def test_relation_pool_uses_only_explicit_masked_children(self) -> None:
        values = torch.tensor(
            [[[2.0, 4.0], [100.0, 200.0], [6.0, 8.0], [50.0, 60.0]]]
        )
        parents = torch.tensor([[1, 1, 1, 2]])
        mask = torch.tensor([[True, False, True, False]])

        pooled = semantic0031_mean_pool_by_parent_device(values, parents, mask, 3)

        torch.testing.assert_close(
            pooled,
            torch.tensor([[[4.0, 6.0], [0.0, 0.0], [0.0, 0.0]]]),
        )

    def test_generic_greedy_decode_matches_compatibility_wrapper(self) -> None:
        torch.manual_seed(7)
        decoder = _ActionDecoder(4).eval()
        batch = SimpleNamespace(
            option_mask=torch.tensor([[True, True, False], [True, True, True]]),
            min_count=torch.tensor([1, 0]),
            max_count=torch.tensor([2, 3]),
        )
        options = torch.randn(2, 3, 4)
        summary = torch.randn(2, 4)

        decoded = semantic0031_decode_device(
            decoder, batch, options, summary, max_select=3, greedy=True
        )
        actions, lengths = semantic0031_greedy_decode_device(
            decoder, batch, options, summary, max_select=3
        )

        torch.testing.assert_close(decoded["actions"], actions)
        torch.testing.assert_close(decoded["lengths"], lengths)
        self.assertEqual(decoded["logprob"].shape, (2,))
        self.assertEqual(decoded["entropy"].shape, (2,))

    def test_greedy_decode_can_skip_distribution_statistics(self) -> None:
        torch.manual_seed(8)
        decoder = _ActionDecoder(4).eval()
        batch = SimpleNamespace(
            option_mask=torch.tensor([[True, True, True]]),
            min_count=torch.tensor([1]),
            max_count=torch.tensor([2]),
        )
        options = torch.randn(1, 3, 4)
        summary = torch.randn(1, 4)

        with mock.patch.object(
            torch.distributions,
            "Categorical",
            side_effect=AssertionError("greedy evaluation must not build distributions"),
        ):
            decoded = semantic0031_decode_device(
                decoder,
                batch,
                options,
                summary,
                max_select=2,
                greedy=True,
                compute_stats=False,
            )

        self.assertEqual(decoded["lengths"].tolist(), [2])
        torch.testing.assert_close(decoded["logprob"], torch.zeros(1))
        torch.testing.assert_close(decoded["entropy"], torch.zeros(1))

    def test_stochastic_decode_is_invariant_to_batch_row_order(self) -> None:
        torch.manual_seed(9)
        decoder = _ActionDecoder(4).eval()
        batch = SimpleNamespace(
            option_mask=torch.tensor([[True, True, True], [True, True, False]]),
            min_count=torch.tensor([1, 1]),
            max_count=torch.tensor([2, 2]),
        )
        options = torch.randn(2, 3, 4)
        summary = torch.randn(2, 4)
        seeds = torch.tensor([101, 202])
        counters = torch.tensor([7, 13])

        direct = semantic0031_decode_device(
            decoder,
            batch,
            options,
            summary,
            max_select=2,
            greedy=False,
            sampling_seeds=seeds,
            sampling_counters=counters,
        )
        reverse = torch.tensor([1, 0])
        reversed_batch = SimpleNamespace(
            option_mask=batch.option_mask[reverse],
            min_count=batch.min_count[reverse],
            max_count=batch.max_count[reverse],
        )
        reordered = semantic0031_decode_device(
            decoder,
            reversed_batch,
            options[reverse],
            summary[reverse],
            max_select=2,
            greedy=False,
            sampling_seeds=seeds[reverse],
            sampling_counters=counters[reverse],
        )

        for name in ("actions", "lengths", "stopped", "logprob", "entropy"):
            torch.testing.assert_close(direct[name], reordered[name][reverse])

    def test_last_option_branch_shares_inputs_and_first_block(self) -> None:
        torch.manual_seed(11)
        layer = nn.TransformerDecoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        transformer = nn.TransformerDecoder(
            layer, num_layers=2, norm=nn.LayerNorm(4)
        ).eval()
        opponent_last = copy.deepcopy(transformer.layers[1]).eval()
        opponent_norm = copy.deepcopy(transformer.norm).eval()
        option_inputs = torch.randn(2, 3, 4)
        batch = SimpleNamespace(
            option_mask=torch.tensor([[True, True, False], [True, True, True]])
        )
        state = SimpleNamespace(
            tokens=torch.randn(2, 5, 4),
            mask=torch.tensor(
                [[True, True, True, False, False], [True, True, True, True, True]]
            ),
        )
        calls = 0

        def count_first(_module, _inputs, _output) -> None:
            nonlocal calls
            calls += 1

        hook = transformer.layers[0].register_forward_hook(count_first)
        adapter = SimpleNamespace(
            model=SimpleNamespace(
                option_encoder=SimpleNamespace(
                    cross_attention_transformer=transformer
                )
            ),
            _encode_option_inputs=lambda _batch, _state: option_inputs,
        )
        masks = {
            "tgt_key_padding_mask": ~batch.option_mask,
            "memory_key_padding_mask": ~state.mask,
        }
        with torch.inference_mode():
            shared = transformer.layers[0](option_inputs, state.tokens, **masks)
            expected_focal = transformer.norm(
                transformer.layers[1](shared, state.tokens, **masks)
            )
            expected_opponent = opponent_norm(
                opponent_last(shared, state.tokens, **masks)
            )
            calls = 0
            focal, opponent = branched_option_outputs(
                adapter, opponent_last, opponent_norm, batch, state
            )
        hook.remove()

        self.assertEqual(calls, 1)
        mask = batch.option_mask.unsqueeze(-1)
        torch.testing.assert_close(focal, expected_focal * mask)
        torch.testing.assert_close(opponent, expected_opponent * mask)


if __name__ == "__main__":
    unittest.main()

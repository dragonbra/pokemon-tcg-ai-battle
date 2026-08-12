from __future__ import annotations

import copy
import math
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
from ptcg_cuda_engine.semantic0031_router import (
    Semantic0031ResidentRouter,
    SharedTrunkIdentityProof,
    branched_option_outputs,
)


class _ActionDecoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.initial = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.option_bias = nn.Linear(width, 1)
        self.query = nn.Linear(width, width)
        self.stop = nn.Linear(width, 1)
        self.recurrent = nn.GRUCell(width, width)


class _Policy(nn.Module):
    def __init__(self, width: int, state_offset: float) -> None:
        super().__init__()
        self.action_decoder = _ActionDecoder(width)
        self.state_offset = state_offset

    def validate_batch(self, batch):
        return batch

    def state_encoder(self, batch, _prototype_memory):
        rows = batch.option_mask.shape[0]
        summary = torch.full((rows, 4), self.state_offset)
        return SimpleNamespace(
            summary=summary,
            tokens=summary.unsqueeze(1),
            mask=torch.ones((rows, 1), dtype=torch.bool),
        )


class _Adapter:
    def __init__(self, model: _Policy, options: torch.Tensor) -> None:
        self.model = model
        self.prototype_memory = None
        self.options = options

    def _encode_options(self, _batch, _state):
        return self.options


class _DeckAwarePolicy(_Policy):
    """Small exact-arithmetic policy used to audit heterogeneous deck rows."""

    def state_encoder(self, batch, _prototype_memory):
        deck_signal = (
            batch.resource_cat[..., 0].float()
            * batch.resource_num[..., 0]
            * batch.resource_mask.float()
        ).sum(dim=1)
        summary = torch.stack(
            (deck_signal, deck_signal * 2, deck_signal * 4, deck_signal * 8),
            dim=1,
        )
        return SimpleNamespace(
            summary=summary,
            tokens=summary.unsqueeze(1),
            mask=torch.ones((summary.shape[0], 1), dtype=torch.bool),
        )


class _DeckAwareAdapter(_Adapter):
    def _encode_options(self, batch, state):
        option_signal = batch.option_values.float()
        return option_signal.unsqueeze(-1) * torch.tensor(
            [1.0, 2.0, 4.0, 8.0]
        ).view(1, 1, 4) + state.summary.unsqueeze(1)


class _CountingPolicy(_Policy):
    def __init__(self, width: int, state_offset: float) -> None:
        super().__init__(width, state_offset)
        self.encoded_batch_sizes: list[int] = []

    def state_encoder(self, batch, prototype_memory):
        self.encoded_batch_sizes.append(int(batch.option_mask.shape[0]))
        return super().state_encoder(batch, prototype_memory)


class _RowAwareAdapter(_Adapter):
    def _encode_options(self, batch, _state):
        return self.options.index_select(0, batch.row_id)


def _slice_namespace(batch, indices):
    return SimpleNamespace(**{
        name: value.index_select(0, indices)
        for name, value in vars(batch).items()
    })


def _first_step_logits(decoder, batch, options, summary):
    hidden = torch.tanh(decoder.initial(summary))
    pointer = (
        decoder.query(hidden).unsqueeze(1) * decoder.key(options)
    ).sum(-1) / math.sqrt(options.shape[-1])
    pointer = pointer + decoder.option_bias(options).squeeze(-1)
    pointer = pointer.masked_fill(
        ~batch.option_mask, torch.finfo(pointer.dtype).min
    )
    stop = decoder.stop(hidden).squeeze(-1)
    stop = stop.masked_fill(
        ~batch.min_count.eq(0), torch.finfo(stop.dtype).min
    )
    return torch.cat((pointer, stop.unsqueeze(1)), dim=1)


class Semantic0031RouterTest(unittest.TestCase):
    def test_cross_policy_role_compaction_matches_legacy_and_avoids_double_rows(self) -> None:
        torch.manual_seed(8_090_042)
        focal_model = _CountingPolicy(4, 0.25).eval()
        opponent_model = _CountingPolicy(4, -0.75).eval()
        options = torch.randn(4, 3, 4)
        audit = {
            "status": "PASS",
            "requested_policy_id": "Policy-0809",
            "effective_policy_sha256": "8" * 64,
        }
        kwargs = {
            "focal_adapter": _RowAwareAdapter(focal_model, options),
            "opponent_adapter": _RowAwareAdapter(opponent_model, options),
            "same_policy": False,
            "requested_opponent_policy_id": "Policy-0809",
            "opponent_identity_audit": audit,
        }
        batch = SimpleNamespace(
            row_id=torch.arange(4),
            option_mask=torch.tensor([
                [True, True, False], [True, True, True],
                [True, False, False], [True, True, True],
            ]),
            min_count=torch.ones(4, dtype=torch.long),
            max_count=torch.full((4,), 2, dtype=torch.long),
        )
        focal_route = torch.tensor([True, False, True, False])
        opponent_route = ~focal_route
        legacy = Semantic0031ResidentRouter(**kwargs).route(
            batch,
            focal_route=focal_route,
            opponent_route=opponent_route,
            max_select=2,
            focal_greedy=True,
            compute_stats=False,
        )
        focal_model.encoded_batch_sizes.clear()
        opponent_model.encoded_batch_sizes.clear()
        compacted = Semantic0031ResidentRouter(
            **kwargs, role_compacted=True
        ).route(
            batch,
            focal_route=focal_route,
            opponent_route=opponent_route,
            max_select=2,
            focal_greedy=True,
            compute_stats=False,
        )
        self.assertTrue(torch.equal(compacted.actions, legacy.actions))
        self.assertTrue(torch.equal(compacted.lengths, legacy.lengths))
        self.assertTrue(torch.equal(compacted.stopped, legacy.stopped))
        self.assertEqual(focal_model.encoded_batch_sizes, [2])
        self.assertEqual(opponent_model.encoded_batch_sizes, [2])

    def test_focal_strategy_readout_is_used_at_every_cuda_decode_step(self) -> None:
        decoder = _ActionDecoder(4).eval()
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.zero_()
            decoder.key.weight.copy_(torch.eye(4))
            decoder.query.weight.copy_(torch.eye(4))
        batch = SimpleNamespace(
            option_mask=torch.tensor([[True, True]]),
            min_count=torch.tensor([1]),
            max_count=torch.tensor([1]),
        )
        options = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
        summary = torch.zeros((1, 4))
        base = semantic0031_decode_device(
            decoder,
            batch,
            options,
            summary,
            max_select=1,
            greedy=True,
            compute_stats=False,
        )
        adapted = semantic0031_decode_device(
            decoder,
            batch,
            options,
            summary,
            max_select=1,
            greedy=True,
            compute_stats=False,
            readout_fn=lambda hidden: hidden + torch.tensor(
                [[0.0, 2.0, 0.0, 0.0]]
            ),
        )
        self.assertEqual(base["actions"].tolist(), [[0]])
        self.assertEqual(adapted["actions"].tolist(), [[1]])

    def test_router_applies_focal_strategy_and_reuses_auxiliary(self) -> None:
        focal_model = _Policy(4, 0.0).eval()
        opponent_model = _Policy(4, 0.0).eval()
        for model in (focal_model, opponent_model):
            with torch.no_grad():
                for parameter in model.action_decoder.parameters():
                    parameter.zero_()
                model.action_decoder.key.weight.copy_(torch.eye(4))
                model.action_decoder.query.weight.copy_(torch.eye(4))
        options = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
        )
        auxiliary = {"value": torch.tensor([0.25])}

        def focal_strategy(_validated, _state, _options):
            return (
                lambda hidden: hidden + torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
                auxiliary,
            )

        router = Semantic0031ResidentRouter(
            focal_adapter=_Adapter(focal_model, options),
            opponent_adapter=_Adapter(opponent_model, options),
            same_policy=False,
            requested_opponent_policy_id="Policy-0809",
            opponent_identity_audit={
                "status": "PASS",
                "requested_policy_id": "Policy-0809",
                "effective_policy_sha256": "8" * 64,
            },
            focal_strategy_fn=focal_strategy,
        )
        batch = SimpleNamespace(
            option_mask=torch.tensor([[True, True]]),
            min_count=torch.tensor([1]),
            max_count=torch.tensor([1]),
        )
        routed = router.route(
            batch,
            focal_route=torch.tensor([True]),
            opponent_route=torch.tensor([False]),
            max_select=1,
            focal_greedy=True,
            compute_stats=True,
        )
        self.assertEqual(routed.actions.tolist(), [[1]])
        self.assertIs(routed.focal_auxiliary, auxiliary)
        expected_logprob = torch.log_softmax(torch.tensor([[0.0, 1.0]]), dim=1)[
            0, 1
        ]
        torch.testing.assert_close(routed.focal_logprob[0], expected_logprob)

    def test_focal_strategy_receives_compacted_resident_job_indices(self) -> None:
        focal_model = _Policy(4, 0.0).eval()
        opponent_model = _Policy(4, 0.0).eval()
        options = torch.zeros((2, 2, 4))
        observed: list[int] = []

        def focal_strategy(_validated, _state, _options, job_indices):
            observed.extend(job_indices.tolist())
            return (lambda hidden: hidden), {"value": torch.zeros(job_indices.numel())}

        router = Semantic0031ResidentRouter(
            focal_adapter=_RowAwareAdapter(focal_model, options),
            opponent_adapter=_RowAwareAdapter(opponent_model, options),
            same_policy=False,
            requested_opponent_policy_id="Policy-0809",
            opponent_identity_audit={
                "status": "PASS",
                "requested_policy_id": "Policy-0809",
                "effective_policy_sha256": "8" * 64,
            },
            focal_strategy_fn=focal_strategy,
            role_compacted=True,
        )
        batch = SimpleNamespace(
            row_id=torch.arange(2),
            option_mask=torch.ones((2, 2), dtype=torch.bool),
            min_count=torch.ones(2, dtype=torch.long),
            max_count=torch.ones(2, dtype=torch.long),
        )
        router.route(
            batch,
            focal_route=torch.tensor([True, False]),
            opponent_route=torch.tensor([False, True]),
            max_select=1,
            focal_greedy=True,
            compute_stats=True,
            job_indices=torch.tensor([7, 11]),
        )
        self.assertEqual(observed, [7])

    def test_heterogeneous_deck_batch_matches_grouped_and_permuted_reference(self) -> None:
        torch.manual_seed(809)
        model = _DeckAwarePolicy(4, 0.0).eval()
        adapter = _DeckAwareAdapter(model, torch.empty(0))
        router = Semantic0031ResidentRouter(
            focal_adapter=adapter,
            same_policy=True,
        )
        batch = SimpleNamespace(
            option_mask=torch.tensor([
                [True, True, False],
                [True, True, True],
                [True, False, False],
                [True, True, True],
            ]),
            min_count=torch.ones(4, dtype=torch.long),
            max_count=torch.ones(4, dtype=torch.long),
            option_values=torch.tensor([
                [1, 3, 0], [4, 2, 1], [5, 0, 0], [2, 6, 3]
            ]),
            resource_cat=torch.tensor([
                [[1], [2]], [[3], [4]], [[1], [2]], [[3], [4]]
            ]),
            resource_num=torch.tensor([
                [[30.0], [30.0]], [[20.0], [40.0]],
                [[45.0], [15.0]], [[10.0], [50.0]],
            ]),
            resource_mask=torch.ones((4, 2), dtype=torch.bool),
        )
        route = torch.ones(4, dtype=torch.bool)

        validated, state, options, _ = router.encode(batch)
        logits = _first_step_logits(
            model.action_decoder, validated, options, state.summary
        )
        heterogeneous = router.route(
            batch,
            focal_route=route,
            opponent_route=~route,
            max_select=1,
            focal_greedy=True,
            compute_stats=False,
        )

        grouped_logits = torch.empty_like(logits)
        grouped_actions = torch.empty_like(heterogeneous.actions)
        grouped_masks = torch.empty_like(batch.option_mask)
        # Rows 0/2 and 1/3 represent two exact-deck groups.
        for indices in (torch.tensor([0, 2]), torch.tensor([1, 3])):
            group = _slice_namespace(batch, indices)
            group_route = torch.ones(len(indices), dtype=torch.bool)
            group_validated, group_state, group_options, _ = router.encode(group)
            grouped_logits.index_copy_(
                0,
                indices,
                _first_step_logits(
                    model.action_decoder,
                    group_validated,
                    group_options,
                    group_state.summary,
                ),
            )
            grouped_masks.index_copy_(0, indices, group_validated.option_mask)
            grouped_actions.index_copy_(
                0,
                indices,
                router.route(
                    group,
                    focal_route=group_route,
                    opponent_route=~group_route,
                    max_select=1,
                    focal_greedy=True,
                    compute_stats=False,
                ).actions,
            )

        self.assertTrue(torch.equal(logits, grouped_logits))
        self.assertTrue(torch.equal(validated.option_mask, grouped_masks))
        self.assertTrue(torch.equal(heterogeneous.actions, grouped_actions))

        permutation = torch.tensor([3, 0, 2, 1])
        inverse = torch.argsort(permutation)
        permuted = _slice_namespace(batch, permutation)
        permuted_validated, permuted_state, permuted_options, _ = router.encode(permuted)
        permuted_logits = _first_step_logits(
            model.action_decoder,
            permuted_validated,
            permuted_options,
            permuted_state.summary,
        ).index_select(0, inverse)
        permuted_actions = router.route(
            permuted,
            focal_route=route,
            opponent_route=~route,
            max_select=1,
            focal_greedy=True,
            compute_stats=False,
        ).actions.index_select(0, inverse)
        permuted_masks = permuted.option_mask.index_select(0, inverse)

        self.assertTrue(torch.equal(logits, permuted_logits))
        self.assertTrue(torch.equal(validated.option_mask, permuted_masks))
        self.assertTrue(torch.equal(heterogeneous.actions, permuted_actions))

    def test_cross_policy_partial_adapter_hard_fails(self) -> None:
        focal = SimpleNamespace(model=SimpleNamespace(action_decoder=object()))
        with self.assertRaisesRegex(RuntimeError, "complete opponent adapter"):
            Semantic0031ResidentRouter(
                focal_adapter=focal,
                opponent_last_option_layer=object(),
                opponent_option_norm=object(),
                opponent_decoder=object(),
                same_policy=False,
            )

    def test_grouped_full_policy_route_matches_independent_inference(self) -> None:
        torch.manual_seed(31)
        batch = SimpleNamespace(
            option_mask=torch.tensor([
                [True, True, True], [True, True, False],
                [True, True, True], [True, True, False],
            ]),
            min_count=torch.tensor([1, 1, 1, 1]),
            max_count=torch.tensor([2, 2, 2, 2]),
        )
        focal_model = _Policy(4, 0.25).eval()
        opponent_model = _Policy(4, -0.75).eval()
        focal_adapter = _Adapter(focal_model, torch.randn(4, 3, 4))
        opponent_adapter = _Adapter(opponent_model, torch.randn(4, 3, 4))
        audit = {
            "status": "PASS",
            "requested_policy_id": "Policy-Opponent",
            "effective_policy_sha256": "f" * 64,
        }
        router = Semantic0031ResidentRouter(
            focal_adapter=focal_adapter,
            opponent_adapter=opponent_adapter,
            same_policy=False,
            requested_opponent_policy_id="Policy-Opponent",
            opponent_identity_audit=audit,
        )
        focal_route = torch.tensor([True, False, True, False])
        opponent_route = ~focal_route

        grouped = router.route(
            batch,
            focal_route=focal_route,
            opponent_route=opponent_route,
            max_select=2,
            focal_greedy=True,
            compute_stats=False,
        )
        focal_state = focal_model.state_encoder(batch, None)
        opponent_state = opponent_model.state_encoder(batch, None)
        focal = semantic0031_decode_device(
            focal_model.action_decoder, batch, focal_adapter.options,
            focal_state.summary, max_select=2, greedy=True, compute_stats=False,
        )
        opponent = semantic0031_decode_device(
            opponent_model.action_decoder, batch, opponent_adapter.options,
            opponent_state.summary, max_select=2, greedy=True, compute_stats=False,
        )
        expected_actions = torch.where(
            focal_route[:, None], focal["actions"], opponent["actions"]
        )
        expected_lengths = torch.where(
            focal_route, focal["lengths"], opponent["lengths"]
        )
        expected_stopped = torch.where(
            focal_route, focal["stopped"], opponent["stopped"]
        )

        torch.testing.assert_close(grouped.actions, expected_actions)
        torch.testing.assert_close(grouped.lengths, expected_lengths)
        torch.testing.assert_close(grouped.stopped, expected_stopped)

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
            proof = SharedTrunkIdentityProof(
                focal_policy_id="Policy-A",
                opponent_policy_id="Policy-B",
                focal_component_sha256={
                    name: "a" * 64 for name in (
                        "prototype_encoder", "state_encoder", "option_input_encoder",
                        "option_transformer_layer_0",
                    )
                },
                opponent_component_sha256={
                    name: "a" * 64 for name in (
                        "prototype_encoder", "state_encoder", "option_input_encoder",
                        "option_transformer_layer_0",
                    )
                },
            )
            focal, opponent = branched_option_outputs(
                adapter, opponent_last, opponent_norm, batch, state,
                identity_proof=proof,
            )
        hook.remove()

        self.assertEqual(calls, 1)
        mask = batch.option_mask.unsqueeze(-1)
        torch.testing.assert_close(focal, expected_focal * mask)
        torch.testing.assert_close(opponent, expected_opponent * mask)

    def test_last_option_branch_rejects_mismatched_shared_weights(self) -> None:
        proof = SharedTrunkIdentityProof(
            focal_policy_id="Policy-0809",
            opponent_policy_id="Policy-0806",
            focal_component_sha256={"prototype_encoder": "a" * 64},
            opponent_component_sha256={"prototype_encoder": "b" * 64},
        )
        with self.assertRaisesRegex(RuntimeError, "no effective-weight identity proof"):
            proof.validate()


if __name__ == "__main__":
    unittest.main()

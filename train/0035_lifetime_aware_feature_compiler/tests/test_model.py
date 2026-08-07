from __future__ import annotations

import importlib
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn


CONTRACTS = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.contracts"
)
DOMAIN = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.domain"
)
MODEL = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.model"
)
_valid_mapping = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.tests.test_contracts"
)._valid_mapping
FEATURES = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.features.compiler"
)
COLLATE = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.features.collate"
)
TYPED_FIELDS = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.model.typed_fields"
)
_row = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.tests.test_features"
)._row


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(28)
        self.model = MODEL.SemanticPolicy(
            MODEL.ModelConfig(
                d_model=64,
                heads=4,
                state_layers=1,
                option_layers=1,
                dropout=0.0,
                max_card_id=8,
                max_attack_id=8,
                max_skill_id=8,
                max_effect_id=8,
                max_options=8,
                max_action_steps=4,
            ),
            DOMAIN.PrototypeIndex.empty(),
        ).eval()
        self.batch = CONTRACTS.DecisionBatch.from_mapping(_valid_mapping())

    def test_forward_exposes_four_named_stages(self) -> None:
        with torch.inference_mode():
            state = self.model.encode_state(self.batch)
            options = self.model.encode_options(self.batch, state)
            decoder_state = self.model.action_decoder.initialize(self.batch, state.summary)
            logits = self.model.decode_next(self.batch, options, decoder_state)
        self.assertEqual(state.tokens.ndim, 3)
        self.assertEqual(state.cards.shape[:2], self.batch.card_mask.shape)
        self.assertEqual(options.shape[:2], self.batch.option_mask.shape)
        self.assertEqual(logits.shape, (self.batch.batch_size, self.batch.option_count + 1))
        self.assertTrue(torch.isfinite(logits).all())

    def test_eval_builds_prototype_memory_once_and_reuses_exact_logits(self) -> None:
        encoder = self.model.prototype_encoder
        with mock.patch.object(
            encoder, "encode_all", wraps=encoder.encode_all
        ) as encode_all:
            with torch.inference_mode():
                first = self.model(self.batch)
                second = self.model(self.batch)
        self.assertEqual(encode_all.call_count, 1)
        torch.testing.assert_close(second, first, rtol=0.0, atol=0.0)
        memory = self.model.prepare_prototype_cache()
        for tensor in (memory.cards, memory.attacks, memory.skills, memory.effects):
            self.assertEqual(tensor.device, next(self.model.parameters()).device)
            self.assertEqual(tensor.dtype, next(self.model.parameters()).dtype)
            self.assertFalse(tensor.requires_grad)
        self.assertEqual(
            self.model.prototype_cache_stats(),
            {"builds": 1, "hits": 2, "invalidations": 0},
        )

    def test_device_dtype_and_state_load_invalidate_prototype_cache(self) -> None:
        with torch.inference_mode():
            self.model(self.batch)
        self.assertEqual(self.model.prototype_cache_stats()["builds"], 1)
        self.model.to(dtype=torch.float64)
        self.assertEqual(self.model.prototype_cache_stats()["invalidations"], 1)
        self.model.to(dtype=torch.float32)
        state = {name: value.clone() for name, value in self.model.state_dict().items()}
        self.model.load_state_dict(state, strict=True)
        self.assertEqual(self.model.prototype_cache_stats()["invalidations"], 1)
        with torch.inference_mode():
            self.model(self.batch)
        self.assertEqual(self.model.prototype_cache_stats()["builds"], 2)

    def test_trainable_prototypes_bypass_cache_and_keep_autograd(self) -> None:
        self.model.train()
        encoder = self.model.prototype_encoder
        with mock.patch.object(
            encoder, "encode_all", wraps=encoder.encode_all
        ) as encode_all:
            loss = self.model.teacher_logits(self.batch).sum()
            loss.backward()
        self.assertEqual(encode_all.call_count, 1)
        self.assertIsNotNone(encoder.card_identity.weight.grad)
        self.assertEqual(self.model.prototype_cache_stats()["builds"], 0)

    def test_frozen_prototypes_cache_in_train_mode_and_downstream_gets_gradients(self) -> None:
        for parameter in self.model.prototype_encoder.parameters():
            parameter.requires_grad_(False)
        self.model.train()
        encoder = self.model.prototype_encoder
        with mock.patch.object(
            encoder, "encode_all", wraps=encoder.encode_all
        ) as encode_all:
            first = self.model.teacher_logits(self.batch)
            second = self.model.teacher_logits(self.batch)
            (first.sum() + second.sum()).backward()
        self.assertEqual(encode_all.call_count, 1)
        self.assertEqual(self.model.prototype_cache_stats()["builds"], 1)
        self.assertIsNotNone(self.model.state_encoder.global_cat.embedding.weight.grad)

    def test_packed_categorical_fields_match_independent_reference(self) -> None:
        vocabularies = (3, 5, 2)
        packed = TYPED_FIELDS.CategoricalFields(vocabularies, 7)
        reference = nn.ModuleList(
            nn.Embedding(size, 7, padding_idx=0) for size in vocabularies
        )
        with torch.no_grad():
            cursor = 1
            for embedding, size in zip(reference, vocabularies, strict=True):
                embedding.weight[0].zero_()
                packed.embedding.weight[cursor : cursor + size - 1].copy_(
                    embedding.weight[1:]
                )
                cursor += size - 1
        values = torch.tensor(
            [[[0, 1, 1], [2, 4, 0]], [[1, 0, 1], [2, 3, 1]]],
            dtype=torch.long,
        )
        expected = sum(
            embedding(values[..., index])
            for index, embedding in enumerate(reference)
        )
        actual = packed(values)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        cursor = 1
        for embedding, size in zip(reference, vocabularies, strict=True):
            torch.testing.assert_close(
                packed.embedding.weight.grad[cursor : cursor + size - 1],
                embedding.weight.grad[1:],
            )
            cursor += size - 1
        self.assertTrue(torch.equal(packed.embedding.weight.grad[0], torch.zeros(7)))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            packed(torch.tensor([[0, 5, 0]]))

    def test_packed_numeric_fields_match_independent_reference_and_gradients(self) -> None:
        width, d_model = 4, 6
        packed = TYPED_FIELDS.NumericFields(width, d_model)
        projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            for _ in range(width)
        )
        states = nn.ModuleList(
            nn.Embedding(4, d_model, padding_idx=0) for _ in range(width)
        )
        with torch.no_grad():
            packed.state_embedding.weight[0].zero_()
            for index, (projection, state_embedding) in enumerate(
                zip(projections, states, strict=True)
            ):
                packed.first_weight[index].copy_(projection[0].weight[:, 0])
                packed.first_bias[index].copy_(projection[0].bias)
                packed.second_weight[index].copy_(projection[2].weight)
                packed.second_bias[index].copy_(projection[2].bias)
                start = 1 + index * 3
                packed.state_embedding.weight[start : start + 3].copy_(
                    state_embedding.weight[1:]
                )
        values = torch.randn(2, 3, width)
        field_states = torch.tensor(
            [[[1, 0, 2, 3], [1, 1, 1, 1], [0, 2, 3, 1]]] * 2
        )
        expected = values.new_zeros((2, 3, d_model))
        for index, (projection, state_embedding) in enumerate(
            zip(projections, states, strict=True)
        ):
            present = field_states[..., index].eq(1).unsqueeze(-1)
            expected = expected + projection(values[..., index : index + 1]) * present
            expected = expected + state_embedding(field_states[..., index])
        actual = packed(values, field_states)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for index, (projection, state_embedding) in enumerate(
            zip(projections, states, strict=True)
        ):
            torch.testing.assert_close(
                packed.first_weight.grad[index], projection[0].weight.grad[:, 0]
            )
            torch.testing.assert_close(
                packed.second_weight.grad[index], projection[2].weight.grad
            )
            start = 1 + index * 3
            torch.testing.assert_close(
                packed.state_embedding.weight.grad[start : start + 3],
                state_embedding.weight.grad[1:],
            )
        changed = values.clone()
        changed[field_states.ne(1)] += 1000
        torch.testing.assert_close(
            packed(changed, field_states), actual.detach(), rtol=1e-6, atol=1e-6
        )

    def test_hierarchical_state_preserves_all_family_tokens(self) -> None:
        with torch.inference_mode():
            state = self.model.encode_state(self.batch)
        expected_lengths = (
            1,
            self.batch.card_cat.shape[1],
            self.batch.resource_cat.shape[1],
            self.batch.event_cat.shape[1],
        )
        self.assertEqual(state.family_lengths, expected_lengths)
        self.assertEqual(state.tokens.shape[1], sum(expected_lengths))
        self.assertEqual(state.mask.shape[1], sum(expected_lengths))
        self.assertTrue(torch.equal(state.tokens[~state.mask], torch.zeros_like(state.tokens[~state.mask])))

    def test_resource_and_event_families_each_affect_policy_logits(self) -> None:
        baseline = self.batch.as_dict()
        resource_changed = {name: value.clone() for name, value in baseline.items()}
        event_changed = {name: value.clone() for name, value in baseline.items()}
        resource_changed["resource_num"][:, 0, 0] += 3.0
        event_changed["event_num"][:, 0, 0] += 3.0
        with torch.inference_mode():
            logits = self.model(baseline)
            resource_logits = self.model(resource_changed)
            event_logits = self.model(event_changed)
        self.assertFalse(torch.allclose(logits, resource_logits))
        self.assertFalse(torch.allclose(logits, event_logits))

    def test_teacher_and_greedy_actions_are_finite_and_legal(self) -> None:
        one = {
            name: value[:1].clone()
            for name, value in self.batch.items()
        }
        with torch.inference_mode():
            teacher = self.model.teacher_logits(one)
            action = self.model.greedy_action(one)
        self.assertTrue(torch.isfinite(teacher).all())
        self.assertGreaterEqual(len(action), 1)
        self.assertLessEqual(len(action), int(one["max_count"][0]))
        self.assertEqual(len(action), len(set(action)))

    def test_real_feature_compiled_row_has_finite_bucketed_forward(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the formal compile boundary")
        prototypes = DOMAIN.PrototypeIndex.load(
            Path(
                "train/0035_lifetime_aware_feature_compiler/assets/"
                "official_public_prototypes_v1.json"
            )
        )
        row, snapshot = _row(15, energy_units=[1, 1, 1, 6])
        record = FEATURES.compile_canonical_row(row, snapshot, prototypes)
        batch = COLLATE.collate_canonical_records(
            [record], bucket_padding=COLLATE.BucketPadding()
        )
        validated = CONTRACTS.DecisionBatch.from_mapping(batch).to("cuda")
        model = MODEL.SemanticPolicy(
            MODEL.ModelConfig(
                d_model=32,
                heads=4,
                state_layers=1,
                option_layers=1,
                ffn_multiplier=2,
                dropout=0.0,
            ),
            prototypes,
        ).eval().cuda()
        with torch.inference_mode():
            eager = model(validated, teacher_forcing=True)
            compiled = torch.compile(
                model, backend="eager", dynamic=False, fullgraph=True
            )(validated, teacher_forcing=True)
        self.assertEqual(eager.shape[:2], batch["targets"].shape)
        self.assertTrue(torch.isfinite(compiled).all())
        torch.testing.assert_close(compiled, eager)

    def test_physical_energy_card_identity_reaches_policy_logits(self) -> None:
        lightning = self.batch.as_dict()
        grass = {name: value.clone() for name, value in lightning.items()}
        lightning = {name: value.clone() for name, value in lightning.items()}
        lightning["card_cat"][:, 1, 0] = 1
        grass["card_cat"][:, 1, 0] = 2
        lightning["option_source"][:] = 2
        grass["option_source"][:] = 2
        with torch.inference_mode():
            left = self.model(lightning)
            right = self.model(grass)
        self.assertFalse(torch.allclose(left, right))

    def test_policy_declares_exact_batch_contract(self) -> None:
        self.assertEqual(self.model.expected_batch_keys, CONTRACTS.EXPECTED_BATCH_KEYS)
        invalid = self.batch.as_dict()
        invalid["team_name"] = torch.zeros(self.batch.batch_size, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "extra=.*team_name"):
            self.model(invalid)

    def test_shared_prototype_memory_matches_direct_encoding(self) -> None:
        encoder = self.model.prototype_encoder
        memory = encoder.encode_all()
        cases = (
            (encoder.card, memory.card, encoder.config.max_card_id),
            (encoder.attack, memory.attack, encoder.config.max_attack_id),
            (encoder.skill, memory.skill, encoder.config.max_skill_id),
            (encoder.effect, memory.effect, encoder.config.max_effect_id),
        )
        with torch.inference_mode():
            for direct, shared, maximum in cases:
                identities = torch.arange(maximum + 1)
                self.assertTrue(torch.equal(direct(identities), shared(identities)))

    def test_shared_prototype_lookups_use_embedding_backward(self) -> None:
        memory = self.model.prototype_encoder.encode_all()
        lookups = (
            memory.card(torch.tensor([1, 2, 1])),
            memory.attack(torch.tensor([1, 2, 1])),
            memory.skill(torch.tensor([1, 2, 1])),
            memory.effect(torch.tensor([1, 2, 1])),
        )
        for lookup in lookups:
            self.assertEqual(type(lookup.grad_fn).__name__, "EmbeddingBackward0")

    def test_cached_decoder_constants_preserve_logits(self) -> None:
        with torch.inference_mode():
            batch, state, options = self.model.encode(self.batch)
            decoder_state = self.model.action_decoder.initialize(batch, state.summary)
            direct = self.model.action_decoder.logits(batch, options, decoder_state)
            cached = self.model.action_decoder.logits(
                batch,
                options,
                decoder_state,
                option_keys=self.model.action_decoder.key(options),
                option_bias=self.model.action_decoder.option_bias(options).squeeze(-1),
            )
        self.assertTrue(torch.equal(direct, cached))

    def test_shared_prototype_policy_matches_direct_recomputation(self) -> None:
        direct = MODEL.SemanticPolicy(
            self.model.config,
            DOMAIN.PrototypeIndex.empty(),
            share_prototype_embeddings=False,
        ).eval()
        direct.load_state_dict(self.model.state_dict())
        with torch.inference_mode():
            shared_logits = self.model.teacher_logits(self.batch)
            direct_logits = direct.teacher_logits(self.batch)
        torch.testing.assert_close(
            shared_logits, direct_logits, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()

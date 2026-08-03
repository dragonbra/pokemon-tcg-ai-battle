from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import torch


CONTRACTS = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.contracts"
)
DOMAIN = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.domain"
)
MODEL = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.model"
)
_valid_mapping = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_contracts"
)._valid_mapping
FEATURES = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.features.compiler"
)
COLLATE = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.features.collate"
)
_row = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_features"
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
                "train/0031_rule_faithful_semantic_foundation_pretraining/assets/"
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

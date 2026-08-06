from __future__ import annotations

import importlib
import unittest

import torch


BASE = "train.0034_clean_recent3_semantic_bc"
CONTRACTS = importlib.import_module(f"{BASE}.contracts")
FIELDS = importlib.import_module(f"{BASE}.contracts.fields")
DOMAIN = importlib.import_module(f"{BASE}.domain")
MODEL = importlib.import_module(f"{BASE}.model")
_valid_mapping = importlib.import_module(f"{BASE}.tests.test_contracts")._valid_mapping


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(32)
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
                max_options=8,
                max_action_steps=4,
            ),
            DOMAIN.PrototypeIndex.empty(),
        ).eval()
        mapping = _valid_mapping()
        mapping["card_cat"][:, 1, 0] = 2
        self.batch = CONTRACTS.DecisionBatch.from_mapping(mapping)

    def _logits(self, mapping: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(mapping)

    def _assert_mutation_changes_logits(
        self,
        name: str,
        mutate,
    ) -> None:
        left = {key: value.clone() for key, value in self.batch.items()}
        right = {key: value.clone() for key, value in self.batch.items()}
        mutate(right)
        before = self._logits(left)
        after = self._logits(right)
        self.assertFalse(torch.allclose(before, after), msg=f"field path is inert: {name}")

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

    def test_every_declared_categorical_and_numeric_column_changes_logits(self) -> None:
        categorical = {
            "global_cat": len(FIELDS.GLOBAL_CAT_FIELDS),
            "card_cat": len(FIELDS.CARD_CAT_FIELDS),
            "resource_cat": len(FIELDS.RESOURCE_CAT_FIELDS),
            "event_cat": len(FIELDS.EVENT_CAT_FIELDS),
            "option_cat": len(FIELDS.OPTION_CAT_FIELDS),
        }
        numeric = {
            "global_num": len(FIELDS.GLOBAL_NUM_FIELDS),
            "card_num": len(FIELDS.CARD_NUM_FIELDS),
            "resource_num": len(FIELDS.RESOURCE_NUM_FIELDS),
            "event_num": len(FIELDS.EVENT_NUM_FIELDS),
            "option_num": len(FIELDS.OPTION_NUM_FIELDS),
        }
        for tensor_name, width in categorical.items():
            for column in range(width):
                if tensor_name == "global_cat":
                    def mutate(row, tensor_name=tensor_name, column=column):
                        row[tensor_name][0, column] = 2
                else:
                    def mutate(row, tensor_name=tensor_name, column=column):
                        row[tensor_name][0, 0, column] = 2
                self._assert_mutation_changes_logits(f"{tensor_name}[{column}]", mutate)
        for tensor_name, width in numeric.items():
            for column in range(width):
                if tensor_name == "global_num":
                    def mutate(row, tensor_name=tensor_name, column=column):
                        row[tensor_name][0, column] = 1.0
                elif tensor_name == "resource_num":
                    def mutate(row, tensor_name=tensor_name, column=column):
                        row[tensor_name][0, :, column] = 1.0
                else:
                    def mutate(row, tensor_name=tensor_name, column=column):
                        row[tensor_name][0, 0, column] = 1.0
                self._assert_mutation_changes_logits(f"{tensor_name}[{column}]", mutate)

    def test_missingness_and_all_explicit_relations_change_logits(self) -> None:
        for state_name in (
            "global_state", "card_state", "resource_state", "event_state", "option_state"
        ):
            if state_name == "global_state":
                def mutate(row, state_name=state_name):
                    row[state_name][0, 0] = 2
            else:
                def mutate(row, state_name=state_name):
                    row[state_name][0, 0, 0] = 2
            self._assert_mutation_changes_logits(state_name, mutate)

        for relation in (
            "card_parent", "event_source", "event_target", "option_source", "option_target"
        ):
            def mutate(row, relation=relation):
                row[relation][0, 0] = 2
            self._assert_mutation_changes_logits(relation, mutate)

    def test_option_card_roles_are_not_collapsed_by_unordered_summing(self) -> None:
        for left_column, right_column in ((5, 6), (5, 9), (9, 10)):
            left = {key: value.clone() for key, value in self.batch.items()}
            right = {key: value.clone() for key, value in self.batch.items()}
            left["option_cat"][0, 0, left_column] = 1
            left["option_cat"][0, 0, right_column] = 2
            right["option_cat"][0, 0, left_column] = 2
            right["option_cat"][0, 0, right_column] = 1
            before = self._logits(left)
            after = self._logits(right)
            self.assertFalse(
                torch.allclose(before, after),
                msg=f"option roles collapsed: {left_column}, {right_column}",
            )

        source_then_target = {key: value.clone() for key, value in self.batch.items()}
        target_then_source = {key: value.clone() for key, value in self.batch.items()}
        source_then_target["option_source"][0, 0] = 1
        source_then_target["option_target"][0, 0] = 2
        target_then_source["option_source"][0, 0] = 2
        target_then_source["option_target"][0, 0] = 1
        before = self._logits(source_then_target)
        after = self._logits(target_then_source)
        self.assertFalse(torch.allclose(before, after))

    def test_event_target_card_and_attack_use_prototype_memory(self) -> None:
        self.assertTrue({3, 4}.issubset(self.model.state_encoder.event_cat.skip_indices))

        self._assert_mutation_changes_logits(
            "event_target_card_prototype",
            lambda row: row["event_cat"].__setitem__((0, 0, 3), 2),
        )
        self._assert_mutation_changes_logits(
            "event_attack_prototype",
            lambda row: row["event_cat"].__setitem__((0, 0, 4), 2),
        )

    def test_event_card_and_instance_roles_are_directional(self) -> None:
        for left_column, right_column in ((2, 3), (7, 8), (9, 10)):
            left = {key: value.clone() for key, value in self.batch.items()}
            right = {key: value.clone() for key, value in self.batch.items()}
            left["event_cat"][0, 0, left_column] = 1
            left["event_cat"][0, 0, right_column] = 2
            right["event_cat"][0, 0, left_column] = 2
            right["event_cat"][0, 0, right_column] = 1
            before = self._logits(left)
            after = self._logits(right)
            self.assertFalse(
                torch.allclose(before, after),
                msg=f"event roles collapsed: {left_column}, {right_column}",
            )

        source_then_target = {key: value.clone() for key, value in self.batch.items()}
        target_then_source = {key: value.clone() for key, value in self.batch.items()}
        source_then_target["event_source"][0, 0] = 1
        source_then_target["event_target"][0, 0] = 2
        target_then_source["event_source"][0, 0] = 2
        target_then_source["event_target"][0, 0] = 1
        before = self._logits(source_then_target)
        after = self._logits(target_then_source)
        self.assertFalse(torch.allclose(before, after))

    def test_selection_bounds_reach_decoder_legality(self) -> None:
        one = {name: value[:1].clone() for name, value in self.batch.items()}
        relaxed = {name: value.clone() for name, value in one.items()}
        relaxed["min_count"][0] = 0
        with torch.inference_mode():
            strict_logits = self.model(one)
            relaxed_logits = self.model(relaxed)
        termination = self.batch.option_count
        self.assertLess(strict_logits[0, termination], relaxed_logits[0, termination])

    def test_teacher_and_greedy_actions_are_finite_and_legal(self) -> None:
        one = {name: value[:1].clone() for name, value in self.batch.items()}
        with torch.inference_mode():
            teacher = self.model.teacher_logits(one)
            action = self.model.greedy_action(one)
        self.assertTrue(torch.isfinite(teacher).all())
        self.assertGreaterEqual(len(action), 1)
        self.assertLessEqual(len(action), int(one["max_count"][0]))
        self.assertEqual(len(action), len(set(action)))

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
        torch.testing.assert_close(shared_logits, direct_logits, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

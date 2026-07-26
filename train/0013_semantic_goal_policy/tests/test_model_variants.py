from __future__ import annotations

import importlib
import unittest

import torch

registry = importlib.import_module("train.0013_semantic_goal_policy.model.registry")
variants = importlib.import_module("train.0013_semantic_goal_policy.model.variants")
decoder = importlib.import_module("train.0013_semantic_goal_policy.model.decoder")


def batch(batch_size=2, options=5):
    return {
        "state_cat": torch.zeros(batch_size, 8, dtype=torch.long),
        "state_num": torch.randn(batch_size, 16),
        "entities_cat": torch.zeros(batch_size, 12, 8, dtype=torch.long),
        "entities_num": torch.randn(batch_size, 12, 12),
        "entity_semantic": torch.randn(batch_size, 12, 64),
        "entity_mask": torch.ones(batch_size, 12, dtype=torch.bool),
        "deck_card_ids": torch.randint(1, 100, (batch_size, 22)),
        "deck_multiplicity": torch.ones(batch_size, 22),
        "deck_semantic": torch.randn(batch_size, 22, 64),
        "deck_mask": torch.ones(batch_size, 22, dtype=torch.bool),
        "ledger_cat": torch.zeros(batch_size, 16, 6, dtype=torch.long),
        "ledger_num": torch.randn(batch_size, 16, 12),
        "ledger_semantic": torch.randn(batch_size, 16, 64),
        "ledger_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "events_cat": torch.zeros(batch_size, 10, 6, dtype=torch.long),
        "events_num": torch.randn(batch_size, 10, 8),
        "event_semantic": torch.randn(batch_size, 10, 64),
        "event_mask": torch.ones(batch_size, 10, dtype=torch.bool),
        "relations": torch.zeros(batch_size, 12, 12, dtype=torch.long),
        "options_cat": torch.zeros(batch_size, options, 12, dtype=torch.long),
        "options_num": torch.randn(batch_size, options, 8),
        "option_semantic": torch.randn(batch_size, options, 64),
        "option_mask": torch.ones(batch_size, options, dtype=torch.bool),
        "min_count": torch.ones(batch_size, dtype=torch.long),
        "max_count": torch.full((batch_size,), min(3, options), dtype=torch.long),
    }


class ModelVariantTest(unittest.TestCase):
    def test_registry_contains_explicit_m0_to_m5(self):
        self.assertEqual(tuple(registry.MODEL_REGISTRY), ("M0", "M1", "M2", "M3", "M4", "M5"))

    def test_all_variants_forward_and_reference_budget(self):
        values = batch()
        for name in registry.MODEL_REGISTRY:
            with self.subTest(name=name):
                model = registry.create_model(name)
                output = model.encode(values)
                self.assertEqual(output.state.shape, (2, model.config.d_model))
                self.assertEqual(output.options.shape, (2, 5, model.config.d_model))
                self.assertTrue(torch.isfinite(output.value).all())
        m5 = registry.create_model("M5")
        count = sum(parameter.numel() for parameter in m5.parameters())
        self.assertGreaterEqual(count, 18_000_000)
        self.assertLessEqual(count, 28_000_000)

    def test_variant_feature_isolation(self):
        values = batch()
        changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in values.items()}
        changed["ledger_num"] += 100
        changed["events_num"] += 100
        for name in ("M0", "M1", "M2", "M3"):
            model = registry.create_model(name).eval()
            with torch.no_grad():
                first = model.encode(values).state
                second = model.encode(changed).state
            self.assertTrue(torch.equal(first, second), name)
        for name in ("M4", "M5"):
            model = registry.create_model(name).eval()
            with torch.no_grad():
                first = model.encode(values).state
                second = model.encode(changed).state
            self.assertFalse(torch.equal(first, second), name)

    def test_m5_relation_bias_is_isolated(self):
        values = batch()
        changed = dict(values)
        changed["relations"] = values["relations"].clone()
        changed["relations"][:, 0, 1] = 3
        for name in ("M0", "M1", "M2", "M3", "M4"):
            model = registry.create_model(name).eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model.encode(values).state, model.encode(changed).state), name)
        model = registry.create_model("M5").eval()
        with torch.no_grad():
            self.assertFalse(torch.equal(model.encode(values).state, model.encode(changed).state))

    def test_option_permutation_does_not_change_state(self):
        values = batch(options=5)
        order = torch.tensor([3, 0, 4, 1, 2])
        permuted = dict(values)
        permuted["options_cat"] = values["options_cat"][:, order]
        permuted["options_num"] = values["options_num"][:, order]
        permuted["option_semantic"] = values["option_semantic"][:, order]
        for name in registry.MODEL_REGISTRY:
            model = registry.create_model(name).eval()
            with torch.no_grad():
                first, second = model.encode(values), model.encode(permuted)
            self.assertTrue(torch.allclose(first.state, second.state, atol=1e-6), name)
            self.assertTrue(torch.allclose(first.options[:, order], second.options, atol=1e-6), name)

    def test_batched_decoder_matches_single_decision_contract(self):
        values = batch(batch_size=4, options=6)
        values["options_cat"][:, :, 0] = torch.arange(1, 7)
        values["option_mask"][1, 4:] = False
        values["max_count"] = torch.tensor([1, 2, 3, 4])
        model = registry.create_model("M0").eval()
        with torch.no_grad():
            batched = model.deterministic_actions(values)
            singles = tuple(
                decoder.sample_action(
                    model.action_scorer(values, row),
                    option_mask=values["option_mask"][row],
                    min_count=int(values["min_count"][row]),
                    max_count=int(values["max_count"][row]),
                    deterministic=True,
                )
                for row in range(4)
            )
        self.assertEqual(batched.sequences, tuple(result.sequence for result in singles))
        self.assertEqual(
            batched.forced_terminal, tuple(result.forced_terminal for result in singles)
        )
        self.assertEqual(batched.legal, (True, True, True, True))

    def test_goal_qkv_has_four_named_contexts(self):
        model = registry.create_model("M3").eval()
        with torch.no_grad():
            output = model.encode(batch())
        self.assertEqual(output.goals.shape, (2, 4, model.config.d_model))
        self.assertEqual(model.goal_roles, ("setup_board", "attack_prize", "resource_recovery", "tempo_survival"))


if __name__ == "__main__":
    unittest.main()

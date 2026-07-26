from __future__ import annotations

import importlib
import unittest

import torch

registry = importlib.import_module("train.0013_semantic_goal_policy.model.registry")
variants = importlib.import_module("train.0013_semantic_goal_policy.model.variants")
decoder = importlib.import_module("train.0013_semantic_goal_policy.model.decoder")
reproducibility = importlib.import_module(
    "train.0013_semantic_goal_policy.training.reproducibility"
)
metrics = importlib.import_module("train.0013_semantic_goal_policy.training.metrics")
runtime_batch = importlib.import_module(
    "train.0013_semantic_goal_policy.training.runtime_batch"
)


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
        self.assertEqual(
            tuple(registry.MODEL_REGISTRY),
            ("M0", "M1", "M2", "M3", "M4", "M5", "M5.1"),
        )

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

    def test_bc_paths_skip_inactive_value_head(self):
        values = batch()
        targets = torch.zeros(2, 1, dtype=torch.long)
        model = registry.create_model("M0").eval()
        value_forward = model.value_head.forward
        calls = 0

        def counted_value(state):
            nonlocal calls
            calls += 1
            return value_forward(state)

        model.value_head.forward = counted_value
        with torch.no_grad():
            model.teacher_logits(values, targets)
            model.deterministic_action_tensors(values)
        self.assertEqual(calls, 0)
        with torch.no_grad():
            encoded = model.encode(values)
        self.assertEqual(calls, 1)
        self.assertTrue(torch.isfinite(encoded.value).all())

    def test_unused_deck_path_is_skipped_before_m2(self):
        values = batch()
        for name, expected_calls in (("M0", 0), ("M1", 0), ("M2", 1)):
            model = registry.create_model(name).eval()
            deck = model._deck
            calls = 0

            def counted_deck(current):
                nonlocal calls
                calls += 1
                return deck(current)

            model._deck = counted_deck
            with torch.no_grad():
                model.encode(values)
            self.assertEqual(calls, expected_calls, name)

    def test_target_trimming_preserves_autoregressive_logits_and_loss(self):
        values = batch(batch_size=2, options=5)
        values["targets"] = torch.tensor(
            [[0, 1, 5, 5, 5, 5], [2, 5, 5, 5, 5, 5]]
        )
        values["target_mask"] = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, False, False, False, False],
            ]
        )
        compacted = runtime_batch.trim_target_padding(values)
        model = registry.create_model("M0").eval()
        with torch.no_grad():
            full_logits = model.teacher_logits(values, values["targets"])
            compact_logits = model.teacher_logits(
                compacted,
                compacted["targets"],
            )
        self.assertTrue(torch.equal(full_logits[:, :3], compact_logits))
        full_loss = torch.nn.functional.cross_entropy(
            full_logits[values["target_mask"]],
            values["targets"][values["target_mask"]],
        )
        compact_loss = torch.nn.functional.cross_entropy(
            compact_logits[compacted["target_mask"]],
            compacted["targets"][compacted["target_mask"]],
        )
        self.assertTrue(torch.equal(full_loss, compact_loss))

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
        for name in ("M4", "M5", "M5.1"):
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
        for name in ("M5", "M5.1"):
            with self.subTest(name=name):
                model = registry.create_model(name).eval()
                with torch.no_grad():
                    self.assertFalse(
                        torch.equal(model.encode(values).state, model.encode(changed).state)
                    )

    def test_m5_relation_bias_preserves_edge_topology(self):
        values = batch()
        values["entities_cat"][:, :, 2] = torch.arange(1, 13)
        forward = dict(values)
        reverse = dict(values)
        forward["relations"] = values["relations"].clone()
        reverse["relations"] = values["relations"].clone()
        forward["relations"][:, 0, 1] = 3
        reverse["relations"][:, 1, 0] = 3
        model = registry.create_model("M5").eval()
        with torch.no_grad():
            first = model.encode(forward).state
            second = model.encode(reverse).state
        self.assertFalse(torch.equal(first, second))

    def test_deck_summary_is_cumulative_from_m2_through_m5(self):
        class ZeroGoals(torch.nn.Module):
            def forward(self, state, resources, mask):
                del resources, mask
                return state.new_zeros(state.size(0), 4, state.size(1))

        values = batch()
        changed = dict(values)
        changed["deck_semantic"] = values["deck_semantic"].clone() + 1
        for name in ("M2", "M3", "M4", "M5", "M5.1"):
            with self.subTest(name=name):
                model = registry.create_model(name).eval()
                model.goal_qkv = ZeroGoals()
                with torch.no_grad():
                    first = model.encode(values).state
                    second = model.encode(changed).state
                self.assertFalse(torch.equal(first, second))

    def test_m5_1_matches_m5_initial_state_exactly(self):
        reproducibility.seed_everything(20260726)
        m5 = registry.create_model("M5")
        reproducibility.seed_everything(20260726)
        ablation = registry.create_model("M5.1")
        self.assertEqual(
            reproducibility.model_state_sha256(m5),
            reproducibility.model_state_sha256(ablation),
        )
        for key, value in m5.state_dict().items():
            self.assertTrue(torch.equal(value, ablation.state_dict()[key]), key)

    def test_m5_1_only_disconnects_goal_qkv_from_policy_forward(self):
        values = batch()
        reproducibility.seed_everything(7)
        m5 = registry.create_model("M5").eval()
        reproducibility.seed_everything(7)
        ablation = registry.create_model("M5.1").eval()

        m5_output = m5.encode(values)
        ablation_output = ablation.encode(values)
        self.assertFalse(torch.equal(m5_output.state, ablation_output.state))
        self.assertTrue(torch.equal(ablation_output.goals, torch.zeros_like(ablation_output.goals)))
        self.assertTrue(torch.isfinite(ablation_output.value).all())

        m5_output.options[..., 0].sum().backward()
        ablation_output.options[..., 0].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and bool(parameter.grad.abs().sum())
                for parameter in m5.goal_qkv.parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in ablation.goal_qkv.parameters())
        )

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

    def test_evaluation_reuses_one_encoding_for_teacher_and_greedy_metrics(self):
        values = batch(batch_size=5, options=6)
        values["min_count"] = torch.tensor([1, 1, 1, 1, 0])
        values["max_count"] = torch.tensor([1, 4, 1, 2, 0])
        values["targets"] = torch.tensor(
            [[0, 6], [1, 6], [2, 6], [3, 6], [6, 6]]
        )
        values["target_mask"] = torch.tensor(
            [
                [True, False],
                [True, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        model = registry.create_model("M0").eval()
        encode = model.encode
        calls = 0

        def counted_encode(current, **kwargs):
            nonlocal calls
            calls += 1
            return encode(current, **kwargs)

        model.encode = counted_encode
        result = metrics.evaluate_full_pass(
            model,
            [values],
            device=torch.device("cpu"),
            namespace="validation",
            amp=False,
        )
        self.assertEqual(calls, 1)
        self.assertEqual(result["bc/validation/decisions"], 5.0)
        self.assertEqual(result["bc/validation/tokens"], 6.0)
        decoded = model.deterministic_actions(values)
        expected_forced = (True, False, True, False, False)
        termination_correct = sum(
            actual == expected
            for actual, expected in zip(decoded.forced_terminal, expected_forced)
        )
        self.assertEqual(
            result["bc/validation/termination_accuracy"],
            termination_correct / 5,
        )
        self.assertGreaterEqual(result["bc/validation/legal_action"], 0.0)
        self.assertLessEqual(result["bc/validation/legal_action"], 1.0)

    def test_goal_qkv_has_four_named_contexts(self):
        model = registry.create_model("M3").eval()
        with torch.no_grad():
            output = model.encode(batch())
        self.assertEqual(output.goals.shape, (2, 4, model.config.d_model))
        self.assertEqual(model.goal_roles, ("setup_board", "attack_prize", "resource_recovery", "tempo_survival"))


if __name__ == "__main__":
    unittest.main()

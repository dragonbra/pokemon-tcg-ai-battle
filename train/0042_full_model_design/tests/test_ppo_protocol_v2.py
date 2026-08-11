from __future__ import annotations

import importlib
import unittest

import torch


ppo = importlib.import_module(
    "train.0042_full_model_design.training.ppo_full_semantic"
)


class PPOProtocolV2Test(unittest.TestCase):
    def test_behavior_guard_sample_is_deterministic_unique_and_rollout_wide(self) -> None:
        first = ppo.behavior_guard_indices(22_137, 4_096, source_policy_update=7)
        second = ppo.behavior_guard_indices(22_137, 4_096, source_policy_update=7)
        other = ppo.behavior_guard_indices(22_137, 4_096, source_policy_update=8)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(first.numel(), 4_096)
        self.assertEqual(first.unique().numel(), first.numel())
        self.assertLess(int(first.min()), 100)
        self.assertGreater(int(first.max()), 22_037)

    def test_behavior_guard_uses_every_row_for_small_rollouts(self) -> None:
        indices = ppo.behavior_guard_indices(37, 4_096, source_policy_update=0)
        self.assertTrue(torch.equal(indices, torch.arange(37)))

    def test_behavior_guard_always_includes_required_macro_rows(self) -> None:
        required = torch.tensor([3, 11_000, 22_136])
        indices = ppo.behavior_guard_indices(
            22_137, 4_096, source_policy_update=0, required_indices=required
        )
        self.assertTrue(set(required.tolist()).issubset(indices.tolist()))

    def test_contiguous_feature_gather_matches_direct_minibatch_collation(self) -> None:
        batching = importlib.import_module(
            "train.0042_full_model_design.policy.batching"
        )
        items = tuple({
            "global_cat": torch.full((1, 12), index, dtype=torch.long),
            "option_num": torch.arange(2 * (index + 2), dtype=torch.float32)
            .reshape(1, index + 2, 2) + index,
            "option_mask": torch.ones((1, index + 2), dtype=torch.bool),
        } for index in range(7))
        contiguous = batching.collate_feature_batches(items)
        widths = {
            name: torch.tensor([item[name].shape[1] for item in items])
            for name, value in items[0].items() if value.ndim >= 2
        }
        indices = torch.tensor([4, 2, 3, 0])
        expected = batching.collate_feature_batches(
            [items[int(index)] for index in indices]
        )
        actual = ppo.index_feature_batch(
            contiguous, indices, torch.device("cpu"), widths
        )
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
            self.assertTrue(actual[name].is_contiguous(), name)

    def test_effective_defaults_are_protocol_v2(self) -> None:
        config = ppo.PPOConfig()
        self.assertEqual(config.protocol_version, "ppo_protocol_v2")
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.batch_size, 2048)
        self.assertEqual(config.actor_learning_rate, 5.0e-6)
        self.assertEqual(config.value_learning_rate, 1.0e-4)
        self.assertEqual(config.clip_ratio, 0.10)
        self.assertEqual(config.entropy_coefficient, 0.003)
        self.assertEqual(config.target_behavior_kl, 0.015)
        self.assertEqual(config.hard_behavior_kl_guard, 0.025)
        self.assertEqual(config.behavior_logprob_mae_limit, 1.0e-4)
        self.assertEqual(config.behavior_guard_samples, 4096)
        self.assertEqual(config.full_behavior_audit_interval, 10)
        config.validate()

    def test_complete_epoch_batches_cover_once_and_keep_tail(self) -> None:
        generator = torch.Generator().manual_seed(420042)
        orders = ppo.complete_epoch_orders(5_001, 3, generator=generator)
        cumulative = torch.zeros(5_001, dtype=torch.int16)
        for epoch, order in enumerate(orders, start=1):
            batches = ppo.epoch_minibatches(order, 2_048)
            self.assertEqual([len(batch) for batch in batches], [2048, 2048, 905])
            self.assertEqual(torch.unique(order).numel(), 5_001)
            cumulative[order] += 1
            self.assertTrue(torch.equal(cumulative, torch.full_like(cumulative, epoch)))

    def test_target_snapshot_detects_any_mutation(self) -> None:
        targets = {
            "rollout_log_prob": torch.tensor([1.0, 2.0]),
            "old_value": torch.tensor([0.1, 0.2]),
            "advantage": torch.tensor([-1.0, 1.0]),
            "gae_return": torch.tensor([-0.5, 0.5]),
        }
        snapshot = ppo.snapshot_frozen_targets(targets)
        ppo.assert_frozen_targets(targets, snapshot)
        targets["advantage"][0] = 0.0
        with self.assertRaisesRegex(RuntimeError, "advantage"):
            ppo.assert_frozen_targets(targets, snapshot)

    def test_usage_metrics_distinguish_examined_and_optimized(self) -> None:
        usage = torch.tensor([0, 1, 1, 2, 4, 5], dtype=torch.int16)
        metrics = ppo.sample_usage_metrics(
            usage, samples_examined=18, samples_optimized=13
        )
        self.assertEqual(metrics["unique_decisions_optimized"], 5.0)
        self.assertAlmostEqual(metrics["coverage_ratio"], 5 / 6)
        self.assertAlmostEqual(metrics["optimized_slots_per_valid_decision"], 13 / 6)
        self.assertAlmostEqual(metrics["optimized_slots_per_unique_decision"], 13 / 5)
        self.assertEqual(metrics["usage_count_0"], 1.0)
        self.assertEqual(metrics["usage_count_1"], 2.0)
        self.assertEqual(metrics["usage_count_2"], 1.0)
        self.assertEqual(metrics["usage_count_4_plus"], 2.0)


if __name__ == "__main__":
    unittest.main()

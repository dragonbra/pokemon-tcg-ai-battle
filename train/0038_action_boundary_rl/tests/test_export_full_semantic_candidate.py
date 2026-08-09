from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import torch


exporter = importlib.import_module(
    "train.0038_action_boundary_rl.export_full_semantic_candidate"
)


class ExportFullSemanticCandidateTest(unittest.TestCase):
    def test_portable_value_network_matches_training_state_contract(self) -> None:
        training_value = importlib.import_module(
            "train.0038_action_boundary_rl.policy.value_network"
        ).LatentQueryValueHead(320, queries=8, layers=2, heads=8, dropout=0.0)
        portable_value = importlib.import_module(
            "train.0038_action_boundary_rl.kaggle_runtime.value_network"
        ).LatentQueryValueHead(320, 8)
        training_state = training_value.state_dict()
        portable_state = portable_value.state_dict()
        self.assertEqual(set(training_state), set(portable_state))
        self.assertEqual(
            {name: tuple(value.shape) for name, value in training_state.items()},
            {name: tuple(value.shape) for name, value in portable_state.items()},
        )

    def test_export_uses_0038_cached_semantic_runtime(self) -> None:
        runtime = exporter.SEMANTIC_RUNTIME_ROOT / "model/policy.py"
        source = runtime.read_text(encoding="utf-8")
        self.assertIn("def prepare_prototype_cache", source)
        self.assertIn("def prototype_cache_stats", source)
        self.assertNotEqual(
            runtime.resolve(),
            Path(
                "evaluation/arena/candidates/"
                "0034_dragapult_third_large_model_zero_shot/strategy/model/policy.py"
            ).resolve(),
        )

    def test_export_uses_exact_frozen_007_deck(self) -> None:
        deck = [
            int(value)
            for value in exporter.FOCAL_DECK_PATH.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        self.assertEqual(len(deck), 60)
        self.assertEqual(exporter._deck_hash(deck), exporter.FOCAL_DECK_SHA256)
        stale = [
            int(value)
            for value in Path(
                "evaluation/arena/candidates/"
                "0034_dragapult_third_large_model_zero_shot/deck.csv"
            ).read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        self.assertNotEqual(exporter._deck_hash(stale), exporter.FOCAL_DECK_SHA256)

    def test_merge_qv_weight_changes_only_q_and_v_rows(self) -> None:
        width = 3
        base = torch.arange(27, dtype=torch.float32).reshape(9, 3)
        q_a = torch.tensor([[1.0, 2.0, 3.0]])
        q_b = torch.tensor([[1.0], [2.0], [3.0]])
        v_a = torch.tensor([[2.0, 1.0, 0.5]])
        v_b = torch.tensor([[3.0], [2.0], [1.0]])

        merged = exporter._merge_qv_weight(
            base,
            q_a=q_a,
            q_b=q_b,
            v_a=v_a,
            v_b=v_b,
            alpha=2.0,
            rank=1,
        )

        torch.testing.assert_close(
            merged[:width], base[:width] + (q_b @ q_a) * 2.0
        )
        self.assertTrue(torch.equal(merged[width : 2 * width], base[width : 2 * width]))
        torch.testing.assert_close(
            merged[2 * width :], base[2 * width :] + (v_b @ v_a) * 2.0
        )
        self.assertTrue(torch.equal(base, torch.arange(27, dtype=torch.float32).reshape(9, 3)))

    def test_merge_qv_weight_rejects_malformed_factors(self) -> None:
        base = torch.zeros(9, 3)
        good_a = torch.zeros(1, 3)
        good_b = torch.zeros(3, 1)
        with self.assertRaises(ValueError):
            exporter._merge_qv_weight(
                base,
                q_a=torch.zeros(2, 3),
                q_b=good_b,
                v_a=good_a,
                v_b=good_b,
                alpha=2.0,
                rank=1,
            )
        with self.assertRaises(ValueError):
            exporter._merge_qv_weight(
                torch.zeros(8, 3),
                q_a=good_a,
                q_b=good_b,
                v_a=good_a,
                v_b=good_b,
                alpha=2.0,
                rank=1,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

import torch


smoke = importlib.import_module(
    "train.0042_full_model_design.diagnostics.smoke_ppo"
)
batching = importlib.import_module(
    "train.0042_full_model_design.policy.batching"
)
sensitivity = importlib.import_module(
    "train.0042_full_model_design.diagnostics.strategy_sensitivity"
)
initialization = importlib.import_module(
    "train.0042_full_model_design.initialization"
)
presets = importlib.import_module(
    "train.0042_full_model_design.integrated.presets"
)
storage = importlib.import_module(
    "train.0042_full_model_design.training.storage_full_semantic"
)


class SmokePPOContractTest(unittest.TestCase):
    def test_fixed_summary_keeps_required_pre_post_metrics(self) -> None:
        value = {
            "auroc": 0.8, "brier": 0.2, "ece_10": 0.1,
            "explained_variance_signed_outcome": 0.3,
            "mean_predicted_win_probability": 0.5, "empirical_win_rate": 0.6,
        }
        meta = {
            "weighted_accuracy": 0.9, "macro_weighted_class_accuracy": 0.8,
            "mean_entropy_nats": 0.2,
        }
        report = {
            "value_overall": value,
            "meta_overall": meta,
            "turn_buckets": [
                {"name": "raw_turn_0", "value": value, "meta": meta},
                {"name": "early_raw_turn_0_5", "value": value, "meta": meta},
                {"name": "middle_raw_turn_6_11", "value": value, "meta": meta},
                {"name": "late_raw_turn_12_plus", "value": value, "meta": meta},
            ],
        }
        summary = smoke._fixed_summary(report)
        self.assertEqual(summary["raw_turn_0_accuracy"], 0.9)
        self.assertEqual(summary["value"]["auroc"], 0.8)
        self.assertEqual(len(summary["turn_buckets"]), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_feature_collation_preserves_cuda_residency(self) -> None:
        items = (
            {"x": torch.ones((1, 2, 3), device="cuda")},
            {"x": torch.ones((1, 4, 3), device="cuda")},
        )
        result = batching.collate_feature_batches(items)
        self.assertEqual(result["x"].device.type, "cuda")
        self.assertEqual(tuple(result["x"].shape), (2, 4, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_feature_roundtrip_benchmark_accepts_contiguous_store(self) -> None:
        features = {
            "x": torch.arange(24, device="cuda").reshape(6, 4),
            "mask": torch.ones((6, 4), dtype=torch.bool, device="cuda"),
        }
        report = smoke._benchmark_feature_roundtrip(features)
        self.assertEqual(report["sample_decisions"], 6.0)
        self.assertGreater(report["feature_bytes_one_way"], 0.0)
        self.assertGreater(report["avoided_transfer_bytes_per_sample"], 0.0)

    def test_smoke_root_is_ignored_diagnostic_namespace(self) -> None:
        self.assertEqual(smoke.SMOKE_ROOT.name, "0042_smoke_ppo")
        self.assertEqual(smoke.SMOKE_ROOT.parent.name, ".tmp")
        self.assertNotEqual(smoke.SMOKE_ROOT, smoke.FORMAL_RUN_ROOT)

    def test_actor_action_targets_are_not_meta_label_leakage(self) -> None:
        self.assertFalse(
            smoke._has_actor_visible_meta_target({"targets": torch.tensor([1])})
        )
        self.assertTrue(
            smoke._has_actor_visible_meta_target(
                {"opponent_meta_label": torch.tensor([1])}
            )
        )

    def test_sensitivity_loads_full_model_checkpoint_inventory(self) -> None:
        deck = smoke.focal_deck()
        model, _ = initialization.build_preset_from_common_update0(
            deck, presets.preset("FULL_MODEL")
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "full-model.pt"
            storage.save_model_only(
                model, checkpoint, update=0,
                metadata={"artifact_role": "TEST_ONLY"},
            )
            report = sensitivity.run(checkpoint)
        self.assertEqual(report["schema_version"], "0042_strategy_sensitivity_v1")


if __name__ == "__main__":
    unittest.main()

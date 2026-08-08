from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ..model.value_network import LatentQueryValueHead
from ..training.checkpoints import load_value_checkpoint, save_value_checkpoint
from ..training.metrics import ValueMetrics
from ..training.trainer import TrainerConfig
from ..model.value_network import ValueOutputs


class TrainingContractTests(unittest.TestCase):
    def test_checkpoint_is_model_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.pt"
            save_value_checkpoint(path, LatentQueryValueHead(16, heads=4, layers=1),
                                  {"epoch": 1, "optimizer_state_saved": False})
            payload = load_value_checkpoint(path)
            self.assertEqual(set(payload), {"schema_version", "value_head_state_dict", "metadata"})
            self.assertFalse(any("optimizer" in key for key in payload["value_head_state_dict"]))

    def test_checkpoint_rejects_recovery_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "recovery"):
                save_value_checkpoint(Path(directory) / "bad.pt", torch.nn.Linear(2, 1),
                                      {"optimizer": {}})

    def test_all_checkpoint_retention_is_mandatory(self) -> None:
        with self.assertRaisesRegex(ValueError, "retention"):
            TrainerConfig(checkpoint_retention="best").validate()

    def test_validation_metrics_include_weighted_auc_and_macro_archetype_accuracy(self) -> None:
        metrics = ValueMetrics()
        outputs = ValueOutputs(
            value_logit=torch.tensor([-2.0, 2.0]),
            archetype_logits=torch.tensor([[3.0] + [0.0] * 14, [0.0, 3.0] + [0.0] * 13]),
            final_diff_logits=torch.zeros(2, 13),
        )
        metrics.update(
            outputs,
            torch.tensor([0.0, 1.0]),
            torch.tensor([0, 1]),
            torch.tensor([6, 6]),
            torch.tensor([0.25, 0.75]),
        )
        result = metrics.compute()
        self.assertEqual(result["value/auroc"], 1.0)
        self.assertEqual(result["aux/archetype_macro_accuracy"], 1.0)

    def test_training_metrics_do_not_retain_all_predictions_for_auc(self) -> None:
        metrics = ValueMetrics(collect_auc=False)
        metrics.update(
            ValueOutputs(torch.tensor([0.0]), torch.zeros(1, 15), torch.zeros(1, 13)),
            torch.tensor([1.0]), torch.tensor([0]), torch.tensor([6]), torch.tensor([1.0]),
        )
        self.assertNotIn("value/auroc", metrics.compute())
        self.assertEqual(metrics.probabilities, [])


if __name__ == "__main__":
    unittest.main()

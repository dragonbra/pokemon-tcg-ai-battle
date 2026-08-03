from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


CHECKPOINTS = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.training.checkpoints"
)


class CheckpointContractTests(unittest.TestCase):
    def test_checkpoint_is_model_only_and_reloadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(3, 2)
            record = CHECKPOINTS.save_checkpoint(
                Path(directory),
                name="best_validation_loss",
                model=model,
                metadata={"arm": "semantic", "epoch": 2},
            )
            payload = torch.load(Path(directory) / record["path"], weights_only=True)
            self.assertEqual(
                set(payload), {"schema_version", "state_dict", "metadata"}
            )
            self.assertFalse(record["optimizer_state_saved"])
            self.assertFalse(record["resumable_training_state_saved"])
            restored = nn.Linear(3, 2)
            restored.load_state_dict(payload["state_dict"], strict=True)

    def test_unsafe_checkpoint_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "checkpoint name"):
                CHECKPOINTS.save_checkpoint(
                    Path(directory),
                    name="../escape",
                    model=nn.Linear(1, 1),
                    metadata={},
                )


if __name__ == "__main__":
    unittest.main()

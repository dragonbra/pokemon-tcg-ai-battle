from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


CHECKPOINTS = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.training.checkpoints"
)


class CheckpointContractTests(unittest.TestCase):
    def test_distinct_epoch_checkpoints_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = nn.Linear(3, 2)
            records = []
            for epoch in (1, 2):
                records.append(
                    CHECKPOINTS.save_epoch_checkpoint(
                        root,
                        epoch=epoch,
                        model=model,
                        metadata={"arm": "semantic", "epoch": epoch},
                    )
                )

            self.assertEqual(
                [record["path"] for record in records],
                ["epoch_0001.pt", "epoch_0002.pt"],
            )
            for epoch, record in enumerate(records, start=1):
                payload = torch.load(root / record["path"], weights_only=True)
                self.assertEqual(payload["metadata"]["epoch"], epoch)
                self.assertEqual(
                    set(payload), {"schema_version", "state_dict", "metadata"}
                )
                self.assertFalse(record["optimizer_state_saved"])
                self.assertFalse(record["resumable_training_state_saved"])

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                CHECKPOINTS.save_epoch_checkpoint(
                    root,
                    epoch=2,
                    model=model,
                    metadata={"arm": "semantic", "epoch": 2},
                )

    def test_epoch_checkpoint_name_rejects_non_positive_epoch(self) -> None:
        for epoch in (0, -1):
            with self.subTest(epoch=epoch):
                with self.assertRaisesRegex(ValueError, "positive"):
                    CHECKPOINTS.epoch_checkpoint_name(epoch)

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

    def test_compiled_wrapper_prefix_is_not_persisted(self) -> None:
        class Wrapper(nn.Module):
            def __init__(self, original: nn.Module):
                super().__init__()
                self._orig_mod = original

        with tempfile.TemporaryDirectory() as directory:
            original = nn.Linear(3, 2)
            record = CHECKPOINTS.save_checkpoint(
                Path(directory),
                name="latest",
                model=Wrapper(original),
                metadata={},
            )
            payload = torch.load(Path(directory) / record["path"], weights_only=True)
            self.assertEqual(set(payload["state_dict"]), set(original.state_dict()))
            self.assertFalse(
                any(key.startswith("_orig_mod.") for key in payload["state_dict"])
            )

    def test_exact_training_state_restores_optimizer_model_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss = model(torch.ones(2, 3)).square().mean()
            loss.backward()
            optimizer.step()
            compatibility = {
                "dataset_manifest_sha256": "a" * 64,
                "training_config_sha256": "b" * 64,
                "model_contract_sha256": "c" * 64,
            }
            expected_model = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            expected_optimizer = optimizer.state_dict()
            np.random.seed(123)
            torch.manual_seed(123)
            record = CHECKPOINTS.save_training_state(
                root,
                models={"semantic": model},
                optimizers={"semantic": optimizer},
                trainer_state={"epoch": 2, "global_step": 17},
                compatibility=compatibility,
            )
            expected_numpy = np.random.random(4)
            expected_random = torch.rand(4)
            with torch.no_grad():
                model.weight.zero_()
            optimizer.param_groups[0]["lr"] = 0.5
            restored = CHECKPOINTS.load_training_state(
                root / record["path"],
                models={"semantic": model},
                optimizers={"semantic": optimizer},
                expected_compatibility=compatibility,
            )
            self.assertEqual(restored, {"epoch": 2, "global_step": 17})
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_model[key]))
            self.assertEqual(
                optimizer.state_dict()["param_groups"],
                expected_optimizer["param_groups"],
            )
            np.testing.assert_array_equal(np.random.random(4), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(4), expected_random))

    def test_exact_training_state_rejects_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            record = CHECKPOINTS.save_training_state(
                root,
                models={"semantic": model},
                optimizers={"semantic": optimizer},
                trainer_state={"epoch": 0},
                compatibility={"dataset": "a"},
            )
            with self.assertRaisesRegex(ValueError, "compatibility mismatch"):
                CHECKPOINTS.load_training_state(
                    root / record["path"],
                    models={"semantic": model},
                    optimizers={"semantic": optimizer},
                    expected_compatibility={"dataset": "b"},
                )

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

from __future__ import annotations

from importlib import import_module
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


CHECKPOINTS = import_module(
    "train.0021_persona_free_universal_bc.training.checkpoints"
)


class CheckpointTests(unittest.TestCase):
    def _metadata(self) -> dict[str, str]:
        return {
            "project_id": "0021_persona_free_universal_bc",
            "version": "V1_corrected_persona_free_r15",
            "dataset_content_sha256": "dataset",
            "feature_compiler_sha256": "compiler",
            "model_config_sha256": "model",
            "training_config_sha256": "training",
        }

    def _populated_optimizer(
        self, model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return optimizer

    def test_checkpoint_restores_optimizer_trainer_and_rng(self) -> None:
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        model = torch.nn.Linear(3, 2)
        optimizer = self._populated_optimizer(model)
        with tempfile.TemporaryDirectory() as directory:
            manifest = CHECKPOINTS.save_checkpoint(
                Path(directory),
                model=model,
                optimizer=optimizer,
                scheduler=None,
                grad_scaler=None,
                completed_epoch=1,
                global_step=2,
                trainer_state={
                    "best": {"loss": 0.5},
                    "no_loss_improvement": 1,
                    "progress_iteration": 9,
                    "gpu_training_seconds": 123.5,
                    "history": [{"trainer/epoch": 1}],
                },
                metadata=self._metadata(),
                criteria=("latest",),
            )
            checkpoint = Path(directory) / manifest["path"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(3)
            for parameter in model.parameters():
                parameter.data.zero_()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
            restored = CHECKPOINTS.load_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                grad_scaler=None,
                expected_metadata=self._metadata(),
            )
            actual_python = random.random()
            actual_numpy = float(np.random.random())
            actual_torch = torch.rand(3)
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "model",
                "optimizer",
                "scheduler",
                "grad_scaler",
                "trainer",
                "rng",
                "metadata",
            },
        )
        self.assertTrue(manifest["optimizer_state_saved"])
        self.assertTrue(manifest["resumable_training_state_saved"])
        self.assertEqual(restored.completed_epoch, 1)
        self.assertEqual(restored.global_step, 2)
        self.assertEqual(restored.best, {"loss": 0.5})
        self.assertEqual(restored.no_loss_improvement, 1)
        self.assertEqual(restored.progress_iteration, 9)
        self.assertEqual(restored.gpu_training_seconds, 123.5)
        self.assertEqual(expected_python, actual_python)
        self.assertEqual(expected_numpy, actual_numpy)
        torch.testing.assert_close(expected_torch, actual_torch)

    def test_metadata_mismatch_fails_before_mutating_model(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = self._populated_optimizer(model)
        with tempfile.TemporaryDirectory() as directory:
            manifest = CHECKPOINTS.save_checkpoint(
                Path(directory),
                model=model,
                optimizer=optimizer,
                scheduler=None,
                grad_scaler=None,
                completed_epoch=1,
                global_step=2,
                trainer_state={
                    "best": {},
                    "no_loss_improvement": 0,
                    "progress_iteration": 2,
                    "gpu_training_seconds": 10.0,
                    "history": [],
                },
                metadata=self._metadata(),
                criteria=("latest",),
            )
            checkpoint = Path(directory) / manifest["path"]
            before = {name: value.clone() for name, value in model.state_dict().items()}
            wrong = {**self._metadata(), "dataset_content_sha256": "wrong"}
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                CHECKPOINTS.load_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    grad_scaler=None,
                    expected_metadata=wrong,
                )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_retention_keeps_only_referenced_models(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = CHECKPOINTS.save_checkpoint(
                root,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                grad_scaler=None,
                completed_epoch=1,
                global_step=1,
                trainer_state={
                    "best": {}, "no_loss_improvement": 0,
                    "progress_iteration": 1, "gpu_training_seconds": 1.0,
                    "history": [],
                },
                metadata=self._metadata(),
                criteria=("latest", "best_validation_loss"),
            )
            second = CHECKPOINTS.save_checkpoint(
                root,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                grad_scaler=None,
                completed_epoch=2,
                global_step=2,
                trainer_state={
                    "best": {}, "no_loss_improvement": 0,
                    "progress_iteration": 2, "gpu_training_seconds": 2.0,
                    "history": [],
                },
                metadata=self._metadata(),
                criteria=("latest",),
            )
            third = CHECKPOINTS.save_checkpoint(
                root,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                grad_scaler=None,
                completed_epoch=3,
                global_step=3,
                trainer_state={
                    "best": {}, "no_loss_improvement": 0,
                    "progress_iteration": 3, "gpu_training_seconds": 3.0,
                    "history": [],
                },
                metadata=self._metadata(),
                criteria=("latest",),
            )
            retained = {path.name for path in root.glob("epoch-*.pt")}
        self.assertEqual(retained, {first["path"], third["path"]})
        self.assertNotIn(second["path"], retained)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

import torch


CHECKPOINTS = importlib.import_module(
    "train.0016_alakazam_multideck_bc.training.checkpoints"
)
RESUME = importlib.import_module("train.0016_alakazam_multideck_bc.training.resume")


class ResumeContractTest(unittest.TestCase):
    def test_checkpoint_captures_rng_and_resume_ignores_partial_next_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "checkpoint"
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            full = {
                "trainer/epoch": 1,
                "trainer/update": 7,
                "bc/validation/loss": 0.4,
                "bc/validation/teacher_exact_action": 0.6,
                "bc/validation/exact_action": 0.5,
            }
            manifest = CHECKPOINTS.save_checkpoint(
                checkpoint_root,
                model=model,
                optimizer=optimizer,
                epoch=1,
                global_step=7,
                metadata={
                    "version": "V1_resume_test",
                    "dataset_content_sha256": "dataset",
                    "metrics": full,
                },
                criteria=["latest"],
            )
            metrics = root / "training_metrics.jsonl"
            metrics.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"trainer/epoch": 1, "progress/iteration": 8},
                        full,
                        {
                            "trainer/epoch": 2,
                            "trainer/update": 9,
                            "progress/iteration": 10,
                        },
                    )
                )
                + "\n"
            )
            checkpoint = checkpoint_root / manifest["path"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("rng_state", payload)
            context = RESUME.load_resume_context(
                checkpoint,
                checkpoint_root=checkpoint_root,
                metrics_path=metrics,
                expected_version="V1_resume_test",
                expected_dataset_sha256="dataset",
                early_stopping_min_delta=0.001,
            )
            self.assertEqual(context.start_epoch, 1)
            self.assertEqual(context.global_step, 7)
            self.assertEqual(context.progress_iteration, 8)
            self.assertEqual(context.best["loss"], 0.4)
            self.assertEqual(len(context.history), 1)


if __name__ == "__main__":
    unittest.main()

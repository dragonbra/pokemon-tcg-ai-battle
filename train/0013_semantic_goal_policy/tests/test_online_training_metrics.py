from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rl_environment.runs import VersionPaths

trainer = importlib.import_module(
    "train.0013_semantic_goal_policy.training.trainer"
)
runtime_batch = importlib.import_module(
    "train.0013_semantic_goal_policy.training.runtime_batch"
)


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.class_logits = torch.nn.Parameter(torch.tensor([0.25, -0.25]))
        self.teacher_calls = 0

    def teacher_logits(
        self,
        batch: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        del batch
        self.teacher_calls += 1
        return self.class_logits.view(1, 1, 2).expand(
            targets.size(0), targets.size(1), -1
        )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "state_num": torch.zeros(2, 1),
        "targets": torch.tensor([[0], [1]]),
        "target_mask": torch.ones(2, 1, dtype=torch.bool),
    }


def _paths(root: Path) -> VersionPaths:
    artifact = root / "artifact"
    checkpoints = root / "checkpoint"
    tensorboard = root / "tensorboard"
    wandb = root / "wandb"
    for path in (artifact, checkpoints, tensorboard, wandb):
        path.mkdir(parents=True)
    (artifact / "status.json").write_text(
        json.dumps({"state": "allocated"}) + "\n",
        encoding="utf-8",
    )
    return VersionPaths(
        project_id="0013_semantic_goal_policy",
        version_name="V99_test",
        project_archive=root / "archive",
        run_root=root,
        artifact=artifact,
        checkpoints=checkpoints,
        tensorboard=tensorboard,
        wandb=wandb,
        evaluation=root / "evaluation.html",
        config=artifact / "training_config.json",
        metrics=artifact / "training_metrics.jsonl",
        summary=artifact / "training_summary.json",
        status=artifact / "status.json",
        checkpoint_selection=artifact / "checkpoint_selection.json",
        model_contract=artifact / "model_contract.json",
        dataset_reference=artifact / "dataset_reference.json",
        metrics_snapshot=artifact / "metrics_snapshot.json",
        wandb_snapshot_manifest=artifact / "wandb_snapshot_manifest.json",
    )


class _Logger:
    records: list[dict[str, float | int]] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        type(self).records = []

    def __enter__(self) -> "_Logger":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def log(
        self,
        _step: int,
        metrics: dict[str, float | int],
    ) -> dict[str, float | int]:
        type(self).records.append(metrics)
        return metrics


class OnlineTrainingMetricsTest(unittest.TestCase):
    def test_target_padding_is_trimmed_without_changing_teacher_loss(self) -> None:
        values = _batch()
        values["targets"] = torch.tensor(
            [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
        )
        values["target_mask"] = torch.tensor(
            [
                [True, True, False, False, False],
                [True, False, False, False, False],
            ]
        )
        compacted = runtime_batch.trim_target_padding(values)
        self.assertEqual(compacted["targets"].shape, (2, 2))
        self.assertEqual(compacted["target_mask"].shape, (2, 2))
        model = _TinyPolicy().eval()
        original = trainer.teacher_forced_batch_metrics(model, values).loss
        trimmed = trainer.teacher_forced_batch_metrics(model, compacted).loss
        self.assertTrue(torch.equal(original, trimmed))

    def test_train_is_not_replayed_for_static_evaluation(self) -> None:
        model = _TinyPolicy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_epochs: list[int] = []
        validation_epochs: list[int] = []

        def train_batches(epoch: int):
            train_epochs.append(epoch)
            return [_batch(), _batch()]

        def validation_batches(epoch: int):
            validation_epochs.append(epoch)
            return [_batch()]

        def validation_metrics(_model, batches, **kwargs):
            self.assertEqual(kwargs["namespace"], "validation")
            self.assertEqual(len(list(batches)), 1)
            kwargs["progress"](1, 2)
            return {
                "bc/validation/loss": 0.5,
                "bc/validation/exact_action": 0.75,
            }

        with tempfile.TemporaryDirectory() as directory:
            paths = _paths(Path(directory))
            with (
                mock.patch.object(trainer, "TrainingLogger", _Logger),
                mock.patch.object(
                    trainer,
                    "evaluate_full_pass",
                    side_effect=validation_metrics,
                ) as evaluate,
                mock.patch.object(
                    trainer,
                    "save_checkpoint",
                    return_value={"path": "checkpoint.pt", "sha256": "a" * 64},
                ),
            ):
                summary = trainer.train_version(
                    paths,
                    model=model,
                    optimizer=optimizer,
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    batch_counts={"train": 2, "validation": 1},
                    epochs=2,
                    config={"metric_contract": "test"},
                    device=torch.device("cpu"),
                    amp=False,
                    progress_log_every=2,
                )

        self.assertEqual(train_epochs, [1, 2])
        self.assertEqual(validation_epochs, [1, 2])
        self.assertEqual(model.teacher_calls, 4)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(summary["global_step"], 4)
        for epoch in summary["history"]:
            self.assertIn("bc/optimization/token_accuracy", epoch)
            self.assertIn("bc/optimization/teacher_exact_action", epoch)
            self.assertEqual(epoch["bc/optimization/decisions"], 4.0)
            self.assertEqual(epoch["bc/optimization/tokens"], 4.0)
            self.assertGreaterEqual(epoch["bc/optimization/token_accuracy"], 0.0)
            self.assertLessEqual(epoch["bc/optimization/token_accuracy"], 1.0)
            self.assertNotIn("bc/train/loss", epoch)
            self.assertIn("bc/validation/exact_action", epoch)


if __name__ == "__main__":
    unittest.main()

"""Calibrate a checkpoint's value head while preserving its policy weights."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rl.core.batch import collate_encoded
from rl.core.checkpoint import CheckpointManager
from rl.core.logging import TrainingLogger
from rl.core.losses import value_huber_loss
from rl.core.model import CandidatePolicyValueNet, ModelConfig
from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .dataset import load_behavior_cloning_dataset


MODEL_INPUT_KEYS = (
    "state_numeric",
    "state_card_ids",
    "action_type_ids",
    "action_card_ids",
    "action_target_ids",
    "action_numeric",
    "action_mask",
)


def _batch(records: list[dict[str, Any]]) -> dict[str, Tensor]:
    batch = collate_encoded([record["encoded"] for record in records])
    batch["terminal_outcome"] = torch.tensor(
        [float(record.get("terminal_outcome", 0.0)) for record in records],
        dtype=torch.float32,
    )
    return batch


def _inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: batch[key] for key in MODEL_INPUT_KEYS}


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _load_model(checkpoint: Path, device: torch.device) -> CandidatePolicyValueNet:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = payload.get("metadata") or {}
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint metadata.model_config is required")
    model = CandidatePolicyValueNet(ModelConfig(**model_config)).to(device)
    model.load_state_dict(payload["model"])
    return model


def _validate_shapes(model: CandidatePolicyValueNet, records: list[dict[str, Any]]) -> None:
    encoded = records[0]["encoded"]
    expected = model.config
    if len(encoded["state_numeric"]) != expected.state_numeric_dim:
        raise ValueError("dataset state_numeric width does not match checkpoint")
    if len(encoded["state_card_ids"]) != expected.state_token_count:
        raise ValueError("dataset state token width does not match checkpoint")
    if len(encoded["action_mask"]) != expected.max_candidates:
        raise ValueError("dataset candidate width does not match checkpoint")


@torch.no_grad()
def _evaluate(
    model: CandidatePolicyValueNet,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    for start in range(0, len(records), 512):
        batch = _move(_batch(records[start : start + 512]), device)
        value, _logits = model(**_inputs(batch))
        predictions.append(value.detach().cpu())
        targets.append(batch["terminal_outcome"].detach().cpu())
    predicted = torch.cat(predictions)
    target = torch.cat(targets)
    error = predicted - target
    if predicted.numel() > 1 and predicted.std() > 0 and target.std() > 0:
        correlation = float(torch.corrcoef(torch.stack([predicted, target]))[0, 1].item())
    else:
        correlation = 0.0
    win_mask = target > 0.5
    loss_mask = target < -0.5
    draw_mask = ~(win_mask | loss_mask)
    return {
        "value_mae": float(error.abs().mean().item()),
        "value_mse": float(error.square().mean().item()),
        "value_correlation": correlation,
        "value_mean": float(predicted.mean().item()),
        "value_win_mean": float(predicted[win_mask].mean().item()) if win_mask.any() else 0.0,
        "value_loss_mean": float(predicted[loss_mask].mean().item())
        if loss_mask.any()
        else 0.0,
        "value_draw_mean": float(predicted[draw_mask].mean().item())
        if draw_mask.any()
        else 0.0,
    }


def train(
    dataset_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    seed: int = 7,
    device_name: str = "auto",
    storage_path: Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, float | int | str]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    storage = assert_storage_safe(storage_path, min_free_gib)
    torch.manual_seed(seed)
    random.seed(seed)
    device = _resolve_device(device_name)
    records = load_behavior_cloning_dataset(dataset_path)
    model = _load_model(checkpoint, device)
    _validate_shapes(model, records)

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(len(shuffled) * validation_fraction)
    if len(shuffled) > 1 and validation_fraction > 0:
        validation_count = max(1, validation_count)
    validation = shuffled[:validation_count]
    training = shuffled[validation_count:] or shuffled

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.value_head.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=learning_rate)
    manager = CheckpointManager(output_dir / "checkpoints")
    best_score = float("inf")
    last_metrics: dict[str, float] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    with TrainingLogger(output_dir / "metrics.jsonl", output_dir / "tensorboard") as logger:
        for epoch in range(1, epochs + 1):
            model.eval()
            model.value_head.train()
            order = list(range(len(training)))
            random.Random(seed + epoch).shuffle(order)
            losses: list[float] = []
            for start in range(0, len(order), batch_size):
                batch = _move(
                    _batch([training[index] for index in order[start : start + batch_size]]),
                    device,
                )
                optimizer.zero_grad()
                value, _logits = model(**_inputs(batch))
                loss = value_huber_loss(value, batch["terminal_outcome"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))

            train_metrics = _evaluate(model, training, device)
            validation_metrics = _evaluate(model, validation or training, device)
            last_metrics = {
                "train/value_loss": sum(losses) / max(1, len(losses)),
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation/{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            logger.log(epoch, last_metrics)
            metadata = {
                "task": "ptcg_value_head_calibration_v1",
                "dataset": str(dataset_path.resolve()),
                "source_checkpoint": str(checkpoint.resolve()),
                "model_config": model.config.to_dict(),
                "seed": seed,
                "device": str(device),
                "freeze_policy": True,
                "trainable_parameter_prefix": "value_head.",
                "learning_rate": learning_rate,
                "storage_path": storage.path,
                "storage_free_gib": round(storage.free_gib, 2),
                "epoch_metrics": last_metrics,
            }
            manager.save("latest", model, optimizer=optimizer, step=epoch, metadata=metadata)
            score = validation_metrics["value_mae"]
            if score < best_score:
                best_score = score
                manager.save(
                    "best_validation",
                    model,
                    optimizer=optimizer,
                    step=epoch,
                    metadata=metadata,
                )

    return {
        "records": len(records),
        "training_records": len(training),
        "validation_records": len(validation),
        "device": str(device),
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
        "best_validation_value_mae": best_score,
        **last_metrics,
        "checkpoint": str((output_dir / "checkpoints" / "best_validation.pt").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.dataset,
                args.checkpoint,
                args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                device_name=args.device,
                storage_path=args.storage_path,
                min_free_gib=args.min_free_gib,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

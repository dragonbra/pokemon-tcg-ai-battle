from __future__ import annotations

from dataclasses import dataclass

import torch

from ..policy.actor_critic import DragapultActorCritic
from ..policy.batching import collate_feature_batches, move_batch
from .batch import PreparedBatch


@dataclass(frozen=True)
class ValueConfig:
    epochs: int = 4
    batch_size: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5


def _explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> float:
    variance = target.var(unbiased=False)
    if float(variance) <= 1e-12:
        return 0.0
    return float(1.0 - (target - prediction).var(unbiased=False) / variance)


def _group_metrics(
    result: dict[str, float],
    prefix: str,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    count = int(mask.sum())
    result[f"{prefix}/decisions"] = float(count)
    if count == 0:
        return
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    error = selected_prediction - selected_target
    result[f"{prefix}/mae"] = float(error.abs().mean())
    result[f"{prefix}/rmse"] = float(error.square().mean().sqrt())
    result[f"{prefix}/prediction_mean"] = float(selected_prediction.mean())


def calibrate_value(
    model: DragapultActorCritic,
    batch: PreparedBatch,
    *,
    device: torch.device,
    config: ValueConfig = ValueConfig(),
) -> list[dict[str, float]]:
    model.freeze_actor()
    optimizer = torch.optim.AdamW(
        model.value_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        # The frozen actor remains deterministic; only the value head receives gradients.
        model.eval()
        order = torch.randperm(batch.decisions)
        loss_sum = 0.0
        weight_sum = 0.0
        gradient_norms: list[float] = []
        for start in range(0, batch.decisions, config.batch_size):
            indices = order[start : start + config.batch_size]
            features = move_batch(
                collate_feature_batches([batch.features[int(i)] for i in indices]),
                device,
            )
            target = batch.terminal_return[indices].to(device)
            weights = batch.episode_weight[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.value(features)
            loss = ((prediction - target).square() * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite value calibration loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.value_head.parameters(), config.max_grad_norm
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite value calibration gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * float(weights.sum())
            weight_sum += float(weights.sum())
            gradient_norms.append(float(norm))

        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, batch.decisions, config.batch_size):
                indices = torch.arange(start, min(start + config.batch_size, batch.decisions))
                features = move_batch(
                    collate_feature_batches([batch.features[int(i)] for i in indices]),
                    device,
                )
                predictions.append(model.value(features).cpu())
        prediction = torch.cat(predictions)
        target = batch.terminal_return
        error = prediction - target
        metrics = {
                "trainer/epoch": float(epoch),
                "value/loss": loss_sum / max(weight_sum, 1e-12),
                "value/mae": float(error.abs().mean()),
                "value/rmse": float(error.square().mean().sqrt()),
                "value/brier": float(error.square().mean()),
                "value/explained_variance": _explained_variance(prediction, target),
                "value/prediction_mean": float(prediction.mean()),
                "value/prediction_std": float(prediction.std(unbiased=False)),
                "value/gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
                "value/decisions": float(batch.decisions),
        }
        _group_metrics(
            metrics,
            "value/calibration/phase/early",
            batch.turns <= 4,
            prediction,
            target,
        )
        _group_metrics(
            metrics,
            "value/calibration/phase/mid",
            (batch.turns > 4) & (batch.turns <= 10),
            prediction,
            target,
        )
        _group_metrics(
            metrics,
            "value/calibration/phase/late",
            batch.turns > 10,
            prediction,
            target,
        )
        for first, name in ((True, "first"), (False, "second")):
            _group_metrics(
                metrics,
                f"value/calibration/seat/{name}",
                batch.candidate_first == first,
                prediction,
                target,
            )
        for outcome, name in ((1.0, "win"), (0.0, "draw"), (-1.0, "loss")):
            _group_metrics(
                metrics,
                f"value/calibration/outcome/{name}",
                target == outcome,
                prediction,
                target,
            )
        for opponent in sorted(set(batch.opponents)):
            mask = torch.tensor(
                [value == opponent for value in batch.opponents], dtype=torch.bool
            )
            _group_metrics(
                metrics,
                f"value/calibration/opponent/{opponent}",
                mask,
                prediction,
                target,
            )
        history.append(metrics)
    return history


__all__ = ["ValueConfig", "calibrate_value"]

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rl_environment.checkpoint import CheckpointManager
from rl_environment.logging import TrainingLogger
from rl_environment.model import ModelConfig
from rl_environment.runs import training_paths
from rl_environment.storage import assert_storage_safe
from train.alakazam_bc_rl.full_action_model import FullActionPolicyValueNet

from .card_categories import CATEGORY_NAMES, load_card_category_lookup
from .config import ExperimentConfig, LossConfig, load_config
from .data import collate_records, load_full_action_dataset, model_inputs, split_records
from .model import CategoryAugmentedFullActionPolicyValueNet
from .objective import combined_loss, per_sample_full_action_losses
from .rewards import compose_rewards, imitation_weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def _load_source(
    checkpoint: Path, config: ExperimentConfig, device: torch.device
) -> tuple[FullActionPolicyValueNet, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("source checkpoint must contain a model state dict")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict) or not isinstance(metadata.get("model_config"), dict):
        raise ValueError("source checkpoint metadata.model_config is required")
    model_config = ModelConfig(**metadata["model_config"])
    maximum = int(metadata.get("max_selection_count", model_config.max_candidates))
    category_config = config.card_category_embedding
    output_metadata = dict(metadata)
    if category_config.enabled:
        lookup, counts = load_card_category_lookup(
            category_config.source_csv,
            card_vocab_size=model_config.card_vocab_size,
        )
        model = CategoryAugmentedFullActionPolicyValueNet(
            model_config,
            card_category_lookup=lookup,
            category_vocab_size=len(CATEGORY_NAMES),
            category_scale=category_config.scale,
            max_selection_count=maximum,
        ).to(device)
        incompatible = model.load_state_dict(payload["model"], strict=False)
        allowed_missing = {"card_category_lookup", "card_category_embedding.weight"}
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "source checkpoint is incompatible with category augmentation: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        category_source = Path(category_config.source_csv)
        output_metadata["card_category_embedding"] = {
            "enabled": True,
            "categories": list(CATEGORY_NAMES),
            "category_vocab_size": len(CATEGORY_NAMES),
            "scale": category_config.scale,
            "source_csv": str(category_source.resolve()),
            "source_csv_sha256": _sha256(category_source),
            "card_counts": counts,
        }
    else:
        model = FullActionPolicyValueNet(model_config, max_selection_count=maximum).to(device)
        model.load_state_dict(payload["model"])
        output_metadata["card_category_embedding"] = {"enabled": False}
    return model, output_metadata


def _reset_value_output(model: FullActionPolicyValueNet) -> None:
    output = model.value_head[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("value head must end in a Linear layer")
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)


def _value_only(model: FullActionPolicyValueNet, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not enabled
    if enabled:
        for parameter in model.value_head.parameters():
            parameter.requires_grad = True


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _batch_objective(
    model: FullActionPolicyValueNet,
    records: list[dict[str, Any]],
    config: ExperimentConfig,
    device: torch.device,
    dataset_reward_mean: float,
    *,
    value_only: bool,
) -> tuple[Tensor, dict[str, float]]:
    batch = _move(collate_records(records), device)
    rewards, components = compose_rewards(records, config.reward)
    rewards = rewards.to(device)
    value, logits, count_logits = model.forward_with_count(**model_inputs(batch))
    policy_losses, count_losses = per_sample_full_action_losses(
        logits,
        count_logits,
        target_mask=batch["target_mask"],
        target_count=batch["target_count"],
        action_mask=batch["action_mask"],
        selection_min_count=batch["selection_min_count"],
        selection_max_count=batch["selection_max_count"],
    )
    advantages, weights = imitation_weights(
        rewards,
        config.weighting,
        dataset_mean=dataset_reward_mean,
        values=value,
    )
    coefficients = (
        LossConfig(policy=0.0, count=0.0, value=config.loss.value)
        if value_only
        else config.loss
    )
    total, parts = combined_loss(
        policy_losses,
        count_losses,
        value,
        rewards,
        weights,
        coefficients,
    )
    metrics = {
        "loss": float(total.detach().item()),
        "policy_loss": float(parts["policy"].detach().item()),
        "count_loss": float(parts["count"].detach().item()),
        "value_loss": float(parts["value"].detach().item()),
        "reward_mean": float(rewards.mean().item()),
        "advantage_mean": float(advantages.mean().item()),
        "weight_mean": float(weights.mean().item()),
        "weight_min": float(weights.min().item()),
        "weight_max": float(weights.max().item()),
        "weight_square_mean": float(weights.square().mean().item()),
    }
    for name, values in components.items():
        metrics[f"reward_component/{name}"] = float(values.mean().item())
    return total, metrics


@torch.no_grad()
def _evaluate(
    model: FullActionPolicyValueNet,
    records: list[dict[str, Any]],
    config: ExperimentConfig,
    device: torch.device,
    dataset_reward_mean: float,
) -> dict[str, float]:
    if not records:
        return {}
    model.eval()
    totals: dict[str, float] = {}
    samples = 0
    exact = 0
    count_correct = 0
    value_absolute_error = 0.0
    weight_sum = 0.0
    weight_square_sum = 0.0
    for start in range(0, len(records), config.batch_size):
        chunk = records[start : start + config.batch_size]
        batch = _move(collate_records(chunk), device)
        rewards, _ = compose_rewards(chunk, config.reward)
        rewards = rewards.to(device)
        value, logits, count_logits = model.forward_with_count(**model_inputs(batch))
        policy_losses, count_losses = per_sample_full_action_losses(
            logits,
            count_logits,
            target_mask=batch["target_mask"],
            target_count=batch["target_count"],
            action_mask=batch["action_mask"],
            selection_min_count=batch["selection_min_count"],
            selection_max_count=batch["selection_max_count"],
        )
        _, weights = imitation_weights(
            rewards,
            config.weighting,
            dataset_mean=dataset_reward_mean,
            values=value,
        )
        loss, parts = combined_loss(
            policy_losses,
            count_losses,
            value,
            rewards,
            weights,
            config.loss,
        )
        size = len(chunk)
        samples += size
        for name, scalar in {
            "loss": loss,
            "policy_loss": parts["policy"],
            "count_loss": parts["count"],
            "value_loss": parts["value"],
            "reward_mean": rewards.mean(),
            "weight_mean": weights.mean(),
        }.items():
            totals[name] = totals.get(name, 0.0) + float(scalar.item()) * size
        value_absolute_error += float((value - rewards).abs().sum().item())
        weight_sum += float(weights.sum().item())
        weight_square_sum += float(weights.square().sum().item())

        positions = torch.arange(count_logits.shape[1], device=device).unsqueeze(0)
        maximum = batch["selection_max_count"].clamp_max(count_logits.shape[1] - 1)
        valid = (positions >= batch["selection_min_count"].unsqueeze(1)) & (
            positions <= maximum.unsqueeze(1)
        )
        predicted_counts = count_logits.masked_fill(
            ~valid, torch.finfo(count_logits.dtype).min
        ).argmax(dim=1)
        predicted_counts = torch.minimum(predicted_counts, batch["action_mask"].sum(dim=1))
        count_correct += int(predicted_counts.eq(batch["target_count"]).sum().item())
        for row, record in enumerate(chunk):
            count = int(predicted_counts[row].item())
            prediction = set(
                int(index)
                for index in torch.topk(logits[row], k=count).indices.tolist()
            ) if count else set()
            exact += int(prediction == set(int(target) for target in record["targets"]))
    metrics = {name: value / samples for name, value in totals.items()}
    metrics.update(
        {
            "exact_action_rate": exact / samples,
            "selection_count_accuracy": count_correct / samples,
            "value_mae": value_absolute_error / samples,
            "effective_sample_ratio": (
                (weight_sum * weight_sum) / (samples * weight_square_sum)
                if weight_square_sum > 0
                else 0.0
            ),
        }
    )
    return metrics


def train(
    dataset: Path,
    source_checkpoint: Path,
    output: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    config.validate()
    storage = assert_storage_safe(config.storage_path, config.min_free_gib)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(config.device)
    records = load_full_action_dataset(dataset)
    splits = split_records(records)
    train_rewards, _ = compose_rewards(splits["train"], config.reward)
    dataset_reward_mean = float(train_rewards.mean().item()) if train_rewards.numel() else 0.0
    model, source_metadata = _load_source(source_checkpoint, config, device)
    if config.reset_value_output:
        _reset_value_output(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    paths = training_paths(output)
    paths.run.mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(paths.checkpoints)
    resolved_config = {
        **config.to_dict(),
        "dataset": str(dataset.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "dataset_records": {name: len(rows) for name, rows in splits.items()},
        "dataset_reward_mean": dataset_reward_mean,
        "storage": storage.to_dict(),
    }
    paths.config.write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best_score = -math.inf
    best_epoch = 0
    started = time.monotonic()
    last_metrics: dict[str, float] = {}
    with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.monotonic()
            warmup = epoch <= config.value_warmup_epochs
            _value_only(model, warmup and config.freeze_backbone_during_value_warmup)
            model.train()
            order = list(range(len(splits["train"])))
            random.Random(config.seed + epoch).shuffle(order)
            accumulators: dict[str, float] = {}
            seen = 0
            for start in range(0, len(order), config.batch_size):
                indices = order[start : start + config.batch_size]
                chunk = [splits["train"][index] for index in indices]
                optimizer.zero_grad()
                loss, batch_metrics = _batch_objective(
                    model,
                    chunk,
                    config,
                    device,
                    dataset_reward_mean,
                    value_only=warmup,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    config.max_grad_norm,
                )
                optimizer.step()
                seen += len(chunk)
                for name, value in batch_metrics.items():
                    accumulators[name] = accumulators.get(name, 0.0) + value * len(chunk)
            train_metrics = {name: value / seen for name, value in accumulators.items()}
            validation = _evaluate(
                model,
                splits["validation"],
                config,
                device,
                dataset_reward_mean,
            )
            weight_square_mean = train_metrics.pop("weight_square_mean", 0.0)
            weight_mean = train_metrics.get("weight_mean", 0.0)
            train_metrics["effective_sample_ratio"] = (
                weight_mean * weight_mean / weight_square_mean if weight_square_mean > 0 else 0.0
            )
            last_metrics = {
                "phase/value_warmup": float(warmup),
                **{f"train/{name}": value for name, value in train_metrics.items()},
                **{f"validation/{name}": value for name, value in validation.items()},
                "runtime/epoch_seconds": time.monotonic() - epoch_started,
            }
            logger.log(epoch, last_metrics)
            print(
                json.dumps(
                    {
                        "event": "reward_weighted_bc_epoch",
                        "epoch": epoch,
                        "warmup": warmup,
                        "train_loss": train_metrics["loss"],
                        "validation_exact_action_rate": validation["exact_action_rate"],
                        "validation_value_mae": validation["value_mae"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            metadata = {
                **source_metadata,
                "task": "ptcg_full_action_reward_weighted_behavior_cloning",
                "source_checkpoint": str(source_checkpoint.resolve()),
                "source_checkpoint_sha256": resolved_config["source_checkpoint_sha256"],
                "stage_two_config": config.to_dict(),
                "dataset": str(dataset.resolve()),
                "epoch_metrics": last_metrics,
            }
            manager.save("latest", model, optimizer=optimizer, step=epoch, metadata=metadata)
            score = validation["exact_action_rate"]
            if not warmup and score > best_score:
                best_score = score
                best_epoch = epoch
                manager.save(
                    "best_validation", model, optimizer=optimizer, step=epoch, metadata=metadata
                )
    _value_only(model, False)
    if best_epoch == 0:
        raise RuntimeError("no post-warm-up epoch was available for checkpoint selection")
    manager.load(paths.checkpoints / "best_validation.pt", model, map_location=device)
    test_metrics = (
        _evaluate(model, splits["test"], config, device, dataset_reward_mean)
        if config.evaluate_test and splits["test"]
        else {}
    )
    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_validation_exact_action_rate": best_score,
        "test_evaluated": bool(test_metrics),
        "final_test": test_metrics,
        "last_epoch": last_metrics,
        "checkpoint": str((paths.checkpoints / "best_validation.pt").resolve()),
        "runtime_seconds": time.monotonic() - started,
        "config": resolved_config,
    }
    paths.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train stage-two reward-weighted Alakazam BC")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = train(
        args.dataset,
        args.source_checkpoint,
        args.output,
        load_config(args.config),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()

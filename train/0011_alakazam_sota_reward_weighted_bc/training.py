from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rl_environment.checkpoint import CheckpointManager
from rl_environment.logging import TrainingLogger
from rl_environment.runs import training_paths
from rl_environment.storage import assert_storage_safe
_base_dataset = importlib.import_module("train.0010_alakazam_sota_model.dataset")
_base_model = importlib.import_module("train.0010_alakazam_sota_model.model")
JsonlShardDataset = _base_dataset.JsonlShardDataset
IDOnlyPointerPolicy = _base_model.IDOnlyPointerPolicy
collate_id_only = _base_model.collate_id_only

from .config import ExperimentConfig
from .dataset import dataset_paths, load_dataset_audit
from .rewards import compose_reward, compose_rewards, imitation_weights
from .rewards import weighted_token_cross_entropy


MODEL_VERSION = "alakazam_sota_reward_weighted_pointer_bc_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _collate(rows: list[dict[str, Any]], config: ExperimentConfig) -> dict[str, Tensor]:
    batch = collate_id_only(rows)
    rewards, components = compose_rewards(rows, config.reward)
    batch["rewards"] = rewards
    for name, values in components.items():
        batch[f"reward_component/{name}"] = values
    return batch


def _loader(
    path: Path,
    *,
    config: ExperimentConfig,
    shuffle: bool,
) -> tuple[JsonlShardDataset, DataLoader[dict[str, Tensor]]]:
    dataset = JsonlShardDataset(
        [path],
        shuffle=shuffle,
        seed=config.seed,
        buffer_size=config.shuffle_buffer_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=lambda rows: _collate(rows, config),
        num_workers=0,
        pin_memory=config.device != "cpu",
    )
    return dataset, loader


def _reward_distribution(path: Path, config: ExperimentConfig) -> dict[str, Any]:
    values: list[float] = []
    component_sums = {
        component.name: 0.0 for component in config.reward.components if component.enabled
    }
    component_nonzero = {name: 0 for name in component_sums}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            reward, components = compose_reward(row, config.reward)
            values.append(reward)
            for name, contribution in components.items():
                component_sums[name] += contribution
                component_nonzero[name] += int(contribution != 0)
    if not values:
        raise ValueError("training shard contains no rewards")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    mean = statistics.fmean(values)
    return {
        "records": len(values),
        "mean": mean,
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "p05": percentile(0.05),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(values),
        "component_means": {
            name: total / len(values) for name, total in component_sums.items()
        },
        "component_nonzero_records": component_nonzero,
    }


def evaluate(
    model: IDOnlyPointerPolicy,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_exact = 0
    total_rows = 0
    with torch.inference_mode():
        for batch in loader:
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model.teacher_logits(batch)
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    batch["targets"].flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
            valid = batch["targets"].ne(-100)
            prediction = logits.argmax(dim=-1)
            total_loss += float(loss.item())
            total_correct += int(((prediction == batch["targets"]) & valid).sum().item())
            total_tokens += int(valid.sum().item())
            total_exact += int(((prediction == batch["targets"]) | ~valid).all(dim=1).sum())
            total_rows += int(valid.size(0))
    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "exact_action_accuracy": total_exact / max(total_rows, 1),
        "decisions": float(total_rows),
    }


def train(dataset_root: Path, output: Path, config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    paths = training_paths(output)
    storage = assert_storage_safe(config.storage_path, config.min_free_gib)
    audit = load_dataset_audit(dataset_root)
    inputs = dataset_paths(dataset_root)
    counts = audit.get("records_by_split") or {}
    for split in ("train", "validation"):
        if not inputs[split].is_file():
            raise FileNotFoundError(inputs[split])
        if int(counts.get(split, 0)) <= 0:
            raise ValueError(f"dataset audit has no {split} records")

    reward_distribution = _reward_distribution(inputs["train"], config)
    dataset_reward_mean = float(reward_distribution["mean"])
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(config.device)
    model = IDOnlyPointerPolicy(config.model).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_mib = parameter_count * 4 / 1024**2
    if model_mib > config.max_model_mib:
        raise RuntimeError(
            f"model is {model_mib:.2f} MiB, over limit {config.max_model_mib:.2f} MiB"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_dataset, train_loader = _loader(inputs["train"], config=config, shuffle=True)
    _, validation_loader = _loader(inputs["validation"], config=config, shuffle=False)
    _, train_eval_loader = _loader(inputs["train"], config=config, shuffle=False)

    paths.run.mkdir(parents=True, exist_ok=False)
    manager = CheckpointManager(paths.checkpoints)
    resolved_config = {
        **config.to_dict(),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_audit": str((dataset_root / "dataset_audit.json").resolve()),
        "dataset_audit_sha256": _sha256(dataset_root / "dataset_audit.json"),
        "dataset_records": counts,
        "device_resolved": str(device),
        "parameter_count": parameter_count,
        "fp32_model_mib": model_mib,
        "reward_distribution": reward_distribution,
        "storage": storage.to_dict(),
    }
    paths.config.write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best_loss = math.inf
    best_loss_epoch = 0
    best_loss_metrics: dict[str, float] = {}
    best_exact = -math.inf
    best_exact_epoch = 0
    best_exact_metrics: dict[str, float] = {}
    epochs_without_loss_improvement = 0
    stopped_early = False
    last_metrics: dict[str, float] = {}
    started = time.monotonic()
    total_batches = (int(counts["train"]) + config.batch_size - 1) // config.batch_size

    with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.monotonic()
            model.train()
            train_dataset.set_epoch(epoch)
            train_losses: list[float] = []
            train_tokens = 0
            weight_sum = 0.0
            weight_rows = 0
            for batch_index, batch in enumerate(train_loader, 1):
                batch = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                _, weights = imitation_weights(
                    batch["rewards"],
                    config.weighting,
                    dataset_mean=dataset_reward_mean,
                )
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model.teacher_logits(batch)
                    loss = weighted_token_cross_entropy(
                        logits,
                        batch["targets"],
                        weights,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
                train_tokens += int(batch["targets"].ne(-100).sum().item())
                weight_sum += float(weights.sum().item())
                weight_rows += int(weights.numel())
                if (
                    batch_index == 1
                    or batch_index % config.log_every_batches == 0
                    or batch_index == total_batches
                ):
                    recent = train_losses[-config.log_every_batches :]
                    print(
                        json.dumps(
                            {
                                "event": "alakazam_sota_rwb_train_batch",
                                "epoch": epoch,
                                "epochs": config.epochs,
                                "batch": batch_index,
                                "batches": total_batches,
                                "weighted_loss": sum(recent) / len(recent),
                                "mean_weight": weight_sum / max(weight_rows, 1),
                                "tokens": train_tokens,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            validation = evaluate(model, validation_loader, device, amp=amp)
            train_evaluation: dict[str, float] = {}
            if config.train_eval_interval and epoch % config.train_eval_interval == 0:
                train_evaluation = evaluate(model, train_eval_loader, device, amp=amp)
            last_metrics = {
                "runtime/epoch_seconds": time.monotonic() - epoch_started,
                "train/weighted_loss": sum(train_losses) / max(len(train_losses), 1),
                "train/mean_weight": weight_sum / max(weight_rows, 1),
                "train/tokens": float(train_tokens),
                **{f"train_eval/{key}": value for key, value in train_evaluation.items()},
                **{f"validation/{key}": value for key, value in validation.items()},
            }
            logger.log(epoch, last_metrics)
            metadata = {
                "model_version": MODEL_VERSION,
                "model_config": config.model.to_dict(),
                "experiment_config": config.to_dict(),
                "dataset_audit_sha256": resolved_config["dataset_audit_sha256"],
                "reward_distribution": reward_distribution,
                "epoch_metrics": last_metrics,
            }
            manager.save("latest", model, optimizer=optimizer, step=epoch, metadata=metadata)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_loss_epoch = epoch
                best_loss_metrics = dict(validation)
                epochs_without_loss_improvement = 0
                manager.save(
                    "best_validation",
                    model,
                    optimizer=optimizer,
                    step=epoch,
                    metadata=metadata,
                )
            else:
                epochs_without_loss_improvement += 1
            if validation["exact_action_accuracy"] > best_exact:
                best_exact = validation["exact_action_accuracy"]
                best_exact_epoch = epoch
                best_exact_metrics = dict(validation)
                manager.save(
                    "best_exact",
                    model,
                    optimizer=optimizer,
                    step=epoch,
                    metadata=metadata,
                )
            print(
                json.dumps(
                    {
                        "event": "alakazam_sota_rwb_epoch",
                        "epoch": epoch,
                        "validation_loss": validation["loss"],
                        "validation_exact_action_accuracy": validation[
                            "exact_action_accuracy"
                        ],
                        "epochs_without_loss_improvement": (
                            epochs_without_loss_improvement
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                epoch >= config.minimum_epochs_before_stop
                and epochs_without_loss_improvement >= config.early_stopping_patience
            ):
                stopped_early = True
                break

    summary = {
        "status": "completed",
        "best_epoch": best_loss_epoch,
        "best_validation": best_loss_metrics,
        "best_checkpoint": str((paths.checkpoints / "best_validation.pt").resolve()),
        "checkpoint_selection": "minimum unweighted validation token cross-entropy",
        "best_exact_epoch": best_exact_epoch,
        "best_exact_validation": best_exact_metrics,
        "best_exact_checkpoint": str((paths.checkpoints / "best_exact.pt").resolve()),
        "last_epoch": last_metrics,
        "runtime_seconds": time.monotonic() - started,
        "parameter_count": parameter_count,
        "fp32_model_mib": model_mib,
        "stopped_early": stopped_early,
        "minimum_epochs_before_stop": config.minimum_epochs_before_stop,
        "early_stopping_patience": config.early_stopping_patience,
        "epochs_without_loss_improvement": epochs_without_loss_improvement,
        "reward_distribution": reward_distribution,
        "test_evaluated": False,
    }
    paths.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary

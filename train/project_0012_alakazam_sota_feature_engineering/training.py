from __future__ import annotations

import hashlib
import json
import math
import random
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
from .dataset import (
    JsonlShardDataset,
    dataset_paths,
    load_dataset_audit,
)

from .batching import collate_id_only, permute_candidates
from .config import ExperimentConfig
from .model import FeatureEngineeringPolicy


MODEL_VERSION = "alakazam_sota_feature_engineering_bc_v1"
PROJECT_ROOT = Path(__file__).resolve().parent


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
    return dataset, DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_id_only,
        num_workers=0,
        pin_memory=config.device != "cpu",
    )


def evaluate(
    model: FeatureEngineeringPolicy,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    amp: bool,
    permutation_seed: int | None = None,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    tokens = 0
    exact_rows = 0
    rows = 0
    delta_loss_sum = 0.0
    delta_rows = 0
    turn_loss_sum = 0.0
    turn_rows = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if permutation_seed is not None:
                batch = permute_candidates(batch, seed=permutation_seed + batch_index)
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                if model.config.action_transition_auxiliary:
                    logits, predicted_delta, predicted_turn = model.teacher_logits_and_transition(
                        batch
                    )
                else:
                    logits = model.teacher_logits(batch)
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    batch["targets"].flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
                if model.config.action_transition_auxiliary:
                    delta_mask = batch["transition_delta_mask"]
                    next_mask = batch["transition_next_mask"]
                    if delta_mask.any():
                        delta_loss_sum += float(
                            F.mse_loss(
                                predicted_delta[delta_mask],
                                batch["transition_delta"][delta_mask],
                                reduction="sum",
                            ).item()
                        )
                        delta_rows += int(delta_mask.sum()) * predicted_delta.size(1)
                    if next_mask.any():
                        turn_loss_sum += float(
                            F.binary_cross_entropy_with_logits(
                                predicted_turn[next_mask],
                                batch["transition_turn_changed"][next_mask],
                                reduction="sum",
                            ).item()
                        )
                        turn_rows += int(next_mask.sum())
            valid = batch["targets"] != -100
            prediction = logits.argmax(dim=-1)
            loss_sum += float(loss.item())
            correct += int(((prediction == batch["targets"]) & valid).sum().item())
            tokens += int(valid.sum().item())
            exact_rows += int(((prediction == batch["targets"]) | ~valid).all(dim=1).sum())
            rows += int(valid.size(0))
    metrics = {
        "loss": loss_sum / max(tokens, 1),
        "token_accuracy": correct / max(tokens, 1),
        "exact_action_accuracy": exact_rows / max(rows, 1),
        "decisions": float(rows),
    }
    if model.config.action_transition_auxiliary:
        metrics["transition_delta_mse"] = delta_loss_sum / max(delta_rows, 1)
        metrics["transition_turn_bce"] = turn_loss_sum / max(turn_rows, 1)
        metrics["transition_delta_rows"] = float(delta_rows // 11)
        metrics["transition_turn_rows"] = float(turn_rows)
    return metrics


def evaluate_permutation_consistency(
    model: FeatureEngineeringPolicy,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    amp: bool,
    seed: int,
) -> dict[str, float]:
    """Evaluate remapped labels and teacher-forced semantic prediction consistency."""

    model.eval()
    loss_sum = 0.0
    correct = 0
    tokens = 0
    exact_rows = 0
    consistent_rows = 0
    rows = 0
    kl_sum = 0.0
    kl_tokens = 0
    with torch.inference_mode():
        for batch_index, original_cpu in enumerate(loader):
            permuted_cpu = permute_candidates(original_cpu, seed=seed + batch_index)
            original = _move(original_cpu, device)
            permuted = _move(permuted_cpu, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                original_logits = model.teacher_logits(original)
                permuted_logits = model.teacher_logits(permuted)
                loss = F.cross_entropy(
                    permuted_logits.flatten(0, 1),
                    permuted["targets"].flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
            valid = original["targets"] != -100
            perm_prediction = permuted_logits.argmax(dim=-1)
            loss_sum += float(loss.item())
            correct += int(((perm_prediction == permuted["targets"]) & valid).sum())
            tokens += int(valid.sum())
            exact_rows += int(((perm_prediction == permuted["targets"]) | ~valid).all(1).sum())

            option_count = original_logits.size(-1) - 1
            old_to_new = permuted["permutation_old_to_new"]
            aligned = torch.empty_like(permuted_logits)
            gather = old_to_new.unsqueeze(1).expand(-1, permuted_logits.size(1), -1)
            aligned[..., :option_count] = permuted_logits[..., :option_count].gather(2, gather)
            aligned[..., option_count] = permuted_logits[..., option_count]
            original_prediction = original_logits.argmax(dim=-1)
            aligned_prediction = aligned.argmax(dim=-1)
            consistent_rows += int(
                ((original_prediction == aligned_prediction) | ~valid).all(dim=1).sum()
            )
            original_log_prob = F.log_softmax(original_logits.float(), dim=-1)
            aligned_log_prob = F.log_softmax(aligned.float(), dim=-1)
            original_prob = original_log_prob.exp()
            per_token_kl = (original_prob * (original_log_prob - aligned_log_prob)).sum(-1)
            kl_sum += float(per_token_kl[valid].sum())
            kl_tokens += int(valid.sum())
            rows += int(valid.size(0))
    return {
        "loss": loss_sum / max(tokens, 1),
        "token_accuracy": correct / max(tokens, 1),
        "exact_action_accuracy": exact_rows / max(rows, 1),
        "semantic_prediction_consistency": consistent_rows / max(rows, 1),
        "teacher_forced_kl": kl_sum / max(kl_tokens, 1),
        "decisions": float(rows),
    }


def _write_analysis(output: Path, config: ExperimentConfig, summary: dict[str, Any]) -> None:
    best = summary["best_exact_validation"]
    last = summary["last_epoch"]
    permutation = summary.get("best_exact_permuted_validation") or {}
    text = f"""# {output.name} training analysis

## 1. 符合假设的结果

- 训练完成并保留 minimum-loss 与 maximum-exact 两套 checkpoint。
- best validation exact-action: `{best['exact_action_accuracy']:.6f}`（epoch {summary['best_exact_epoch']}）。
- best validation token loss: `{summary['best_validation']['loss']:.6f}`（epoch {summary['best_epoch']}）。
- permutation validation exact-action: `{permutation.get('exact_action_accuracy', float('nan')):.6f}`。

## 2. 不符合或尚未证明假设的结果

- 单个版本的离线曲线不能证明策略强度，也不能把 BC 表示解释为真实 action value。
- 需要与冻结 baseline 做同 seed、同预算、episode-group bootstrap 对照后，才能判断结构假设。
- 当前最后一轮 validation exact-action 为 `{last['validation/exact_action_accuracy']:.6f}`；是否过拟合需结合 best epoch、loss 与后续官方评测判断。

## 3. 下一步修改措施

- 运行统一 comparison/robustness audit，按 select context、option count、single/multi-action 分 slice。
- 若通过离线对照，则导出独立 candidate 并运行官方 engine 固定池；若未通过，保留本版本并进入下一项单变量消融。

## 4. 本版本具体执行内容

- experiment: `{config.experiment_name}`
- remove_option_position: `{config.model.remove_option_position}`
- role_separated_action: `{config.model.role_separated_action}`
- permutation_augmentation: `{config.permutation_augmentation}`
- seed: `{config.seed}`；batch: `{config.batch_size}`；LR: `{config.learning_rate}`
- stopped_early: `{summary['stopped_early']}`；last epoch: `{summary['last_completed_epoch']}`
- parameter_count: `{summary['parameter_count']}`；peak GPU MiB: `{summary['peak_gpu_memory_mib']:.2f}`
"""
    (output / "ANALYSIS.md").write_text(text, encoding="utf-8")


def train(dataset_root: Path, output: Path, config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    paths = training_paths(output)
    storage = assert_storage_safe(config.storage_path, config.min_free_gib)
    audit = load_dataset_audit(dataset_root)
    inputs = dataset_paths(dataset_root)
    counts = audit.get("records_by_split") or {}
    if int(counts.get("train", 0)) <= 0 or int(counts.get("validation", 0)) <= 0:
        raise ValueError("dataset audit must contain train and validation records")
    for split in ("train", "validation"):
        if not inputs[split].is_file():
            raise FileNotFoundError(inputs[split])

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = FeatureEngineeringPolicy(config.model).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_mib = parameter_count * 4 / 1024**2
    if model_mib > config.max_model_mib:
        raise RuntimeError(f"model size {model_mib:.2f} MiB exceeds {config.max_model_mib:.2f}")
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
    resolved = {
        **config.to_dict(),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_audit": str((dataset_root / "dataset_audit.json").resolve()),
        "dataset_audit_sha256": _sha256(dataset_root / "dataset_audit.json"),
        "dataset_records": counts,
        "dataset_output_sha256": {
            split: (audit.get("outputs", {}).get(split, {}) or {}).get("sha256")
            for split in ("train", "validation", "test")
        },
        "device_resolved": str(device),
        "parameter_count": parameter_count,
        "fp32_model_mib": model_mib,
        "storage": storage.to_dict(),
        "source_sha256": {
            "batching.py": _sha256(PROJECT_ROOT / "batching.py"),
            "codec.py": _sha256(PROJECT_ROOT / "codec.py"),
            "config.py": _sha256(PROJECT_ROOT / "config.py"),
            "dataset.py": _sha256(PROJECT_ROOT / "dataset.py"),
            "model.py": _sha256(PROJECT_ROOT / "model.py"),
            "training.py": _sha256(PROJECT_ROOT / "training.py"),
            "0010_base_model.py": _sha256(
                PROJECT_ROOT.parent / "project_0010_alakazam_sota_model" / "model.py"
            ),
        },
    }
    paths.config.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best_loss = math.inf
    best_exact = -math.inf
    best_epoch = 0
    best_exact_epoch = 0
    best_validation: dict[str, float] = {}
    best_exact_validation: dict[str, float] = {}
    best_exact_permuted: dict[str, float] = {}
    last_metrics: dict[str, float] = {}
    no_improvement = 0
    stopped_early = False
    started = time.monotonic()
    total_batches = (int(counts["train"]) + config.batch_size - 1) // config.batch_size

    with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.monotonic()
            model.train()
            train_dataset.set_epoch(epoch)
            loss_values: list[float] = []
            policy_loss_values: list[float] = []
            delta_loss_values: list[float] = []
            turn_loss_values: list[float] = []
            train_tokens = 0
            for batch_index, batch in enumerate(train_loader, 1):
                if config.permutation_augmentation:
                    batch = permute_candidates(
                        batch,
                        seed=config.seed + epoch * 1_000_003 + batch_index,
                    )
                batch = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    if model.config.action_transition_auxiliary:
                        logits, predicted_delta, predicted_turn = (
                            model.teacher_logits_and_transition(batch)
                        )
                    else:
                        logits = model.teacher_logits(batch)
                    policy_loss = F.cross_entropy(
                        logits.flatten(0, 1),
                        batch["targets"].flatten(),
                        ignore_index=-100,
                    )
                    delta_loss = torch.zeros((), device=device)
                    turn_loss = torch.zeros((), device=device)
                    if model.config.action_transition_auxiliary:
                        delta_mask = batch["transition_delta_mask"]
                        next_mask = batch["transition_next_mask"]
                        if delta_mask.any():
                            delta_loss = F.mse_loss(
                                predicted_delta[delta_mask],
                                batch["transition_delta"][delta_mask],
                            )
                        if next_mask.any():
                            turn_loss = F.binary_cross_entropy_with_logits(
                                predicted_turn[next_mask],
                                batch["transition_turn_changed"][next_mask],
                            )
                    loss = (
                        policy_loss
                        + config.transition_delta_weight * delta_loss
                        + config.transition_turn_weight * turn_loss
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                loss_values.append(float(loss.item()))
                policy_loss_values.append(float(policy_loss.item()))
                delta_loss_values.append(float(delta_loss.item()))
                turn_loss_values.append(float(turn_loss.item()))
                train_tokens += int((batch["targets"] != -100).sum())
                if batch_index == 1 or batch_index % config.log_every_batches == 0:
                    print(
                        json.dumps(
                            {
                                "event": "0012_train_batch",
                                "experiment": config.experiment_name,
                                "epoch": epoch,
                                "batch": batch_index,
                                "batches": total_batches,
                                "loss": sum(loss_values[-config.log_every_batches :])
                                / len(loss_values[-config.log_every_batches :]),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            validation = evaluate(model, validation_loader, device, amp=amp)
            train_evaluation = evaluate(model, train_eval_loader, device, amp=amp)
            permuted = (
                evaluate_permutation_consistency(
                    model,
                    validation_loader,
                    device,
                    amp=amp,
                    seed=config.seed + 90_000_000,
                )
                if config.permutation_validation
                else {}
            )
            last_metrics = {
                "runtime/epoch_seconds": time.monotonic() - epoch_started,
                "train/loss": sum(loss_values) / max(len(loss_values), 1),
                "train/policy_loss": sum(policy_loss_values)
                / max(len(policy_loss_values), 1),
                "train/transition_delta_mse": sum(delta_loss_values)
                / max(len(delta_loss_values), 1),
                "train/transition_turn_bce": sum(turn_loss_values)
                / max(len(turn_loss_values), 1),
                "train/tokens": float(train_tokens),
                **{f"train_eval/{key}": value for key, value in train_evaluation.items()},
                **{f"validation/{key}": value for key, value in validation.items()},
                **{f"validation_permuted/{key}": value for key, value in permuted.items()},
            }
            logger.log(epoch, last_metrics)
            metadata = {
                "model_version": MODEL_VERSION,
                "experiment_name": config.experiment_name,
                "model_config": config.model.to_dict(),
                "training_config": config.to_dict(),
                "dataset_audit_sha256": resolved["dataset_audit_sha256"],
                "epoch_metrics": last_metrics,
            }
            manager.save("latest", model, optimizer=optimizer, step=epoch, metadata=metadata)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_epoch = epoch
                best_validation = dict(validation)
                manager.save("best_validation", model, optimizer=optimizer, step=epoch, metadata=metadata)
            if validation["exact_action_accuracy"] > best_exact:
                best_exact = validation["exact_action_accuracy"]
                best_exact_epoch = epoch
                best_exact_validation = dict(validation)
                best_exact_permuted = dict(permuted)
                no_improvement = 0
                manager.save("best_exact", model, optimizer=optimizer, step=epoch, metadata=metadata)
            else:
                no_improvement += 1
            print(
                json.dumps(
                    {
                        "event": "0012_epoch",
                        "experiment": config.experiment_name,
                        "epoch": epoch,
                        "validation_loss": validation["loss"],
                        "validation_exact": validation["exact_action_accuracy"],
                        "permuted_exact": permuted.get("exact_action_accuracy"),
                        "epochs_without_exact_improvement": no_improvement,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if no_improvement >= config.early_stopping_patience:
                stopped_early = True
                break

    if not best_epoch or not best_exact_epoch:
        raise RuntimeError("training produced no best checkpoints")
    peak_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    summary = {
        "status": "completed",
        "experiment_name": config.experiment_name,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "best_checkpoint": str((paths.checkpoints / "best_validation.pt").resolve()),
        "best_exact_epoch": best_exact_epoch,
        "best_exact_validation": best_exact_validation,
        "best_exact_permuted_validation": best_exact_permuted,
        "best_exact_checkpoint": str((paths.checkpoints / "best_exact.pt").resolve()),
        "last_epoch": last_metrics,
        "last_completed_epoch": int(last_metrics and epoch),
        "runtime_seconds": time.monotonic() - started,
        "parameter_count": parameter_count,
        "fp32_model_mib": model_mib,
        "peak_gpu_memory_mib": peak_mib,
        "stopped_early": stopped_early,
        "early_stopping_patience": config.early_stopping_patience,
        "epochs_without_exact_improvement": no_improvement,
        "test_evaluated": False,
    }
    paths.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_analysis(paths.run, config, summary)
    return summary

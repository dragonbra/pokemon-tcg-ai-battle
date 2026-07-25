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

from .config import TrainingConfig
from .dataset import JsonlShardDataset, dataset_paths, load_dataset_audit
from .model import IDOnlyPointerPolicy, collate_id_only


REFERENCE_NOTEBOOK_ID = "horizen12/ptcg-yushin-id-only-bc-v1"
REFERENCE_NOTEBOOK_URL = "https://www.kaggle.com/code/horizen12/ptcg-yushin-id-only-bc-v1"


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


def _restore_rng(payload: dict[str, Any]) -> None:
    state = payload.get("rng_state") or {}
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _loader(
    path: Path,
    *,
    config: TrainingConfig,
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
        collate_fn=collate_id_only,
        num_workers=0,
        pin_memory=config.device != "cpu",
    )
    return dataset, loader


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
            valid = batch["targets"] != -100
            prediction = logits.argmax(dim=-1)
            total_loss += float(loss.item())
            total_correct += int(((prediction == batch["targets"]) & valid).sum().item())
            total_tokens += int(valid.sum().item())
            exact = ((prediction == batch["targets"]) | ~valid).all(dim=1)
            total_exact += int(exact.sum().item())
            total_rows += int(valid.size(0))
    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "exact_action_accuracy": total_exact / max(total_rows, 1),
        "decisions": float(total_rows),
    }


def train(
    dataset_root: Path,
    output: Path,
    config: TrainingConfig,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    paths = training_paths(output)
    storage = assert_storage_safe(config.storage_path, config.min_free_gib)
    audit = load_dataset_audit(dataset_root)
    inputs = dataset_paths(dataset_root)
    for split in ("train", "validation"):
        if not inputs[split].is_file():
            raise FileNotFoundError(f"missing {split} shard: {inputs[split]}")
    counts = audit.get("records_by_split") or {}
    if int(counts.get("train", 0)) <= 0 or int(counts.get("validation", 0)) <= 0:
        raise ValueError("dataset audit must contain non-empty train and validation splits")

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
    resume_payload: dict[str, Any] | None = None
    start_epoch = 0
    if resume_checkpoint is not None:
        try:
            resume_payload = torch.load(
                resume_checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            resume_payload = torch.load(resume_checkpoint, map_location=device)
        resume_metadata = resume_payload.get("metadata") or {}
        if resume_metadata.get("model_version") != "alakazam_sota_id_only_pointer_bc_v1":
            raise ValueError("resume checkpoint has the wrong model version")
        model.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_epoch = int(resume_payload.get("step", 0))
        if start_epoch <= 0:
            raise ValueError("resume checkpoint is missing a positive epoch step")
        _restore_rng(resume_payload)
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
        "reference_notebook_id": REFERENCE_NOTEBOOK_ID,
        "reference_notebook_url": REFERENCE_NOTEBOOK_URL,
        "storage": storage.to_dict(),
        "resume_checkpoint": (
            str(resume_checkpoint.resolve()) if resume_checkpoint is not None else None
        ),
        "resume_checkpoint_sha256": (
            _sha256(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "start_epoch": start_epoch,
    }
    paths.config.write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resume_metrics = (
        ((resume_payload or {}).get("metadata") or {}).get("epoch_metrics") or {}
    )
    baseline_validation = {
        key.removeprefix("validation/"): float(value)
        for key, value in resume_metrics.items()
        if key.startswith("validation/")
    }
    best_loss = float(baseline_validation.get("loss", math.inf))
    best_exact = float(baseline_validation.get("exact_action_accuracy", -math.inf))
    best_epoch = start_epoch
    best_exact_epoch = start_epoch
    best_validation: dict[str, float] = dict(baseline_validation)
    best_exact_validation: dict[str, float] = dict(baseline_validation)
    last_metrics: dict[str, float] = {}
    no_exact_improvement = 0
    stopped_early = False
    started = time.monotonic()
    total_batches = (int(counts["train"]) + config.batch_size - 1) // config.batch_size
    final_requested_epoch = start_epoch + config.epochs
    if resume_payload is not None:
        manager.save(
            "best_validation",
            model,
            optimizer=optimizer,
            step=start_epoch,
            metadata=resume_payload.get("metadata") or {},
        )
        manager.save(
            "best_exact",
            model,
            optimizer=optimizer,
            step=start_epoch,
            metadata=resume_payload.get("metadata") or {},
        )
    with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
        for epoch in range(start_epoch + 1, final_requested_epoch + 1):
            epoch_started = time.monotonic()
            model.train()
            train_dataset.set_epoch(epoch)
            train_losses: list[float] = []
            train_tokens = 0
            for batch_index, batch in enumerate(train_loader, 1):
                batch = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model.teacher_logits(batch)
                    loss = F.cross_entropy(
                        logits.flatten(0, 1),
                        batch["targets"].flatten(),
                        ignore_index=-100,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
                train_tokens += int((batch["targets"] != -100).sum().item())
                if (
                    batch_index == 1
                    or batch_index % config.log_every_batches == 0
                    or batch_index == total_batches
                ):
                    recent = train_losses[-config.log_every_batches :]
                    print(
                        json.dumps(
                            {
                                "event": "alakazam_sota_train_batch",
                                "epoch": epoch,
                                "epochs": final_requested_epoch,
                                "batch": batch_index,
                                "batches": total_batches,
                                "loss": sum(recent) / len(recent),
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
                "train/loss": sum(train_losses) / max(len(train_losses), 1),
                "train/tokens": float(train_tokens),
                **{f"train_eval/{key}": value for key, value in train_evaluation.items()},
                **{f"validation/{key}": value for key, value in validation.items()},
            }
            logger.log(epoch, last_metrics)
            metadata = {
                "model_version": "alakazam_sota_id_only_pointer_bc_v1",
                "model_config": config.model.to_dict(),
                "training_config": config.to_dict(),
                "dataset_audit_sha256": resolved_config["dataset_audit_sha256"],
                "reference_notebook_id": REFERENCE_NOTEBOOK_ID,
                "epoch_metrics": last_metrics,
            }
            manager.save("latest", model, optimizer=optimizer, step=epoch, metadata=metadata)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_epoch = epoch
                best_validation = dict(validation)
                manager.save(
                    "best_validation",
                    model,
                    optimizer=optimizer,
                    step=epoch,
                    metadata=metadata,
                )
            if validation["exact_action_accuracy"] > best_exact:
                best_exact = validation["exact_action_accuracy"]
                best_exact_epoch = epoch
                best_exact_validation = dict(validation)
                no_exact_improvement = 0
                manager.save(
                    "best_exact",
                    model,
                    optimizer=optimizer,
                    step=epoch,
                    metadata=metadata,
                )
            else:
                no_exact_improvement += 1
            print(
                json.dumps(
                    {
                        "event": "alakazam_sota_epoch",
                        "epoch": epoch,
                        "train_loss": last_metrics["train/loss"],
                        "validation_loss": validation["loss"],
                        "validation_exact_action_accuracy": validation[
                            "exact_action_accuracy"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                config.early_stopping_patience
                and no_exact_improvement >= config.early_stopping_patience
            ):
                stopped_early = True
                print(
                    json.dumps(
                        {
                            "event": "alakazam_sota_early_stop",
                            "epoch": epoch,
                            "best_validation_exact_action_accuracy": best_exact,
                            "epochs_without_exact_improvement": no_exact_improvement,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break

    if not best_epoch:
        raise RuntimeError("training produced no best validation checkpoint")
    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "best_checkpoint": str((paths.checkpoints / "best_validation.pt").resolve()),
        "checkpoint_selection": "minimum validation token cross-entropy",
        "best_exact_epoch": best_exact_epoch,
        "best_exact_validation": best_exact_validation,
        "best_exact_checkpoint": str((paths.checkpoints / "best_exact.pt").resolve()),
        "best_exact_checkpoint_selection": "maximum validation exact-action accuracy",
        "last_epoch": last_metrics,
        "runtime_seconds": time.monotonic() - started,
        "parameter_count": parameter_count,
        "fp32_model_mib": model_mib,
        "test_evaluated": False,
        "resume_start_epoch": start_epoch,
        "stopped_early": stopped_early,
        "early_stopping_patience": config.early_stopping_patience,
        "epochs_without_exact_improvement": no_exact_improvement,
        "best_validation_exact_action_accuracy_observed": best_exact,
    }
    paths.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary

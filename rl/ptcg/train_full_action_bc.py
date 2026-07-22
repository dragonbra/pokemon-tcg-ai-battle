"""Train a pure full-action masked behavior-cloning checkpoint."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from rl.core.batch import collate_encoded
from rl.core.checkpoint import CheckpointManager
from rl.core.logging import TrainingLogger
from rl.core.model import ModelConfig
from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .full_action_model import FullActionPolicyValueNet


MODEL_INPUT_KEYS = (
    "state_numeric",
    "state_card_ids",
    "action_type_ids",
    "action_card_ids",
    "action_target_ids",
    "action_numeric",
    "action_mask",
)


def _load(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("dataset_version") != "ptcg_kaggle_bc_v1":
                raise ValueError(f"unsupported Kaggle BC record at line {line_number}")
            targets = record.get("targets")
            encoded = record.get("encoded")
            if not isinstance(targets, list) or not isinstance(encoded, dict):
                raise ValueError(f"malformed Kaggle BC record at line {line_number}")
            mask = encoded.get("action_mask")
            if not isinstance(mask, list) or not mask or not any(mask):
                raise ValueError(f"missing legal action mask at line {line_number}")
            if any(
                not isinstance(target, int)
                or target < 0
                or target >= len(mask)
                or not mask[target]
                for target in targets
            ):
                raise ValueError(f"illegal target at line {line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"dataset contains no records: {path}")
    return records


def _split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "validation": [], "test": []}
    for record in records:
        split = str(record.get("split", "train"))
        if split not in result:
            raise ValueError(f"unknown dataset split: {split}")
        result[split].append(record)
    if not result["validation"]:
        raise ValueError("dataset must include a validation split")
    return result


def _batch(records: list[dict[str, Any]], max_candidates: int) -> dict[str, Tensor]:
    batch = collate_encoded([record["encoded"] for record in records])
    target_mask = torch.zeros((len(records), max_candidates), dtype=torch.float32)
    counts = []
    minimums = []
    maximums = []
    outcomes = []
    for row, record in enumerate(records):
        targets = [int(target) for target in record["targets"]]
        for target in targets:
            target_mask[row, target] = 1.0
        counts.append(len(targets))
        minimums.append(int(record.get("selection_min_count", 0)))
        maximums.append(int(record.get("selection_max_count", len(targets))))
        outcomes.append(float(record.get("terminal_outcome", 0.0)))
    batch["target_mask"] = target_mask
    batch["target_count"] = torch.tensor(counts, dtype=torch.long)
    batch["selection_min_count"] = torch.tensor(minimums, dtype=torch.long)
    batch["selection_max_count"] = torch.tensor(maximums, dtype=torch.long)
    batch["terminal_outcome"] = torch.tensor(outcomes, dtype=torch.float32)
    return batch


def _inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: batch[key] for key in MODEL_INPUT_KEYS}


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _masked_count_logits(count_logits: Tensor, minimum: Tensor, maximum: Tensor) -> Tensor:
    positions = torch.arange(count_logits.shape[1], device=count_logits.device).unsqueeze(0)
    valid = (positions >= minimum.unsqueeze(1)) & (positions <= maximum.unsqueeze(1))
    return count_logits.masked_fill(~valid, torch.finfo(count_logits.dtype).min)


def _losses(
    logits: Tensor,
    count_logits: Tensor,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    action_mask = batch["action_mask"]
    targets = batch["target_mask"]
    counts = batch["target_count"]
    single = counts.eq(1)
    single_target = targets.argmax(dim=1)
    single_loss = F.cross_entropy(logits, single_target, reduction="none")
    binary = F.binary_cross_entropy_with_logits(
        logits.clamp(min=-30.0, max=30.0), targets, reduction="none"
    )
    multi_loss = (binary * action_mask.float()).sum(dim=1) / (
        action_mask.float().sum(dim=1).clamp_min(1.0)
    )
    policy_loss = torch.where(single, single_loss, multi_loss).mean()
    valid_count_logits = _masked_count_logits(
        count_logits,
        batch["selection_min_count"].clamp_min(0),
        batch["selection_max_count"].clamp_max(count_logits.shape[1] - 1),
    )
    count_loss = F.cross_entropy(valid_count_logits, counts.clamp_max(count_logits.shape[1] - 1))
    return policy_loss + count_loss, policy_loss, count_loss


@torch.no_grad()
def _evaluate(
    model: FullActionPolicyValueNet,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    total = exact = single_total = single_correct = multi_total = multi_exact = count_correct = 0
    policy_losses: list[float] = []
    count_losses: list[float] = []
    context_total: collections.Counter[str] = collections.Counter()
    context_exact: collections.Counter[str] = collections.Counter()
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = _move(_batch(chunk, model.config.max_candidates), device)
        _, logits, count_logits = model.forward_with_count(**_inputs(batch))
        loss, policy_loss, count_loss = _losses(logits, count_logits, batch)
        del loss
        policy_losses.append(float(policy_loss.item()))
        count_losses.append(float(count_loss.item()))
        valid_count_logits = _masked_count_logits(
            count_logits,
            batch["selection_min_count"],
            batch["selection_max_count"].clamp_max(count_logits.shape[1] - 1),
        )
        predicted_counts = valid_count_logits.argmax(dim=1).clamp_max(
            batch["action_mask"].sum(dim=1)
        )
        count_correct += int(predicted_counts.eq(batch["target_count"]).sum().item())
        for row, record in enumerate(chunk):
            target = set(int(value) for value in record["targets"])
            count = int(predicted_counts[row])
            if count:
                predicted = set(torch.topk(logits[row], k=count).indices.tolist())
            else:
                predicted = set()
            exact += predicted == target
            total += 1
            context = (
                f"type={record.get('selection_type')},"
                f"context={record.get('selection_context')}"
            )
            context_total[context] += 1
            context_exact[context] += predicted == target
            if len(target) == 1:
                single_total += 1
                single_correct += predicted == target
            else:
                multi_total += 1
                multi_exact += predicted == target
    return {
        "records": float(total),
        "exact_action_rate": exact / max(1, total),
        "single_action_accuracy": single_correct / max(1, single_total),
        "multi_action_exact_rate": multi_exact / max(1, multi_total),
        "selection_count_accuracy": count_correct / max(1, total),
        "policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
        "count_loss": sum(count_losses) / max(1, len(count_losses)),
        "legal_action_rate": 1.0,
        "by_selection": {
            context: {
                "records": context_total[context],
                "exact_action_rate": context_exact[context] / context_total[context],
            }
            for context in sorted(context_total)
        },
    }


def train(
    dataset: Path,
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    d_model: int,
    hidden_dim: int,
    num_heads: int,
    transformer_layers: int,
    dropout: float,
    storage_path: Path,
    min_free_gib: float,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    storage = assert_storage_safe(storage_path, min_free_gib)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    records = _load(dataset)
    splits = _split_records(records)
    feature_config = records[0]["encoded"]
    state_numeric_dim = len(feature_config["state_numeric"])
    state_token_count = len(feature_config["state_card_ids"])
    candidate_numeric_dim = len(feature_config["action_numeric"][0])
    max_candidates = len(feature_config["action_mask"])
    model_config = ModelConfig(
        state_numeric_dim=state_numeric_dim,
        state_token_count=state_token_count,
        candidate_numeric_dim=candidate_numeric_dim,
        max_candidates=max_candidates,
        card_vocab_size=4096,
        action_type_vocab_size=32,
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_transformer_layers=transformer_layers,
        dropout=dropout,
    )
    model = FullActionPolicyValueNet(model_config, max_selection_count=max_candidates).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    manager = CheckpointManager(output / "checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": str(dataset.resolve()),
        "model_config": model_config.to_dict(),
        "max_selection_count": max_candidates,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "dataset_records": {split: len(values) for split, values in splits.items()},
        "selection_contract": "single categorical CE + multi-label candidate BCE + count CE",
        "fallback": None,
        "reward_profile": "none_pure_behavior_cloning",
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    best_score = -1.0
    last_summary: dict[str, Any] = {}
    with TrainingLogger(output / "metrics.jsonl", output / "tensorboard") as logger:
        for epoch in range(1, epochs + 1):
            model.train()
            order = list(range(len(splits["train"])))
            random.Random(seed + epoch).shuffle(order)
            train_losses: list[float] = []
            for start in range(0, len(order), batch_size):
                chunk = [splits["train"][index] for index in order[start : start + batch_size]]
                batch = _move(_batch(chunk, max_candidates), device)
                optimizer.zero_grad()
                _, logits, count_logits = model.forward_with_count(**_inputs(batch))
                loss, _, _ = _losses(logits, count_logits, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
            train_metrics = _evaluate(model, splits["train"], device, batch_size)
            validation_metrics = _evaluate(model, splits["validation"], device, batch_size)
            last_summary = {
                "train/loss": sum(train_losses) / max(1, len(train_losses)),
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
            }
            logger.log(epoch, last_summary)
            metadata = {
                "task": "ptcg_full_action_behavior_cloning",
                "dataset": str(dataset.resolve()),
                "feature_config": {
                    "state_numeric_dim": state_numeric_dim,
                    "state_token_count": state_token_count,
                    "candidate_numeric_dim": candidate_numeric_dim,
                    "max_candidates": max_candidates,
                    "card_vocab_size": 4096,
                    "action_type_vocab_size": 32,
                    "schema_version": "ptcg_features_v6",
                },
                "model_config": model_config.to_dict(),
                "max_selection_count": max_candidates,
                "selection_contract": config["selection_contract"],
                "fallback": None,
                "seed": seed,
                "epoch_metrics": last_summary,
                "storage_path": storage.path,
            }
            manager.save("latest", model, optimizer=optimizer, step=epoch, metadata=metadata)
            score = validation_metrics["exact_action_rate"]
            if score > best_score:
                best_score = score
                manager.save(
                    "best_validation", model, optimizer=optimizer, step=epoch, metadata=metadata
                )
    manager.load(output / "checkpoints" / "best_validation.pt", model, map_location=device)
    test_metrics = _evaluate(model, splits["test"], device, batch_size) if splits["test"] else {}
    summary = {
        "best_validation_exact_action_rate": best_score,
        "final_test": test_metrics,
        "last_epoch": last_summary,
        "config": config,
        "checkpoint": str((output / "checkpoints" / "best_validation.pt").resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    result = train(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        storage_path=args.storage_path,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

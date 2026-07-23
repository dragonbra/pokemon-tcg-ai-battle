"""Train a pure full-action masked behavior-cloning checkpoint."""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from rl.core.batch import UNIVERSAL_INPUT_KEYS, collate_encoded
from rl.core.checkpoint import CheckpointManager
from rl.core.logging import TrainingLogger
from rl.core.model import ModelConfig
from rl.core.runs import training_paths
from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from rl.model.full_action_model import FullActionPolicyValueNet


SUPPORTED_DATASET_VERSIONS = frozenset(
    {"ptcg_kaggle_bc_v1", "ptcg_kaggle_bc_universal"}
)


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
            if record.get("dataset_version") not in SUPPORTED_DATASET_VERSIONS:
                raise ValueError(f"unsupported Kaggle BC record at line {line_number}")
            targets = record.get("targets")
            encoded = record.get("encoded")
            if not isinstance(targets, list) or not isinstance(encoded, dict):
                raise ValueError(f"malformed Kaggle BC record at line {line_number}")
            mask = encoded.get("action_mask")
            if not isinstance(mask, list) or not mask or not any(mask):
                raise ValueError(f"missing legal action mask at line {line_number}")
            minimum = int(record.get("selection_min_count", 0))
            maximum = int(record.get("selection_max_count", len(targets)))
            target_count = int(record.get("target_count", len(targets)))
            legal_count = sum(bool(value) for value in mask)
            if target_count != len(targets) or len(set(targets)) != len(targets):
                raise ValueError(f"invalid target cardinality at line {line_number}")
            if not 0 <= minimum <= target_count <= maximum <= legal_count:
                raise ValueError(f"selection count is outside legal range at line {line_number}")
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
    expert_teams = {
        str(record.get("expert_team_name"))
        for record in records
        if record.get("expert_team_name")
    }
    if len(expert_teams) != 1:
        raise ValueError(
            "behavior cloning requires exactly one expert team; "
            f"found {sorted(expert_teams)}"
        )
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
    keys = MODEL_INPUT_KEYS + tuple(key for key in UNIVERSAL_INPUT_KEYS if key in batch)
    return {key: batch[key] for key in keys}


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
    training_started = time.monotonic()
    random.seed(seed)
    torch.manual_seed(seed)
    storage = assert_storage_safe(storage_path, min_free_gib)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    records = _load(dataset)
    card_metadata_path = dataset.with_suffix(dataset.suffix + ".card_metadata.json")
    card_metadata = (
        json.loads(card_metadata_path.read_text(encoding="utf-8"))
        if card_metadata_path.is_file()
        else {}
    )
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
        deck_token_count=len(feature_config.get("deck_card_ids", [])),
        deck_numeric_dim=(
            len(feature_config.get("deck_card_numeric", [[]])[0])
            if feature_config.get("deck_card_numeric")
            else 0
        ),
        entity_token_count=len(feature_config.get("entity_card_ids", [])),
        entity_numeric_dim=(
            len(feature_config.get("entity_numeric", [[]])[0])
            if feature_config.get("entity_numeric")
            else 0
        ),
        history_token_count=len(feature_config.get("history_card_ids", [])),
        history_numeric_dim=(
            len(feature_config.get("history_numeric", [[]])[0])
            if feature_config.get("history_numeric")
            else 0
        ),
        expert_vocab_size=64 if "expert_ids" in feature_config else 0,
    )
    model = FullActionPolicyValueNet(model_config, max_selection_count=max_candidates).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    paths = training_paths(output)
    manager = CheckpointManager(paths.checkpoints)
    paths.run.mkdir(parents=True, exist_ok=True)
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
        "feature_schema": "ptcg_features_universal"
        if "deck_card_ids" in feature_config
        else "ptcg_features_v6",
        "card_metadata": str(card_metadata_path.resolve()) if card_metadata else None,
    }
    paths.config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    best_score = -1.0
    best_epoch = 0
    last_summary: dict[str, Any] = {}
    with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
        for epoch in range(1, epochs + 1):
            epoch_started = time.monotonic()
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
                "runtime/epoch_seconds": time.monotonic() - epoch_started,
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
                    "schema_version": config["feature_schema"],
                    "deck_token_count": model_config.deck_token_count,
                    "deck_numeric_dim": model_config.deck_numeric_dim,
                    "entity_token_count": model_config.entity_token_count,
                    "entity_numeric_dim": model_config.entity_numeric_dim,
                    "history_token_count": model_config.history_token_count,
                    "history_numeric_dim": model_config.history_numeric_dim,
                    "expert_vocab_size": model_config.expert_vocab_size,
                },
                "deck": [
                    int(value) - 1
                    for value in feature_config.get("deck_card_ids", [])
                    if int(value) > 0
                ],
                "expert_id": int(feature_config.get("expert_ids", 1)),
                "card_metadata": card_metadata,
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
                best_epoch = epoch
                manager.save(
                    "best_validation", model, optimizer=optimizer, step=epoch, metadata=metadata
                )
    manager.load(paths.checkpoints / "best_validation.pt", model, map_location=device)
    test_metrics = _evaluate(model, splits["test"], device, batch_size) if splits["test"] else {}
    summary = {
        "best_validation_exact_action_rate": best_score,
        "best_epoch": best_epoch,
        "final_test": test_metrics,
        "last_epoch": last_summary,
        "config": config,
        "checkpoint": str((paths.checkpoints / "best_validation.pt").resolve()),
        "runtime": {
            "train_function_seconds": time.monotonic() - training_started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "state_dict_tensors": len(model.state_dict()),
        },
    }
    paths.summary.write_text(
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

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rl_environment.batch import collate_encoded
from rl_environment.checkpoint import CheckpointManager
from rl_environment.losses import (
    masked_cross_entropy,
    masked_cross_entropy_per_sample,
    masked_soft_cross_entropy_per_sample,
    value_huber_loss,
)
from rl_environment.logging import TrainingLogger
from rl_environment.model import CandidatePolicyValueNet, ModelConfig
from rl_environment.runs import training_paths
from rl_environment.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .dataset import load_behavior_cloning_dataset
from archive.train_legacy.alakazam_bc_rl.features import PTCGFeatureConfig, feature_config_for_schema


MODEL_INPUT_KEYS = (
    "state_numeric",
    "state_card_ids",
    "action_type_ids",
    "action_card_ids",
    "action_target_ids",
    "action_numeric",
    "action_mask",
)
DEFAULT_OUTPUT = Path("/tmp/ptcg_behavior_cloning")


def _model_config(feature_config: PTCGFeatureConfig, args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        state_numeric_dim=feature_config.state_numeric_dim,
        state_token_count=feature_config.state_token_count,
        candidate_numeric_dim=feature_config.candidate_numeric_dim,
        max_candidates=feature_config.max_candidates,
        card_vocab_size=feature_config.card_vocab_size,
        action_type_vocab_size=feature_config.action_type_vocab_size,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_transformer_layers=args.transformer_layers,
        dropout=args.dropout,
    )


def _batch(records: list[dict[str, Any]]) -> dict[str, Tensor]:
    batch = collate_encoded([record["encoded"] for record in records])
    batch["target"] = torch.tensor(
        [int(record["target"]) for record in records], dtype=torch.long
    )
    batch["terminal_outcome"] = torch.tensor(
        [float(record.get("terminal_outcome", 0.0)) for record in records],
        dtype=torch.float32,
    )
    batch["potential_shaping"] = torch.tensor(
        [float((record.get("potential_shaping") or {}).get("total", 0.0)) for record in records],
        dtype=torch.float32,
    )
    batch["transition_return"] = torch.tensor(
        [
            float(record.get("transition_return", record.get("terminal_outcome", 0.0)))
            for record in records
        ],
        dtype=torch.float32,
    )
    candidate_width = len(records[0]["encoded"]["action_mask"])
    mcts_targets: list[list[float]] = []
    for record in records:
        if "mcts_policy" in record:
            target = [float(value) for value in record["mcts_policy"]]
        else:
            target = [0.0] * candidate_width
            target[int(record["target"])] = 1.0
        if len(target) > candidate_width:
            raise ValueError("mcts_policy is wider than the encoded candidate mask")
        mcts_targets.append(target + [0.0] * (candidate_width - len(target)))
    batch["mcts_policy"] = torch.tensor(mcts_targets, dtype=torch.float32)
    return batch


def _model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: batch[key] for key in MODEL_INPUT_KEYS}


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


@torch.no_grad()
def _evaluate(
    model: CandidatePolicyValueNet,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_correct = 0
    total_legal = 0
    total_count = 0
    losses: list[float] = []
    for start in range(0, len(records), 256):
        batch_records = records[start : start + 256]
        batch = _move_batch(_batch(batch_records), device)
        _, logits = model(**_model_inputs(batch))
        loss = masked_cross_entropy(logits, batch["target"], batch["action_mask"])
        prediction = logits.argmax(dim=-1)
        total_correct += int(prediction.eq(batch["target"]).sum().item())
        total_legal += int(
            batch["action_mask"].gather(1, prediction[:, None]).sum().item()
        )
        total_count += len(batch_records)
        losses.append(float(loss.item()))
    return {
        "action_accuracy": total_correct / max(1, total_count),
        "legal_action_rate": total_legal / max(1, total_count),
        "bc_loss": sum(losses) / max(1, len(losses)),
    }


def train(
    dataset_path: Path,
    output_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 7,
    validation_fraction: float = 0.1,
    device_name: str = "auto",
    storage_path: Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if (
        args.outcome_weight < 0
        or args.value_loss_weight < 0
        or args.potential_weight < 0
        or args.return_weight < 0
    ):
        raise ValueError("reward weights must not be negative")
    if not 0.0 <= args.mcts_policy_weight <= 1.0:
        raise ValueError("mcts_policy_weight must be in [0, 1]")
    torch.manual_seed(seed)
    random.seed(seed)
    storage = assert_storage_safe(storage_path, min_free_gib)
    device = _resolve_device(device_name)
    records = load_behavior_cloning_dataset(dataset_path)
    schemas = {str(record.get("feature_schema_version")) for record in records}
    if len(schemas) != 1:
        raise ValueError(f"dataset must contain exactly one feature schema: {sorted(schemas)}")
    feature_schema_version = next(iter(schemas))
    feature_config = feature_config_for_schema(feature_schema_version)
    selection_scope = getattr(args, "selection_scope", "all")
    if selection_scope not in {"all", "main", "effect"}:
        raise ValueError("selection_scope must be all, main, or effect")
    if selection_scope == "main":
        records = [
            record
            for record in records
            if int(record.get("selection_type", -1)) == 0
            and int(record.get("selection_context", -1)) == 0
        ]
    elif selection_scope == "effect":
        records = [
            record
            for record in records
            if not (
                int(record.get("selection_type", -1)) == 0
                and int(record.get("selection_context", -1)) == 0
            )
        ]
    if not records:
        raise ValueError(f"selection scope contains no records: {selection_scope}")
    model_config = _model_config(feature_config, args)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(len(shuffled) * validation_fraction)
    if len(shuffled) > 1 and validation_fraction > 0:
        validation_count = max(1, validation_count)
    validation = shuffled[:validation_count]
    training = shuffled[validation_count:] or shuffled
    model = CandidatePolicyValueNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    paths = training_paths(output_dir)
    manager = CheckpointManager(paths.checkpoints)
    best_score = -1.0
    last_metrics: dict[str, float] = {}
    paths.run.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path.resolve()),
                "model_config": model_config.to_dict(),
                "feature_config": feature_config.__dict__,
                "feature_schema_version": feature_schema_version,
                "selection_scope": selection_scope,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_fraction": validation_fraction,
                "seed": seed,
                "device": str(device),
                "outcome_weight": args.outcome_weight,
                "value_loss_weight": args.value_loss_weight,
                "mcts_policy_weight": args.mcts_policy_weight,
                "potential_weight": args.potential_weight,
                "return_weight": args.return_weight,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
        for epoch in range(1, epochs + 1):
            model.train()
            order = list(range(len(training)))
            random.Random(seed + epoch).shuffle(order)
            train_losses: list[float] = []
            train_policy_losses: list[float] = []
            train_value_losses: list[float] = []
            for start in range(0, len(order), batch_size):
                batch_records = [training[index] for index in order[start : start + batch_size]]
                batch = _move_batch(_batch(batch_records), device)
                optimizer.zero_grad()
                value, logits = model(**_model_inputs(batch))
                hard_per_sample = masked_cross_entropy_per_sample(
                    logits, batch["target"], batch["action_mask"]
                )
                if args.mcts_policy_weight:
                    soft_per_sample = masked_soft_cross_entropy_per_sample(
                        logits, batch["mcts_policy"], batch["action_mask"]
                    )
                    per_sample = (
                        (1.0 - args.mcts_policy_weight) * hard_per_sample
                        + args.mcts_policy_weight * soft_per_sample
                    )
                else:
                    per_sample = hard_per_sample
                weights = (
                    1.0
                    + args.outcome_weight * batch["terminal_outcome"]
                    + args.potential_weight * batch["potential_shaping"]
                    + args.return_weight * batch["transition_return"]
                ).clamp_min(0.1)
                policy_loss = (per_sample * weights).mean()
                value_loss = value_huber_loss(value, batch["terminal_outcome"])
                loss = policy_loss + args.value_loss_weight * value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
                train_policy_losses.append(float(policy_loss.item()))
                train_value_losses.append(float(value_loss.item()))

            train_metrics = _evaluate(model, training, device)
            validation_metrics = _evaluate(model, validation or training, device)
            last_metrics = {
                "train/bc_loss": sum(train_losses) / max(1, len(train_losses)),
                "train/policy_loss": sum(train_policy_losses) / max(1, len(train_policy_losses)),
                "train/value_loss": sum(train_value_losses) / max(1, len(train_value_losses)),
                "train/action_accuracy": train_metrics["action_accuracy"],
                "train/legal_action_rate": train_metrics["legal_action_rate"],
                "validation/bc_loss": validation_metrics["bc_loss"],
                "validation/action_accuracy": validation_metrics["action_accuracy"],
                "validation/legal_action_rate": validation_metrics["legal_action_rate"],
            }
            logger.log(epoch, last_metrics)
            metadata = {
                "task": "ptcg_behavior_cloning",
                "dataset": str(dataset_path.resolve()),
                "feature_schema_version": feature_schema_version,
                "model_config": model_config.to_dict(),
                "feature_config": feature_config.__dict__,
                "selection_scope": selection_scope,
                "seed": seed,
                "device": str(device),
                "outcome_weight": args.outcome_weight,
                "value_loss_weight": args.value_loss_weight,
                "mcts_policy_weight": args.mcts_policy_weight,
                "potential_weight": args.potential_weight,
                "return_weight": args.return_weight,
                "reward_profile": "terminal_plus_transition_return_v1"
                if args.return_weight
                else (
                    "terminal_plus_visible_potential_v1"
                    if args.potential_weight
                    else (
                        "terminal_outcome_v1"
                        if args.outcome_weight or args.value_loss_weight
                        else "none"
                    )
                ),
                "storage_path": storage.path,
                "storage_free_gib": round(storage.free_gib, 2),
                "epoch_metrics": last_metrics,
            }
            manager.save("latest", model, optimizer=optimizer, step=epoch, metadata=metadata)
            score = validation_metrics["action_accuracy"]
            if score > best_score:
                best_score = score
                manager.save(
                    "best_validation",
                    model,
                    optimizer=optimizer,
                    step=epoch,
                    metadata=metadata,
                )
    summary = {
        "records": len(records),
        "training_records": len(training),
        "validation_records": len(validation),
        "device": str(device),
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
        "best_validation_accuracy": best_score,
        **last_metrics,
        "checkpoint": str((paths.checkpoints / "best_validation.pt").resolve()),
    }
    paths.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--outcome-weight",
        type=float,
        default=0.0,
        help="reweight policy loss by terminal win/loss outcome",
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=0.0,
        help="add value-head Huber loss against terminal outcome",
    )
    parser.add_argument(
        "--mcts-policy-weight",
        type=float,
        default=0.0,
        help="blend MCTS visit-count cross entropy with hard target loss",
    )
    parser.add_argument(
        "--selection-scope",
        choices=("all", "main", "effect"),
        default="all",
        help="train all records, main actions only, or effect selections only",
    )
    parser.add_argument(
        "--potential-weight",
        type=float,
        default=0.0,
        help="reweight policy loss by visible transition potential shaping",
    )
    parser.add_argument(
        "--return-weight",
        type=float,
        default=0.0,
        help="reweight policy loss by clipped transition return",
    )
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")
    result = train(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        device_name=args.device,
        storage_path=args.storage_path,
        min_free_gib=args.min_free_gib,
        args=args,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

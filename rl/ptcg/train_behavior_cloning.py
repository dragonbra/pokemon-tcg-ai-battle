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
from rl.core.losses import masked_cross_entropy
from rl.core.logging import TrainingLogger
from rl.core.model import CandidatePolicyValueNet, ModelConfig

from .dataset import load_behavior_cloning_dataset
from .features import FEATURE_SCHEMA_VERSION, PTCGFeatureConfig


MODEL_INPUT_KEYS = (
    "state_numeric",
    "state_card_ids",
    "action_type_ids",
    "action_card_ids",
    "action_target_ids",
    "action_numeric",
    "action_mask",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "rl" / "runs" / "ptcg_behavior_cloning"


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
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    torch.manual_seed(seed)
    random.seed(seed)
    device = _resolve_device(device_name)
    records = load_behavior_cloning_dataset(dataset_path)
    feature_config = PTCGFeatureConfig()
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
    manager = CheckpointManager(output_dir / "checkpoints")
    best_score = -1.0
    last_metrics: dict[str, float] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    with TrainingLogger(output_dir / "metrics.jsonl", output_dir / "tensorboard") as logger:
        for epoch in range(1, epochs + 1):
            model.train()
            order = list(range(len(training)))
            random.Random(seed + epoch).shuffle(order)
            train_losses: list[float] = []
            for start in range(0, len(order), batch_size):
                batch_records = [training[index] for index in order[start : start + batch_size]]
                batch = _move_batch(_batch(batch_records), device)
                optimizer.zero_grad()
                _, logits = model(**_model_inputs(batch))
                loss = masked_cross_entropy(logits, batch["target"], batch["action_mask"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            train_metrics = _evaluate(model, training, device)
            validation_metrics = _evaluate(model, validation or training, device)
            last_metrics = {
                "train/bc_loss": sum(train_losses) / max(1, len(train_losses)),
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
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "model_config": model_config.to_dict(),
                "feature_config": feature_config.__dict__,
                "seed": seed,
                "device": str(device),
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
    return {
        "records": len(records),
        "training_records": len(training),
        "validation_records": len(validation),
        "device": str(device),
        "best_validation_accuracy": best_score,
        **last_metrics,
        "checkpoint": str((output_dir / "checkpoints" / "best_validation.pt").resolve()),
    }


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
        args=args,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

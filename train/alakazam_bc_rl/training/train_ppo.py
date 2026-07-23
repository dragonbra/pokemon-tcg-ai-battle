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
from rl_environment.logging import TrainingLogger
from rl_environment.losses import value_huber_loss
from rl_environment.model import CandidatePolicyValueNet, ModelConfig
from rl_environment.runs import training_paths
from rl_environment.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe
from train.alakazam_bc_rl.features import PTCGFeatureConfig
from train.alakazam_bc_rl.training.dataset import load_behavior_cloning_dataset


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


def _load_model(
    checkpoint: Path, device: torch.device
) -> tuple[CandidatePolicyValueNet, dict[str, Any], str]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = payload.get("metadata") or {}
    saved_config = metadata.get("model_config")
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint metadata.model_config is required")
    model = CandidatePolicyValueNet(ModelConfig(**saved_config)).to(device)
    model.load_state_dict(payload["model"])
    saved_features = metadata.get("feature_config")
    feature_config = (
        dict(saved_features) if isinstance(saved_features, dict) else PTCGFeatureConfig().__dict__
    )
    feature_schema_version = str(
        metadata.get("feature_schema_version") or "ptcg_features_v3"
    )
    return model, feature_config, feature_schema_version


@torch.no_grad()
def _old_policy_targets(
    model: CandidatePolicyValueNet,
    records: list[dict[str, Any]],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    old_log_probs: list[Tensor] = []
    old_values: list[Tensor] = []
    outcomes: list[Tensor] = []
    model.eval()
    for start in range(0, len(records), 512):
        batch = _move(_batch(records[start : start + 512]), device)
        value, logits = model(**_inputs(batch))
        log_probs = torch.log_softmax(logits, dim=-1)
        old_log_probs.append(log_probs.gather(1, batch["target"][:, None]).squeeze(1))
        old_values.append(value)
        outcomes.append(batch["terminal_outcome"])
    old_log_prob = torch.cat(old_log_probs)
    old_value = torch.cat(old_values)
    outcome = torch.cat(outcomes)
    return old_log_prob, old_value, outcome


def train(
    dataset_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-5,
    clip_ratio: float = 0.2,
    value_loss_weight: float = 0.5,
    entropy_weight: float = 0.01,
    shaping_weight: float = 0.0,
    seed: int = 7,
    device_name: str = "auto",
    storage_path: Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, float | int | str]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if clip_ratio <= 0 or value_loss_weight < 0 or entropy_weight < 0:
        raise ValueError("invalid PPO loss coefficients")
    storage = assert_storage_safe(storage_path, min_free_gib)
    torch.manual_seed(seed)
    random.seed(seed)
    device = _resolve_device(device_name)
    records = load_behavior_cloning_dataset(dataset_path)
    model, feature_config, feature_schema_version = _load_model(checkpoint, device)
    old_log_prob, old_value, terminal_outcome = _old_policy_targets(model, records, device)
    shaping = torch.tensor(
        [float((record.get("potential_shaping") or {}).get("total", 0.0)) for record in records],
        dtype=torch.float32,
        device=device,
    )
    shaped_return = terminal_outcome + shaping_weight * shaping
    policy_target = shaped_return.clamp(-1.0, 1.0)
    advantage = policy_target - old_value
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    paths = training_paths(output_dir)
    manager = CheckpointManager(paths.checkpoints)
    paths.run.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path.resolve()),
                "source_checkpoint": str(checkpoint.resolve()),
                "model_config": model.config.to_dict(),
                "feature_config": feature_config,
                "feature_schema_version": feature_schema_version,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "clip_ratio": clip_ratio,
                "value_loss_weight": value_loss_weight,
                "entropy_weight": entropy_weight,
                "shaping_weight": shaping_weight,
                "seed": seed,
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    last_metrics: dict[str, float] = {}

    with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
        for epoch in range(1, epochs + 1):
            model.train()
            order = list(range(len(records)))
            random.Random(seed + epoch).shuffle(order)
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropy_values: list[float] = []
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch = _move(_batch([records[index] for index in indices]), device)
                old_log = old_log_prob[indices]
                batch_advantage = advantage[indices]
                optimizer.zero_grad()
                value, logits = model(**_inputs(batch))
                log_probs = torch.log_softmax(logits, dim=-1)
                new_log = log_probs.gather(1, batch["target"][:, None]).squeeze(1)
                ratio = (new_log - old_log).exp()
                unclipped = ratio * batch_advantage
                clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                batch_shaped_return = (
                    batch["terminal_outcome"] + shaping_weight * batch["potential_shaping"]
                ).clamp(-1.0, 1.0)
                value_loss = value_huber_loss(value, batch_shaped_return)
                probabilities = torch.softmax(logits, dim=-1)
                entropy = -(probabilities * log_probs).sum(dim=-1).mean()
                loss = policy_loss + value_loss_weight * value_loss - entropy_weight * entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_values.append(float(entropy.item()))

            last_metrics = {
                "train/ppo_policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
                "train/ppo_value_loss": sum(value_losses) / max(1, len(value_losses)),
                "train/ppo_entropy": sum(entropy_values) / max(1, len(entropy_values)),
                "train/terminal_outcome_mean": float(terminal_outcome.mean().item()),
                "train/shaped_return_mean": float(policy_target.mean().item()),
                "train/shaping_reward_mean": float(shaping.mean().item()),
            }
            logger.log(epoch, last_metrics)
            metadata = {
                "task": "ptcg_masked_ppo_terminal_v1",
                "dataset": str(dataset_path.resolve()),
                "source_checkpoint": str(checkpoint.resolve()),
                "model_config": model.config.to_dict(),
                "feature_config": feature_config,
                "feature_schema_version": feature_schema_version,
                "seed": seed,
                "device": str(device),
                "clip_ratio": clip_ratio,
                "value_loss_weight": value_loss_weight,
                "entropy_weight": entropy_weight,
                "shaping_weight": shaping_weight,
                "value_target": "clipped_terminal_plus_potential",
                "storage_path": storage.path,
                "storage_free_gib": round(storage.free_gib, 2),
                "epoch_metrics": last_metrics,
            }
            manager.save("latest", model, optimizer=optimizer, step=epoch, metadata=metadata)

    summary = {
        "records": len(records),
        "device": str(device),
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
        **last_metrics,
        "checkpoint": str((paths.checkpoints / "latest.pt").resolve()),
    }
    paths.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--shaping-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    result = train(
        args.dataset,
        args.checkpoint,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        shaping_weight=args.shaping_weight,
        seed=args.seed,
        device_name=args.device,
        storage_path=args.storage_path,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

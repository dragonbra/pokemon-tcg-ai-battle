"""Run one immutable 0036 Value ablation version."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .model import FrozenEncoderValueNetwork, load_source_policy
from .training.materialized import MaterializedValueDataset
from .training.objective import LossWeights
from .training.trainer import TrainerConfig, ValueTrainer


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "0036_dedicated_action_value_network"
PRESETS = {
    "raw_pool_mlp": ("raw_pool_mlp", LossWeights()),
    "summary_mlp": ("summary_mlp", LossWeights()),
    "latent_value_only": ("latent_queries", LossWeights()),
    "latent_value_archetype": ("latent_queries", LossWeights(archetype=0.1)),
    "latent_value_archetype_diff": ("latent_queries", LossWeights(archetype=0.1, final_diff=0.1)),
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prepare_run(version: str) -> Path:
    run_root = REPO_ROOT / "rl_runs" / PROJECT_ID / "versions" / version
    evaluation = REPO_ROOT / "experiments" / PROJECT_ID / "evaluation" / f"{version}.html"
    if run_root.exists() and any(run_root.rglob("*")):
        raise FileExistsError(f"0036 version is already used: {run_root}")
    if evaluation.exists():
        raise FileExistsError(f"0036 evaluation version is already used: {evaluation}")
    for name in ("artifact", "tensorboard", "checkpoint", "wandb"):
        (run_root / name).mkdir(parents=True, exist_ok=False)
    return run_root


def run(version: str, preset: str, dataset_path: Path, source_dataset_path: Path,
        source_checkpoint: Path,
        trainer_config: TrainerConfig, device: str = "cuda") -> dict[str, object]:
    if preset not in PRESETS:
        raise ValueError(f"unknown 0036 ablation preset: {preset}")
    run_root = _prepare_run(version)
    architecture, loss_weights = PRESETS[preset]
    wandb_display_name = f"0036 · dedicated_action_value_network · {version}"
    config = {
        "project_id": PROJECT_ID,
        "version": version,
        "preset": preset,
        "architecture": architecture,
        "loss_weights": asdict(loss_weights),
        "trainer": asdict(trainer_config),
        "dataset": str(dataset_path),
        "source_dataset": str(source_dataset_path),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_expected_sha256": "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8",
        "wandb": {
            "mode": "online",
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "display_name": wandb_display_name,
        },
    }
    _write_json(run_root / "artifact" / "training_config.json", config)
    _write_json(run_root / "artifact" / "status.json", {"state": "initializing", "version": version})
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_ENTITY"] = "dragon_bra"
    os.environ["WANDB_PROJECT"] = "pokemon-tcg-policy-learning"
    os.environ["WANDB_JOB_TYPE"] = "value"
    os.environ["WANDB_NAME"] = wandb_display_name
    random.seed(trainer_config.seed)
    torch.manual_seed(trainer_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trainer_config.seed)
    try:
        encoder, source = load_source_policy(source_checkpoint, device)
        model = FrozenEncoderValueNetwork(encoder, architecture=architecture)
        dataset = MaterializedValueDataset(dataset_path, source_root=source_dataset_path)
        summary = ValueTrainer(model, dataset, source, run_root, trainer_config,
                               loss_weights, device).train()
        _write_json(run_root / "artifact" / "training_summary.json", summary)
        status_path = run_root / "artifact" / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        status.update({"state": "complete", "version": version,
                       "source_checkpoint_sha256": source.checkpoint_sha256,
                       "checkpoint_retention": "all"})
        _write_json(status_path, status)
        return summary
    except BaseException as exc:
        status_path = run_root / "artifact" / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        status.update({"state": "failed", "version": version,
                       "failure": f"{type(exc).__name__}: {exc}"})
        _write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    result = run(args.version, args.preset, args.dataset, args.source_dataset,
                 args.source_checkpoint,
                 TrainerConfig(epochs=args.epochs, batch_size=args.batch_size,
                               validation_batch_size=args.validation_batch_size,
                               learning_rate=args.learning_rate), args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["PRESETS", "run"]

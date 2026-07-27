"""Preflight, allocate and train the faithful 0010 control."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from rl_environment.runs import initialize_version

from . import PROJECT_ID
from .config import MODEL_READY_DATASET
from .model import Faithful0010PointerPolicy, IDOnlyConfig, parameter_count
from .card_semantics import CardSemanticRegistry
from .training.a0_trainer import train
from .training.materialized import iter_a0_batches, validate_materialized_dataset


def _batch_count(reference: dict, split: str, batch_size: int) -> int:
    return sum(
        math.ceil(item["a0_eligible"] / batch_size)
        for item in reference["shards"][split]
        if item["a0_eligible"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="V1_0010_faithful_control")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="load a prior checkpoint (model, optimizer and validation criterion)",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tag", default="V0_noncanonical_smoke")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("0014 A0 requires CUDA")
    registry = CardSemanticRegistry.from_official_csv(
        Path("data/official/EN_Card_Data.csv")
    )
    reference = validate_materialized_dataset(MODEL_READY_DATASET, registry=registry)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    model = Faithful0010PointerPolicy(IDOnlyConfig())
    if parameter_count(model) != 7_154_562:
        raise ValueError("A0 parameter count drift")
    if arguments.smoke:
        version = arguments.smoke_tag
        root = Path(".tmp/0014_faithful_board_causal_features") / version
        if root.exists():
            raise FileExistsError(root)
        from rl_environment.runs import project_version_paths

        paths = project_version_paths(PROJECT_ID, "V1_0010_faithful_control")
        paths = type(paths)(
            **{
                **paths.__dict__,
                "version_name": version,
                "run_root": root,
                "artifact": root / "artifact",
                "checkpoints": root / "checkpoint",
                "tensorboard": root / "tensorboard",
                "wandb": root / "wandb",
                "config": root / "artifact/training_config.json",
                "metrics": root / "artifact/training_metrics.jsonl",
                "summary": root / "artifact/training_summary.json",
                "status": root / "artifact/status.json",
                "checkpoint_selection": root / "artifact/checkpoint_selection.json",
                "model_contract": root / "artifact/model_contract.json",
                "dataset_reference": root / "artifact/dataset_reference.json",
                "metrics_snapshot": root / "artifact/metrics_snapshot.json",
                "wandb_snapshot_manifest": root / "artifact/wandb_snapshot_manifest.json",
            }
        )
        for directory in (paths.artifact, paths.checkpoints, paths.tensorboard, paths.wandb):
            directory.mkdir(parents=True, exist_ok=False)
        paths.status.write_text(
            json.dumps({"state": "allocated", "version": version}) + "\n"
        )
        os.environ["WANDB_MODE"] = "disabled"
        epochs, max_train, max_validation = 1, 20, 4
    else:
        paths = initialize_version(PROJECT_ID, arguments.version)
        project_number, project_slug = PROJECT_ID.split("_", 1)
        wandb_name = f"{project_number} · {project_slug} · {arguments.version}"
        os.environ.update(
            {
                "WANDB_MODE": "online",
                "WANDB_ENTITY": "dragon_bra",
                "WANDB_PROJECT": "pokemon-tcg-policy-learning",
                "WANDB_DIR": str(paths.wandb),
                "WANDB_NAME": wandb_name,
                "WANDB_RUN_GROUP": PROJECT_ID,
                "WANDB_JOB_TYPE": "bc",
                "WANDB_TAGS": "0014,a0,faithful_0010",
            }
        )
        epochs, max_train, max_validation = arguments.epochs, None, None
    warm_start_payload = None
    warm_start_best: dict[str, float] | None = None
    start_epoch = 0
    initial_global_step = 0
    if arguments.warm_start is not None:
        if arguments.smoke:
            raise ValueError("--warm-start is not supported for smoke runs")
        if not arguments.warm_start.is_file():
            raise FileNotFoundError(arguments.warm_start)
        # Load optimizer moments on the same device as the policy.  Loading the
        # checkpoint on CPU and moving only the model leaves AdamW's state on
        # CPU, which fails on the first CUDA optimizer step.
        warm_start_payload = torch.load(arguments.warm_start, map_location="cuda")
        if not isinstance(warm_start_payload, dict):
            raise ValueError("warm-start checkpoint must contain a mapping")
        model.load_state_dict(warm_start_payload["model"])
        start_epoch = int(warm_start_payload.get("epoch", 0))
        initial_global_step = int(warm_start_payload.get("global_step", 0))
        metadata = warm_start_payload.get("metadata", {})
        prior_metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
        if isinstance(prior_metrics, dict):
            warm_start_best = {
                "loss": float(prior_metrics.get("bc/validation/loss", math.inf)),
                "teacher_exact": float(
                    prior_metrics.get("bc/validation/teacher_exact_action", -math.inf)
                ),
                "greedy_exact": float(
                    prior_metrics.get("bc/validation/exact_action", -math.inf)
                ),
            }
        if arguments.epochs <= start_epoch:
            raise ValueError(
                f"--epochs ({arguments.epochs}) must exceed warm-start epoch ({start_epoch})"
            )

    config = {
        "schema_version": "0014_a0_training_v1",
        "project_id": PROJECT_ID,
        "version": paths.version_name,
        "dataset_path": str(MODEL_READY_DATASET),
        "dataset_content_sha256": reference["content_sha256"],
        "dataset_counts": reference["a0_eligible_counts"],
        "model": IDOnlyConfig().to_dict(),
        "parameter_count": parameter_count(model),
        "batch_size": arguments.batch_size,
        "validation_batch_size": arguments.validation_batch_size,
        "epochs": epochs,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "warm_start": str(arguments.warm_start) if arguments.warm_start else None,
        "start_epoch": start_epoch,
        "initial_global_step": initial_global_step,
        "early_stopping_patience": arguments.early_stopping_patience,
        "early_stopping_min_delta": arguments.early_stopping_min_delta,
        "seed": arguments.seed,
        "amp_dtype": "bfloat16",
        "grad_scaler": False,
        "train_static_evaluation": False,
        "train_greedy_decode": False,
        "legacy_always_stop_objective": True,
        "option_permutation": False,
        "wandb": {
            "mode": "online" if not arguments.smoke else "disabled",
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "name": (
                f"{PROJECT_ID.split('_', 1)[0]} · {PROJECT_ID.split('_', 1)[1]} · {paths.version_name}"
                if not arguments.smoke
                else None
            ),
            "group": PROJECT_ID,
            "job_type": "bc",
        },
    }
    paths.dataset_reference.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths.model_contract.write_text(
        json.dumps(
            {
                "model": "Faithful0010PointerPolicy",
                "parameter_count": parameter_count(model),
                "auxiliary_readers": False,
                "input_keys": [
                    "global_cat", "global_num", "entity_cat", "entity_num",
                    "entity_mask", "option_cat", "option_mask", "targets",
                    "min_count", "max_count"
                ],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    if warm_start_payload is not None:
        optimizer.load_state_dict(warm_start_payload["optimizer"])
        # torch.load may preserve AdamW's scalar ``step`` on CUDA while its
        # moment buffers remain on CPU.  Normalize every tensor state before
        # the first resumed update.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to("cuda", non_blocking=True)
    result = train(
        paths=paths,
        model=model,
        optimizer=optimizer,
        train_batches=lambda epoch: iter_a0_batches(
            MODEL_READY_DATASET,
            "train",
            batch_size=arguments.batch_size,
            shuffle=True,
            seed=arguments.seed + epoch,
        ),
        validation_batches=lambda epoch: iter_a0_batches(
            MODEL_READY_DATASET,
            "validation",
            batch_size=arguments.validation_batch_size,
            shuffle=False,
            seed=arguments.seed,
        ),
        batch_counts={
            "train": _batch_count(reference, "train", arguments.batch_size),
            "validation": _batch_count(
                reference, "validation", arguments.validation_batch_size
            ),
        },
        epochs=epochs,
        config=config,
        device=torch.device("cuda"),
        amp=True,
        maximum_train_batches=max_train,
        maximum_validation_batches=max_validation,
        start_epoch=start_epoch,
        initial_global_step=initial_global_step,
        initial_best=warm_start_best,
        early_stopping_patience=arguments.early_stopping_patience,
        early_stopping_min_delta=arguments.early_stopping_min_delta,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

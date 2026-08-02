"""Preflight and run the matched James Cox Raging Bolt BC ablation."""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from rl_environment.runs import (
    initialize_version,
    project_version_paths,
    wandb_run_id,
)

from . import PROJECT_ID
from .features.prototypes import PrototypeIndex
from .legacy.base_model import IDOnlyConfig, IDOnlyPointerPolicy
from .model.multi_memory import SemanticFoundationPolicy, SemanticModelConfig
from .training.dataset import SemanticDecisionDataset
from .training.trainer import train_ablation


DATASET_ROOT = Path(
    "rl_runs/0025_semantic_foundation_pretraining/dataset/"
    "V1_james_cox_raging_bolt_semantic"
)
PROTOTYPES = Path(
    "train/0025_semantic_foundation_pretraining/assets/"
    "official_public_prototypes_v1.json"
)


def _smoke_paths():
    version = "V0_noncanonical_smoke"
    root = Path(".tmp/0025_james_cox_bc_smoke") / f"run-{uuid.uuid4().hex[:10]}"
    template = project_version_paths(PROJECT_ID, "V1_semantic_contract_smoke")
    paths = replace(
        template,
        version_name=version,
        run_root=root,
        artifact=root / "artifact",
        checkpoints=root / "checkpoint",
        tensorboard=root / "tensorboard",
        wandb=root / "wandb",
        config=root / "artifact/training_config.json",
        metrics=root / "artifact/training_metrics.jsonl",
        summary=root / "artifact/training_summary.json",
        status=root / "artifact/status.json",
        checkpoint_selection=root / "artifact/checkpoint_selection.json",
        model_contract=root / "artifact/model_contract.json",
        dataset_reference=root / "artifact/dataset_reference.json",
        metrics_snapshot=root / "artifact/metrics_snapshot.json",
        wandb_snapshot_manifest=root / "artifact/wandb_snapshot_manifest.json",
    )
    for directory in (
        paths.artifact,
        paths.checkpoints,
        paths.tensorboard,
        paths.wandb,
    ):
        directory.mkdir(parents=True, exist_ok=False)
    paths.status.write_text(
        json.dumps({"state": "allocated", "version": version}) + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="V2_james_cox_raging_bolt_ablation")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0005)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--maximum-train-batches", type=int)
    parser.add_argument("--maximum-validation-batches", type=int)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.validation_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if not args.smoke and not torch.cuda.is_available():
        raise RuntimeError("formal 0025 BC training requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = SemanticDecisionDataset(DATASET_ROOT)
    prototypes = PrototypeIndex.load(PROTOTYPES)
    if args.smoke:
        paths = _smoke_paths()
        legacy_default_config = IDOnlyConfig(d_model=64, heads=4, encoder_layers=2)
        legacy_matched_config = IDOnlyConfig(d_model=96, heads=4, encoder_layers=2)
        semantic_config = SemanticModelConfig(d_model=64, heads=4)
        epochs = 1
        maximum_train = args.maximum_train_batches or 2
        maximum_validation = args.maximum_validation_batches or 2
        os.environ["WANDB_MODE"] = "disabled"
    else:
        paths = initialize_version(PROJECT_ID, args.version)
        legacy_default_config = IDOnlyConfig(d_model=320, heads=8, encoder_layers=4)
        legacy_matched_config = IDOnlyConfig(d_model=576, heads=8, encoder_layers=4)
        semantic_config = SemanticModelConfig(d_model=320, heads=8, dropout=0.10)
        epochs = args.epochs
        maximum_train = args.maximum_train_batches
        maximum_validation = args.maximum_validation_batches
        os.environ.update(
            {
                "WANDB_MODE": "online",
                "WANDB_ENTITY": "dragon_bra",
                "WANDB_PROJECT": "pokemon-tcg-policy-learning",
                "WANDB_DIR": str(paths.wandb),
                "WANDB_NAME": f"0025 · james_cox_raging_bolt · {paths.version_name}",
                "WANDB_RUN_GROUP": PROJECT_ID,
                "WANDB_RUN_ID": wandb_run_id(PROJECT_ID, paths.version_name),
                "WANDB_JOB_TYPE": "bc",
                "WANDB_TAGS": "0025,james_cox,raging_bolt,semantic_ablation",
            }
        )

    models = {
        "legacy_default": IDOnlyPointerPolicy(legacy_default_config),
        "legacy_budget_matched": IDOnlyPointerPolicy(legacy_matched_config),
        "semantic": SemanticFoundationPolicy(semantic_config, prototypes),
    }
    arm_contracts = {
        "legacy_default": {
            "model": "IDOnlyPointerPolicy",
            "config": legacy_default_config.to_dict(),
            "parameter_count": sum(p.numel() for p in models["legacy_default"].parameters()),
            "actor_view": "legacy_only",
        },
        "legacy_budget_matched": {
            "model": "IDOnlyPointerPolicy",
            "config": legacy_matched_config.to_dict(),
            "parameter_count": sum(
                p.numel() for p in models["legacy_budget_matched"].parameters()
            ),
            "actor_view": "legacy_only",
        },
        "semantic": {
            "model": "SemanticFoundationPolicy",
            "config": asdict(semantic_config),
            "parameter_count": sum(p.numel() for p in models["semantic"].parameters()),
            "actor_view": "legacy_plus_typed_multi_memory",
        },
    }
    config = {
        "schema_version": "0025_james_cox_raging_bolt_ablation_v1",
        "project_id": PROJECT_ID,
        "version": paths.version_name,
        "dataset_path": str(DATASET_ROOT),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_split_counts": dataset.split_counts,
        "expert_team_names": ["James Cox", "James Cox & Henry Chao"],
        "exact_deck_sha256": "f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a",
        "source_identity_actor_visible": False,
        "arms": arm_contracts,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp_dtype": "bfloat16",
        "grad_clip": 1.0,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "train_passes_per_epoch_per_active_arm": 1,
        "validation_full_pass_per_epoch": maximum_validation is None,
        "checkpoint_payload": "model_weights_only_no_optimizer_no_resume_state",
        "checkpoint_retention_slots_per_arm": 4,
        "wandb": {
            "mode": "disabled" if args.smoke else "online",
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": None
            if args.smoke
            else wandb_run_id(PROJECT_ID, paths.version_name),
        },
    }
    paths.dataset_reference.write_text(
        json.dumps(
            {
                "dataset_path": str(DATASET_ROOT),
                "manifest_sha256": dataset.manifest_sha256,
                "manifest": dataset.manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.model_contract.write_text(
        json.dumps(
            {
                "schema_version": "0025_paired_model_contract_v1",
                "arms": arm_contracts,
                "shared_action_contract": "ordered_legal_option_pointer_plus_stop",
                "shared_dataset_split": dataset.manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    optimizers = {
        arm: torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for arm, model in models.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "event": "0025_bc_preflight",
                "run_root": str(paths.run_root),
                "device": str(device),
                "arms": arm_contracts,
                "dataset": dataset.split_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    summary = train_ablation(
        paths=paths,
        models=models,
        optimizers=optimizers,
        dataset=dataset,
        config=config,
        device=device,
        epochs=epochs,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        seed=args.seed,
        amp=True,
        grad_clip=1.0,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        maximum_train_batches=maximum_train,
        maximum_validation_batches=maximum_validation,
    )
    print(
        json.dumps(
            {
                "event": "0025_bc_complete",
                "run_root": str(paths.run_root),
                "best": summary["best"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

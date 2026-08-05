"""Train the 0031 rule-faithful semantic policy on audited official winners."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_environment.runs import (
    initialize_version,
    project_version_paths,
    wandb_run_id,
)

from . import PROJECT_ID
from .domain.prototypes import PrototypeIndex
from .features.collate import BucketPadding
from .model import ModelConfig, SemanticPolicy
from .training.dataset import CanonicalDecisionDataset
from .training.trainer import train_ablation


DATASET_ROOT = Path(
    "rl_runs/0031_rule_faithful_semantic_foundation_pretraining/dataset/"
    "V1_rule_faithful_winners_20260710_20260802"
)
PROTOTYPES = Path(
    "train/0031_rule_faithful_semantic_foundation_pretraining/assets/"
    "official_public_prototypes_v1.json"
)
ARM = "rule_faithful_semantic"


def _implementation_sha256() -> str:
    package = Path(__file__).resolve().parent
    files = [
        Path(__file__).resolve(),
        *(package / "features").glob("*.py"),
        *(package / "model").glob("*.py"),
        *(package / "contracts").glob("*.py"),
        package / "domain/prototypes.py",
        package / "knowledge/ledger.py",
        package / "knowledge/state.py",
        package / "training/dataset.py",
        package / "training/checkpoints.py",
        package / "training/objective.py",
        package / "training/trainer.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(package.parent.parent).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _smoke_paths():
    version = "V0_noncanonical_smoke"
    root = Path(".tmp/0031_rule_faithful_bc_smoke") / f"run-{uuid.uuid4().hex[:10]}"
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
    parser.add_argument(
        "--version", default="V1_rule_faithful_foundation"
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0005)
    parser.add_argument("--prefetch-depth", type=int, default=4)
    parser.add_argument(
        "--fixed-bucket-padding",
        action="store_true",
        help="Pad every ragged family to its finite compile bucket upper bound.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this exact version from checkpoint/resume/latest_resume.pt.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-formal-model",
        action="store_true",
        help="Use the production model size in an untracked smoke run.",
    )
    parser.add_argument("--maximum-train-batches", type=int)
    parser.add_argument("--maximum-validation-batches", type=int)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.validation_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.prefetch_depth < 0:
        raise ValueError("prefetch depth must be nonnegative")
    if args.resume and args.smoke:
        raise ValueError("noncanonical smoke runs do not support in-place resume")
    if not args.smoke and not torch.cuda.is_available():
        raise RuntimeError("formal 0031 rule-faithful BC training requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    fixed_bucket_padding = args.fixed_bucket_padding
    dataset = CanonicalDecisionDataset(
        args.dataset_root,
        bucket_padding=BucketPadding() if fixed_bucket_padding else None,
    )
    prototypes = PrototypeIndex.load(PROTOTYPES)
    if args.smoke:
        paths = _smoke_paths()
        model_config = (
            ModelConfig()
            if args.smoke_formal_model
            else ModelConfig(
                d_model=64,
                heads=4,
                state_layers=2,
                option_layers=2,
                ffn_multiplier=2,
                dropout=0.0,
            )
        )
        epochs = args.epochs
        maximum_train = args.maximum_train_batches or 2
        maximum_validation = args.maximum_validation_batches or 2
        os.environ["WANDB_MODE"] = "disabled"
    else:
        paths = (
            project_version_paths(PROJECT_ID, args.version)
            if args.resume
            else initialize_version(PROJECT_ID, args.version)
        )
        if args.resume and not (
            paths.checkpoints / "resume/latest_resume.pt"
        ).is_file():
            raise FileNotFoundError("0031 exact-resume checkpoint does not exist")
        model_config = ModelConfig()
        epochs = args.epochs
        maximum_train = args.maximum_train_batches
        maximum_validation = args.maximum_validation_batches
        os.environ.update(
            {
                "WANDB_MODE": "online",
                "WANDB_ENTITY": "dragon_bra",
                "WANDB_PROJECT": "pokemon-tcg-policy-learning",
                "WANDB_DIR": str(paths.wandb),
                "WANDB_NAME": f"0031 rule-faithful semantic {paths.version_name}",
                "WANDB_RUN_GROUP": PROJECT_ID,
                "WANDB_RUN_ID": wandb_run_id(PROJECT_ID, paths.version_name),
                "WANDB_JOB_TYPE": "bc",
                "WANDB_TAGS": "0031,rule_faithful,semantic,winner_bc",
                "WANDB_RESUME": "must" if args.resume else "never",
            }
        )

    model = SemanticPolicy(model_config, prototypes)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    arm_contract = {
        "model": "SemanticPolicy",
        "config": model_config.to_dict(),
        "parameter_count": parameter_count,
        "actor_view": "canonical_typed_state_and_option_relations_only",
        "legacy_tensor_dependency": False,
    }
    config = {
        "schema_version": "0031_rule_faithful_bc_training_v1",
        "project_id": PROJECT_ID,
        "version": paths.version_name,
        "dataset_path": str(args.dataset_root),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_split_counts": dataset.split_counts,
        "dataset_interval": ["2026-07-10", "2026-08-02"],
        "winner_perspective_only": True,
        "exact_deck_conditioning": "registered_60_card_multiset",
        "source_count": len(dataset.manifest.get("sources", [])),
        "exact_deck_count": int(dataset.manifest.get("exact_deck_count", 0)),
        "source_identity_actor_visible": False,
        "implementation_sha256": _implementation_sha256(),
        "initialized_from_checkpoint": None,
        "arms": {ARM: arm_contract},
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
        "train_passes_per_epoch": 1,
        "validation_full_pass_per_epoch": maximum_validation is None,
        "length_bucketed_train_batches": True,
        "fixed_bucket_padding": fixed_bucket_padding,
        "fixed_bucket_bounds": {
            name: list(values) for name, values in BucketPadding().__dict__.items()
        }
        if fixed_bucket_padding
        else None,
        "torch_compile": False,
        "torch_compile_status": "not_admitted_pending_bounded_compile_benchmark",
        "optimizer_fused": torch.cuda.is_available(),
        "prefetch_depth": args.prefetch_depth,
        "checkpoint_payload": "all_epoch_model_only_plus_four_aliases_and_one_exact_resume_slot",
        "checkpoint_retention": "all",
        "model_only_checkpoint_retention_slots": "all_completed_epochs_plus_four_aliases",
        "training_state_checkpoint_retention_slots": 1,
        "training_state_checkpoint_boundary": "completed_epoch",
        "wandb": {
            "mode": "disabled" if args.smoke else "online",
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": None
            if args.smoke
            else wandb_run_id(PROJECT_ID, paths.version_name),
        },
    }
    dataset_reference = (
        json.dumps(
            {
                "dataset_path": str(args.dataset_root),
                "manifest_sha256": dataset.manifest_sha256,
                "manifest": dataset.manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    model_contract = (
        json.dumps(
            {
                "schema_version": "0031_rule_faithful_model_contract_v1",
                "arm": arm_contract,
                "action_contract": "ordered_legal_option_pointer_plus_stop",
                "dataset_manifest_sha256": dataset.manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.resume:
        if paths.dataset_reference.read_text(encoding="utf-8") != dataset_reference:
            raise ValueError("resume dataset reference does not match the allocated run")
        if paths.model_contract.read_text(encoding="utf-8") != model_contract:
            raise ValueError("resume model contract does not match the allocated run")
    else:
        paths.dataset_reference.write_text(dataset_reference, encoding="utf-8")
        paths.model_contract.write_text(model_contract, encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    print(
        json.dumps(
            {
                "event": "0031_rule_faithful_bc_preflight",
                "run_root": str(paths.run_root),
                "device": str(device),
                "parameter_count": parameter_count,
                "dataset": dataset.split_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    summary = train_ablation(
        paths=paths,
        models={ARM: model},
        optimizers={ARM: optimizer},
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
        prefetch_depth=args.prefetch_depth,
        resume_checkpoint=(paths.checkpoints / "resume/latest_resume.pt")
        if args.resume
        else None,
    )
    print(
        json.dumps(
            {
                "event": "0031_rule_faithful_bc_complete",
                "run_root": str(paths.run_root),
                "best": summary["best"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

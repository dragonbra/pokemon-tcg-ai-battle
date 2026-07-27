"""Train one 0015 source-conditioned R15-family BC arm."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rl_environment.runs import initialize_version, project_version_paths

from . import PROJECT_ID
from .initialization import freeze_to_new_model_keys, load_r15_base_checkpoint
from .model import SourceConditionedR15Config, SourceConditionedR15Policy
from .rule_contract_model import (
    SourceConditionedR15RuleConfig,
    SourceConditionedR15RulePolicy,
)
from .training.data import decision_count, iter_batches, load_reference
from .training.outcome_weighted_trainer import make_optimization_step

_trainer = importlib.import_module(
    "train.0014_faithful_board_causal_features.training.a0_trainer"
)

DEFAULT_DATASET = Path(
    "rl_runs/0015_dragapult_conditioned_bc/dataset/V1_core_r15_features"
)
MODEL_FAMILIES = ("r15", "r15_rule_contract")


def build_policy(
    *,
    model_family: str,
    source_vocabulary_size: int,
    rule_initial_scale: float,
    rule_model_width: int,
    rule_heads: int,
    ontology_path: Path,
    source_initial_scale: float = 0.10,
) -> tuple[torch.nn.Module, object]:
    """Build an explicit model family without changing the R15 default path."""

    if model_family == "r15":
        config = SourceConditionedR15Config(
            source_vocabulary_size=source_vocabulary_size,
            source_initial_scale=source_initial_scale,
        )
        return SourceConditionedR15Policy(config, ontology_path=ontology_path), config
    if model_family == "r15_rule_contract":
        config = SourceConditionedR15RuleConfig(
            source_vocabulary_size=source_vocabulary_size,
            source_initial_scale=source_initial_scale,
            rule_initial_scale=rule_initial_scale,
            rule_model_width=rule_model_width,
            rule_heads=rule_heads,
        )
        return SourceConditionedR15RulePolicy(config, ontology_path=ontology_path), config
    raise ValueError(f"unknown model family: {model_family}")


def _smoke_paths(version: str):
    root = Path(".tmp/0015_dragapult_conditioned_bc") / version
    if root.exists():
        raise FileExistsError(root)
    template = project_version_paths(PROJECT_ID, "V1_placeholder")
    paths = type(template)(
        **{
            **template.__dict__,
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
    paths.status.write_text(json.dumps({"state": "allocated", "version": version}) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one 0015 R15-family BC arm")
    parser.add_argument("--arm", choices=("t0", "t1", "t2", "t3", "t4"), required=True)
    parser.add_argument("--version")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--win-weight", type=float, default=1.0)
    parser.add_argument("--loss-weight", type=float, default=1.0)
    parser.add_argument("--draw-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--source-initial-scale", type=float, default=0.10)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, default="r15")
    parser.add_argument("--rule-initial-scale", type=float, default=0.10)
    parser.add_argument("--rule-model-width", type=int, default=320)
    parser.add_argument("--rule-heads", type=int, default=8)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--train-rule-only", action="store_true")
    parser.add_argument("--no-deck", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("0015 R15 requires CUDA")
    if args.no_deck and args.arm not in {"t1", "t3"}:
        raise ValueError("--no-deck is defined only for --arm t1 or t3")
    if args.model_family == "r15" and not math.isclose(args.rule_initial_scale, 0.10):
        raise ValueError("--rule-initial-scale applies only to r15_rule_contract")
    if args.model_family == "r15" and (
        args.rule_model_width != 320 or args.rule_heads != 8
    ):
        raise ValueError("rule bottleneck options apply only to r15_rule_contract")
    if args.initial_checkpoint is not None and args.model_family != "r15_rule_contract":
        raise ValueError("--initial-checkpoint is defined only for r15_rule_contract")
    if args.train_rule_only and args.initial_checkpoint is None:
        raise ValueError("--train-rule-only requires --initial-checkpoint")
    outcome_weighting = any(
        not math.isclose(value, 1.0)
        for value in (args.win_weight, args.loss_weight, args.draw_weight)
    )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.win_weight, args.loss_weight, args.draw_weight)
    ):
        raise ValueError("outcome weights must be positive finite")
    if not math.isfinite(args.label_smoothing) or not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be finite and in [0, 1)")
    if outcome_weighting and args.arm != "t1":
        raise ValueError("formal outcome weighting is currently defined only for --arm t1")
    reference = load_reference(args.dataset)
    train_decisions = decision_count(args.dataset, "train", args.arm)
    validation_decisions = decision_count(args.dataset, "validation", args.arm)
    if not train_decisions or not validation_decisions:
        raise ValueError(f"arm {args.arm} has an empty train or target-validation view")
    if args.arm in {"t2", "t3", "t4"}:
        expected = {"t2": 3, "t3": 3, "t4": 4}[args.arm]
        present = {
            int(build_id)
            for item in reference["shards"]["train"]
            for build_id, count in item["eligible_by_build_id"].items()
            if count
        }
        if expected not in present and not (args.smoke and args.arm == "t3" and args.no_deck):
            raise ValueError(f"arm {args.arm} source build {expected} is not materialized")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, model_config = build_policy(
        model_family=args.model_family,
        source_vocabulary_size=max(reference["source_vocabulary"].values()),
        rule_initial_scale=args.rule_initial_scale,
        rule_model_width=args.rule_model_width,
        rule_heads=args.rule_heads,
        ontology_path=args.dataset / "card_ontology.json",
        source_initial_scale=args.source_initial_scale,
    )
    initialization = (
        load_r15_base_checkpoint(model, args.initial_checkpoint)
        if args.initial_checkpoint is not None
        else None
    )
    training_scope = (
        freeze_to_new_model_keys(model, initialization["new_model_keys"])
        if args.train_rule_only and initialization is not None
        else None
    )
    parameters = sum(value.numel() for value in model.parameters())
    trainable_parameters = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )

    if args.smoke:
        version = args.version or f"smoke_{args.arm}{'_no_deck' if args.no_deck else ''}"
        paths = _smoke_paths(version)
        os.environ["WANDB_MODE"] = "disabled"
        epochs, maximum_train, maximum_validation = 1, 5, 2
    else:
        if not args.version:
            raise ValueError("formal training requires --version")
        paths = initialize_version(PROJECT_ID, args.version)
        os.environ.update(
            {
                "WANDB_MODE": "online",
                "WANDB_ENTITY": "dragon_bra",
                "WANDB_PROJECT": "pokemon-tcg-policy-learning",
                "WANDB_DIR": str(paths.wandb),
                "WANDB_NAME": f"0015 · dragapult conditioned BC · {args.version}",
                "WANDB_RUN_GROUP": PROJECT_ID,
                "WANDB_JOB_TYPE": "bc",
                "WANDB_TAGS": (
                    f"0015,{args.model_family},{args.arm},source_conditioned"
                    + (",outcome_weighted" if outcome_weighting else "")
                    + (",label_smoothed" if args.label_smoothing > 0.0 else "")
                    + (",warm_started" if initialization is not None else "")
                    + (",rule_only" if training_scope is not None else "")
                ),
            }
        )
        epochs, maximum_train, maximum_validation = args.epochs, None, None

    config = {
        "schema_version": (
            "0015_r15_source_conditioned_training_v1"
            if args.model_family == "r15"
            else "0015_r15_rule_contract_training_v1"
        ),
        "project_id": PROJECT_ID,
        "version": paths.version_name,
        "arm": args.arm,
        "no_deck": args.no_deck,
        "outcome_weighting": {
            "enabled": outcome_weighting,
            "win_weight": args.win_weight,
            "loss_weight": args.loss_weight,
            "draw_weight": args.draw_weight,
            "scope": "training_loss_only",
            "validation": "unweighted",
        },
        "label_smoothing": {
            "epsilon": args.label_smoothing,
            "scope": "training_loss_only",
            "validation": "unsmoothed",
        },
        "dataset": str(args.dataset),
        "dataset_content_sha256": reference["content_sha256"],
        "train_decisions": train_decisions,
        "target_validation_decisions": validation_decisions,
        "model_family": args.model_family,
        "model": model_config.to_dict(),
        "initialization": initialization,
        "training_scope": training_scope,
        "parameter_count": parameters,
        "trainable_parameter_count": trainable_parameters,
        "r15_baseline": "0014/V25_r15_deterministic_gradual_option",
        "fresh_initialization": initialization is None,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "epochs": epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp_dtype": "bfloat16",
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "wandb": {
            "mode": "disabled" if args.smoke else "online",
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
        },
    }
    paths.dataset_reference.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    paths.model_contract.write_text(
        json.dumps(
            {
                "model": type(model).__name__,
                "model_family": args.model_family,
                "parameter_count": parameters,
                "r15_config": asdict(model_config.r15),
                "model_config": model_config.to_dict(),
                "initialization": initialization,
                "training_scope": training_scope,
                "rule_contract": (
                    {
                        "role_names": list(model.rule_role_names),
                        "token_widths": list(model.rule_token_widths),
                        "initial_scale": model_config.rule_initial_scale,
                    }
                    if isinstance(model, SourceConditionedR15RulePolicy)
                    else None
                ),
                "source_vocabulary": reference["source_vocabulary"],
                "target_source_id": reference["source_vocabulary"]["third_ptcg_club"],
                "no_deck_contract": (
                    "visible-live-ledger-only-with-safe-sentinel" if args.no_deck else None
                ),
                "outcome_weighting": config["outcome_weighting"],
                "label_smoothing": config["label_smoothing"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    result = _trainer.train(
        paths=paths,
        model=model,
        optimizer=optimizer,
        train_batches=lambda epoch: iter_batches(
            args.dataset,
            "train",
            arm=args.arm,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            no_deck=args.no_deck,
        ),
        validation_batches=lambda epoch: iter_batches(
            args.dataset,
            "validation",
            arm=args.arm,
            batch_size=args.validation_batch_size,
            shuffle=False,
            seed=args.seed,
            no_deck=args.no_deck,
        ),
        batch_counts={
            "train": math.ceil(train_decisions / args.batch_size),
            "validation": math.ceil(validation_decisions / args.validation_batch_size),
        },
        epochs=epochs,
        config=config,
        device=torch.device("cuda"),
        amp=True,
        maximum_train_batches=maximum_train,
        maximum_validation_batches=maximum_validation,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        optimization_step=(
            make_optimization_step(
                win_weight=args.win_weight,
                loss_weight=args.loss_weight,
                draw_weight=args.draw_weight,
                label_smoothing=args.label_smoothing,
            )
            if outcome_weighting or args.label_smoothing > 0.0
            else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

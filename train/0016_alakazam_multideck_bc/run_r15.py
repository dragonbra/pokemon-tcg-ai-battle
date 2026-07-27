"""Preflight, allocate, resume and train the 0016 R15 gradual-option policy."""
from __future__ import annotations

import argparse
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
from .card_semantics import CardSemanticRegistry
from .config import AC_SWITCHES, MODEL_READY_DATASET
from .r15_model import R15ModelConfig
from .source_model import SourceModelConfig
from .source_r15_model import SourceConditionedR15Policy
from .training.a0_trainer import train
from .training.ac_data import iter_ac_batches
from .training.materialized import validate_materialized_dataset
from .training.resume import load_resume_context, restore_rng_state


def _batch_count(reference: dict, split: str, batch_size: int) -> int:
    return sum(
        math.ceil(item["a0_eligible"] / batch_size)
        for item in reference["shards"][split]
        if item["a0_eligible"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="V2_win_multideck_r15_gradual_option")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tag", default="V0_r15_noncanonical_smoke")
    parser.add_argument("--resume-checkpoint", type=Path)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("0016 source-conditioned R15 requires CUDA")

    registry = CardSemanticRegistry.from_official_csv(
        Path("data/official/EN_Card_Data.csv")
    )
    reference = validate_materialized_dataset(MODEL_READY_DATASET, registry=registry)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    model_config = R15ModelConfig()
    source_count = len(reference["sources"])
    source_config = SourceModelConfig(vocabulary_size=source_count)
    model = SourceConditionedR15Policy(
        model_config,
        source_config,
        ontology_path=MODEL_READY_DATASET / "card_ontology.json",
    )

    resume_context = None
    if arguments.resume_checkpoint is not None and arguments.smoke:
        raise ValueError("--resume-checkpoint is not supported for smoke runs")
    if arguments.smoke:
        version = arguments.smoke_tag
        root = Path(".tmp/0016_alakazam_multideck_bc") / version
        if root.exists():
            raise FileExistsError(root)
        template = project_version_paths(PROJECT_ID, "V1_0010_faithful_control")
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
        for directory in (
            paths.artifact,
            paths.checkpoints,
            paths.tensorboard,
            paths.wandb,
        ):
            directory.mkdir(parents=True, exist_ok=False)
        paths.status.write_text(json.dumps({"state": "allocated", "version": version}) + "\n")
        os.environ["WANDB_MODE"] = "disabled"
        epochs, max_train, max_validation = 1, 20, 4
    elif arguments.resume_checkpoint is not None:
        paths = project_version_paths(PROJECT_ID, arguments.version)
        if not all(
            path.is_dir()
            for path in (paths.artifact, paths.checkpoints, paths.tensorboard, paths.wandb)
        ):
            raise FileNotFoundError("resume version directories are incomplete")
        if paths.summary.exists() or paths.checkpoint_selection.exists():
            raise FileExistsError("completed training version cannot be resumed")
        epochs, max_train, max_validation = arguments.epochs, None, None
    else:
        paths = initialize_version(PROJECT_ID, arguments.version)
        project_number, project_slug = PROJECT_ID.split("_", 1)
        os.environ.update(
            {
                "WANDB_MODE": "online",
                "WANDB_ENTITY": "dragon_bra",
                "WANDB_PROJECT": "pokemon-tcg-policy-learning",
                "WANDB_DIR": str(paths.wandb),
                "WANDB_NAME": f"{project_number} · {project_slug} · {arguments.version}",
                "WANDB_RUN_GROUP": PROJECT_ID,
                "WANDB_JOB_TYPE": "bc",
                "WANDB_TAGS": "0016,r15,multideck,source_conditioned,gradual_option",
            }
        )
        epochs, max_train, max_validation = arguments.epochs, None, None

    if not arguments.smoke:
        project_number, project_slug = PROJECT_ID.split("_", 1)
        os.environ.update(
            {
                "WANDB_MODE": "online",
                "WANDB_ENTITY": "dragon_bra",
                "WANDB_PROJECT": "pokemon-tcg-policy-learning",
                "WANDB_DIR": str(paths.wandb),
                "WANDB_NAME": f"{project_number} · {project_slug} · {arguments.version}",
                "WANDB_RUN_GROUP": PROJECT_ID,
                "WANDB_JOB_TYPE": "bc",
                "WANDB_TAGS": "0016,r15,multideck,source_conditioned,gradual_option",
            }
        )

    config = {
        "schema_version": "0016_source_r15_training_v1",
        "project_id": PROJECT_ID,
        "version": paths.version_name,
        "dataset_path": str(MODEL_READY_DATASET),
        "dataset_content_sha256": reference["content_sha256"],
        "dataset_counts": reference["a0_eligible_counts"],
        "ontology_file_sha256": reference["ontology_file_sha256"],
        "feature_switches": asdict(AC_SWITCHES),
        "model": model_config.to_dict(),
        "source_model": asdict(source_config),
        "deployment_source_id": 0,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": arguments.batch_size,
        "validation_batch_size": arguments.validation_batch_size,
        "epochs": epochs,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "seed": arguments.seed,
        "amp_dtype": "bfloat16",
        "grad_scaler": False,
        "train_static_evaluation": False,
        "train_greedy_decode": False,
        "legacy_always_stop_objective": True,
        "option_permutation": False,
        "early_stopping_patience": arguments.early_stopping_patience,
        "early_stopping_min_delta": arguments.early_stopping_min_delta,
        "early_stopping_metric": "bc/validation/loss",
        "checkpoint_candidates": [
            "best_validation_loss",
            "best_teacher_exact",
            "best_greedy_exact",
            "latest",
        ],
        "wandb": {
            "mode": "online" if not arguments.smoke else "disabled",
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "name": (
                f"{PROJECT_ID.split('_', 1)[0]} · {PROJECT_ID.split('_', 1)[1]} · "
                f"{paths.version_name}"
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
                "model": "SourceConditionedR15Policy",
                "backbone": "R15DeterministicGradualOptionPolicy",
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "backbone_parameter_count": 17_387_842,
                "source_model": asdict(source_config),
                "deployment_source_id": 0,
                "base_parameter_count": 7_154_562,
                "feature_switches": asdict(AC_SWITCHES),
                "goal_roles": list(model.goal_role_names),
                "scenario_families": [
                    "public_zone_inventory",
                    "registered_deck_causal_ledger",
                    "causal_event_memory",
                    "known_unknown_opponent_hand",
                ],
                "scenario_scale_gate": {
                    "formula": "2 * sigmoid(MLP(actor_visible_scenario))",
                    "range": [0.0, 2.0],
                    "state_initial_scale": 1.0,
                    "option_initial_scale": model_config.option_initial_scale,
                    "granularity": "per_sample_per_option_per_channel",
                    "independent_of_goal_qkv": True,
                },
                "r15_single_variable": {
                    "state_initial_scale": 1.0,
                    "option_initial_scale": model_config.option_initial_scale,
                    "v1_option_initial_scale": 1.0,
                    "all_other_model_and_training_contracts_unchanged": True,
                },
                "goal_qkv": {
                    "roles": list(model.goal_role_names),
                    "resource_memory": "registered_card_ledger_tokens",
                    "separate_from_scale_gate": True,
                },
                "input_keys": [
                    "global_cat", "global_num", "entity_cat", "entity_num",
                    "entity_mask", "option_cat", "option_mask", "targets",
                    "min_count", "max_count", "zone_inventory_num",
                    "registered_card_ids", "registered_multiplicity",
                    "registered_mask", "ledger_cat", "ledger_num", "ledger_mask",
                    "event_cat", "event_num", "event_mask",
                    "known_opponent_hand_card_ids", "known_opponent_hand_mask",
                    "unknown_opponent_hand_count",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    rng_state_restored = False
    if arguments.resume_checkpoint is not None:
        resume_context = load_resume_context(
            arguments.resume_checkpoint,
            checkpoint_root=paths.checkpoints,
            metrics_path=paths.metrics,
            expected_version=paths.version_name,
            expected_dataset_sha256=reference["content_sha256"],
            early_stopping_min_delta=arguments.early_stopping_min_delta,
        )
        model.load_state_dict(resume_context.payload["model"], strict=True)
        optimizer.load_state_dict(resume_context.payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to("cuda", non_blocking=True)
        rng_state_restored = restore_rng_state(resume_context.payload)
        prior_config = resume_context.payload["metadata"]
        for key in (
            "batch_size",
            "validation_batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "seed",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "model",
            "source_model",
            "feature_switches",
        ):
            if prior_config.get(key) != config.get(key):
                raise ValueError(f"resume training contract changed: {key}")
        config["resume"] = {
            "checkpoint": str(arguments.resume_checkpoint),
            "checkpoint_sha256": resume_context.checkpoint_sha256,
            "start_epoch": resume_context.start_epoch,
            "initial_global_step": resume_context.global_step,
            "rng_state_restored": rng_state_restored,
            "interrupted_epoch_replayed_from_start": resume_context.start_epoch + 1,
        }
    result = train(
        paths=paths,
        model=model,
        optimizer=optimizer,
        train_batches=lambda epoch: iter_ac_batches(
            MODEL_READY_DATASET,
            "train",
            batch_size=arguments.batch_size,
            shuffle=True,
            seed=arguments.seed + epoch,
        ),
        validation_batches=lambda epoch: iter_ac_batches(
            MODEL_READY_DATASET,
            "validation",
            batch_size=arguments.validation_batch_size,
            shuffle=False,
            seed=arguments.seed,
        ),
        batch_counts={
            "train": _batch_count(reference, "train", arguments.batch_size),
            "validation": _batch_count(reference, "validation", arguments.validation_batch_size),
        },
        epochs=epochs,
        config=config,
        device=torch.device("cuda"),
        amp=True,
        maximum_train_batches=max_train,
        maximum_validation_batches=max_validation,
        start_epoch=resume_context.start_epoch if resume_context is not None else 0,
        initial_global_step=resume_context.global_step if resume_context is not None else 0,
        initial_best=resume_context.best if resume_context is not None else None,
        initial_no_loss_improvement=(
            resume_context.no_loss_improvement if resume_context is not None else 0
        ),
        initial_progress_iteration=(
            resume_context.progress_iteration if resume_context is not None else 0
        ),
        initial_history=resume_context.history if resume_context is not None else (),
        early_stopping_patience=arguments.early_stopping_patience,
        early_stopping_min_delta=arguments.early_stopping_min_delta,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

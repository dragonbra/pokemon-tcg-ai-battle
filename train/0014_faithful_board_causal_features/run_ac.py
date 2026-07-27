"""Preflight, allocate and train the 0014 all-feature causal policy."""
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

from rl_environment.runs import initialize_version

from . import PROJECT_ID
from .ac_model import ACModelConfig, AllFeatureCausalPolicy, parameter_count
from .card_semantics import CardSemanticRegistry
from .config import AC_SWITCHES, MODEL_READY_DATASET
from .training.a0_trainer import train
from .training.ac_data import iter_ac_batches
from .training.materialized import validate_materialized_dataset


def _batch_count(reference: dict, split: str, batch_size: int) -> int:
    return sum(
        math.ceil(item["a0_eligible"] / batch_size)
        for item in reference["shards"][split]
        if item["a0_eligible"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="V9_ac_all_feature_causal_goal_qkv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tag", default="V0_ac_noncanonical_smoke")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("0014 AC requires CUDA")

    registry = CardSemanticRegistry.from_official_csv(
        Path("data/official/EN_Card_Data.csv")
    )
    reference = validate_materialized_dataset(MODEL_READY_DATASET, registry=registry)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)

    model_config = ACModelConfig()
    model = AllFeatureCausalPolicy(
        model_config,
        ontology_path=MODEL_READY_DATASET / "card_ontology.json",
    )
    if arguments.smoke:
        version = arguments.smoke_tag
        root = Path(".tmp/0014_faithful_board_causal_features") / version
        if root.exists():
            raise FileExistsError(root)
        from rl_environment.runs import project_version_paths

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
        paths.status.write_text(
            json.dumps({"state": "allocated", "version": version}) + "\n"
        )
        os.environ["WANDB_MODE"] = "disabled"
        epochs, max_train, max_validation = 1, 20, 4
    else:
        paths = initialize_version(PROJECT_ID, arguments.version)
        project_number, project_slug = PROJECT_ID.split("_", 1)
        os.environ.update(
            {
                "WANDB_MODE": "online",
                "WANDB_ENTITY": "dragon_bra",
                "WANDB_PROJECT": "pokemon-tcg-policy-learning",
                "WANDB_DIR": str(paths.wandb),
                "WANDB_NAME": (
                    f"{project_number} · {project_slug} · {arguments.version}"
                ),
                "WANDB_RUN_GROUP": PROJECT_ID,
                "WANDB_JOB_TYPE": "bc",
                "WANDB_TAGS": "0014,ac,all_features,causal_goal_qkv",
            }
        )
        epochs, max_train, max_validation = arguments.epochs, None, None

    config = {
        "schema_version": "0014_ac_training_v1",
        "project_id": PROJECT_ID,
        "version": paths.version_name,
        "dataset_path": str(MODEL_READY_DATASET),
        "dataset_content_sha256": reference["content_sha256"],
        "dataset_counts": reference["a0_eligible_counts"],
        "ontology_file_sha256": reference["ontology_file_sha256"],
        "feature_switches": asdict(AC_SWITCHES),
        "model": model_config.to_dict(),
        "parameter_count": parameter_count(model),
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
        "wandb": {
            "mode": "online" if not arguments.smoke else "disabled",
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "name": (
                f"{PROJECT_ID.split('_', 1)[0]} · "
                f"{PROJECT_ID.split('_', 1)[1]} · {paths.version_name}"
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
                "model": "AllFeatureCausalPolicy",
                "parameter_count": parameter_count(model),
                "base_parameter_count": 7_154_562,
                "feature_switches": asdict(AC_SWITCHES),
                "goal_roles": list(model.goal_role_names),
                "zero_initialized_gates": [
                    "entity_semantic_gate",
                    "global_zone_gate",
                    "state_aux_gate",
                    "option_semantic_gate",
                    "option_aux_gate",
                ],
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
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
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
        early_stopping_patience=arguments.early_stopping_patience,
        early_stopping_min_delta=arguments.early_stopping_min_delta,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

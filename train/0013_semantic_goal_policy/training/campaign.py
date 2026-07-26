"""Executable smoke and formal BC campaign for the M0-M5 ladder."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rl_environment.runs import initialize_version
from .. import PROJECT_ID
from ..features.compiler import COMPILER_VERSION, compiler_sha256
from ..model.decoder import sample_action
from ..model.registry import MODEL_REGISTRY, create_model
from ..protocol import PreRunProtocol
from .batching import default_registry
from .materialized import MaterializedBatchSource
from .reproducibility import model_state_sha256, seed_everything
from .trainer import train_version

REPOSITORY_ROOT = Path(__file__).parents[3]
DEFAULT_DATASET = (
    REPOSITORY_ROOT / "rl_runs" / PROJECT_ID / "dataset" / "V3_model_ready_semantic_v2"
)
PROTOCOL_PATH = Path(__file__).parents[1] / "configs" / "pre_run_protocol.json"


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def create_batch_source(
    dataset: Path,
    *,
    batch_size: int,
    seed: int,
) -> MaterializedBatchSource:
    return MaterializedBatchSource(
        dataset,
        registry=default_registry(),
        batch_size=batch_size,
        seed=seed,
    )


def measure_m0_runtime(
    source: MaterializedBatchSource,
    *,
    device: torch.device,
    decisions: int = 256,
) -> dict[str, float]:
    model = create_model("M0").to(device).eval()
    completed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for source_batch in source.batches("validation", epoch=0, training=False):
            batch = {key: value.to(device) for key, value in source_batch.items()}
            for row, scorer in enumerate(model.action_scorers(batch)):
                sample_action(
                    scorer,
                    option_mask=batch["option_mask"][row],
                    min_count=int(batch["min_count"][row]),
                    max_count=int(batch["max_count"][row]),
                    deterministic=True,
                )
                completed += 1
                if completed >= decisions:
                    elapsed = time.perf_counter() - started
                    return {
                        "decisions": float(completed),
                        "seconds": elapsed,
                        "decisions_per_second": completed / elapsed,
                    }
    raise ValueError("validation split did not contain enough smoke decisions")


def run_formal_training(args: argparse.Namespace) -> dict[str, Any]:
    protocol = PreRunProtocol.from_json(PROTOCOL_PATH)
    if args.protocol_sha256 != protocol.sha256():
        raise ValueError("protocol hash mismatch")
    authorized_variants = protocol.data["minimum_formal_allocation"]["models"]
    if args.variant not in authorized_variants:
        raise ValueError(
            f"variant {args.variant} is not authorized by the bound formal protocol"
        )
    runtime_floor = protocol.resolve_runtime_floor(args.m0_smoke_throughput)
    reproducibility = seed_everything(args.seed)
    source = create_batch_source(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if args.variant != "M0" and source.reference["feature_compiler_version"] != COMPILER_VERSION:
        raise ValueError("semantic variants require the frozen shared compiler")
    model = create_model(args.variant)
    initial_model_sha256 = model_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    device = torch.device(args.device)
    paths = initialize_version(PROJECT_ID, args.version)
    dataset_reference = dict(source.reference)
    wandb_tags = tuple(tag.strip() for tag in args.wandb_tags.split(",") if tag.strip())
    wandb_name = f"0013 · semantic_goal_policy · {args.version}"
    if not wandb_tags:
        raise ValueError("at least one W&B tag is required")
    wandb_notes = args.wandb_notes.strip()
    if not wandb_notes:
        raise ValueError("formal W&B notes must not be empty")
    _write_exclusive(paths.dataset_reference, dataset_reference)
    _write_exclusive(paths.model_contract, {
        "schema_version": "semantic_goal_model_contract_v1",
        "variant": args.variant,
        "model_config": asdict(model.config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_model_sha256": initial_model_sha256,
        "reproducibility": reproducibility,
        "feature_compiler_version": COMPILER_VERSION,
        "feature_compiler_sha256": compiler_sha256(),
        "ontology_sha256": source.registry.sha256,
        "dataset_content_sha256": source.reference["content_sha256"],
        "materializer_sha256": source.reference["materializer_sha256"],
        "feature_dataset_schema_version": source.reference["schema_version"],
        "protocol_sha256": protocol.sha256(),
        "action_contract": "ordered_full_action_v1",
        "value_objective_active": False,
    })
    config = {
        "project_id": PROJECT_ID,
        "version": args.version,
        "variant": args.variant,
        "dataset": str(args.dataset),
        "dataset_content_sha256": source.reference["content_sha256"],
        "materializer_sha256": source.reference["materializer_sha256"],
        "feature_dataset_schema_version": source.reference["schema_version"],
        "protocol_sha256": protocol.sha256(),
        "feature_compiler_sha256": compiler_sha256(),
        "ontology_sha256": source.registry.sha256,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "batch_counts": {
            split: source.batch_count(split) for split in ("train", "validation")
        },
        "seed": args.seed,
        "initial_model_sha256": initial_model_sha256,
        "reproducibility": reproducibility,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "amp": not args.no_amp,
        "device": str(device),
        "m0_smoke_throughput": args.m0_smoke_throughput,
        "runtime_floor": runtime_floor,
        "wandb": {
            "mode": "online",
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "name": wandb_name,
            "group": PROJECT_ID,
            "job_type": "bc_train",
            "tags": list(wandb_tags),
            "notes": wandb_notes,
        },
    }
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_ENTITY"] = "dragon_bra"
    os.environ["WANDB_PROJECT"] = "pokemon-tcg-policy-learning"
    os.environ["WANDB_DIR"] = str(paths.wandb)
    os.environ["WANDB_NAME"] = wandb_name
    os.environ["WANDB_RUN_GROUP"] = PROJECT_ID
    os.environ["WANDB_JOB_TYPE"] = "bc_train"
    os.environ["WANDB_TAGS"] = ",".join(wandb_tags)
    os.environ["WANDB_NOTES"] = wandb_notes
    return train_version(
        paths,
        model=model,
        optimizer=optimizer,
        train_batches=lambda epoch: source.batches("train", epoch=epoch, training=True),
        evaluation_batches=lambda split, epoch: source.batches(
            split, epoch=epoch, training=False
        ),
        batch_counts={
            split: source.batch_count(split) for split in ("train", "validation")
        },
        epochs=args.epochs,
        config=config,
        device=device,
        max_grad_norm=args.max_grad_norm,
        amp=not args.no_amp,
        progress_log_every=args.progress_log_every,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="0013 record-backed BC campaign")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    smoke.add_argument("--batch-size", type=int, default=64)
    smoke.add_argument("--seed", type=int, default=20260726)
    smoke.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    smoke.add_argument("--decisions", type=int, default=256)
    train = subparsers.add_parser("train")
    train.add_argument("--protocol-sha256", required=True)
    train.add_argument("--m0-smoke-throughput", type=float, required=True)
    train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--version", required=True)
    train.add_argument("--variant", choices=tuple(MODEL_REGISTRY), default="M0")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--progress-log-every", type=int, default=20)
    train.add_argument(
        "--wandb-tags",
        default="0013,m0,model_ready",
        help="comma-separated W&B run tags",
    )
    train.add_argument(
        "--wandb-notes",
        default="0013 正式 BC 训练；具体数据、模型和优化合同见本次 run config。",
    )
    train.add_argument("--seed", type=int, default=20260726)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        source = create_batch_source(
            args.dataset,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(json.dumps(measure_m0_runtime(
            source, device=torch.device(args.device), decisions=args.decisions
        ), sort_keys=True))
        return 0
    summary = run_formal_training(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

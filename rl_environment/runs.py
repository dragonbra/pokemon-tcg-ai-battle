"""Helpers for allocating tracked, globally numbered RL experiments."""

from __future__ import annotations

import re
import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPOSITORY_ROOT / "rl_runs"
EXPERIMENT_ROOT = RUNS_ROOT / "artifact"
CHECKPOINT_ROOT = RUNS_ROOT / "checkpoint"
NUMBERED_ARTIFACT = re.compile(r"^(?P<number>\d{4})-(?P<label>.+)$")
EXPERIMENT_DIR = re.compile(r"^(?P<number>\d{4})-(?P<label>[a-z0-9][a-z0-9_-]*)$")
VERSIONED_ATTEMPT = re.compile(r"^V(?P<version>[1-9]\d*)_(?P<tag>[a-z0-9][a-z0-9_]*)$")
WANDB_PROJECT = "pokemon-tcg-policy-learning"
WANDB_METRIC_SCHEMA = "ptcg_tracking_v1"


@dataclass(frozen=True)
class TrainingPaths:
    """Separated paths for one training run."""

    run: Path
    checkpoints: Path
    tensorboard: Path
    config: Path
    metrics: Path
    summary: Path


def wandb_run_id(experiment_id: str, version_name: str) -> str:
    """Return a stable W&B-safe ID for one immutable repository version."""
    identity = f"{experiment_id}--{version_name}"
    normalized = re.sub(r"[^a-z0-9_-]+", "-", identity.lower()).strip("-_")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    prefix_length = 64 - len(digest) - 1
    prefix = normalized[:prefix_length].rstrip("-_") or "run"
    return f"{prefix}-{digest}"


def training_paths(output: Path) -> TrainingPaths:
    """Resolve one immutable version inside an experiment or a local smoke run."""
    resolved = output.resolve()
    runs_root = RUNS_ROOT.resolve()
    experiment_root = EXPERIMENT_ROOT.resolve()
    if resolved.parent == experiment_root:
        raise ValueError(
            "training output must be a versioned experiment child such as "
            f"{output / 'V1_initial_contract'}"
        )
    if resolved.parent.parent == experiment_root:
        experiment = resolved.parent
        if EXPERIMENT_DIR.fullmatch(experiment.name) is None:
            raise ValueError(f"invalid experiment directory name: {experiment.name}")
        if VERSIONED_ATTEMPT.fullmatch(resolved.name) is None:
            raise ValueError(
                "training attempt must match V<n>_<snake_case_tag>: "
                f"{resolved.name}"
            )
        paths = TrainingPaths(
            run=output,
            checkpoints=CHECKPOINT_ROOT / experiment.name / resolved.name,
            tensorboard=RUNS_ROOT / "tensorboard" / experiment.name / resolved.name,
            config=output / "training_config.json",
            metrics=output / "training_metrics.jsonl",
            summary=output / "training_summary.json",
        )
        occupied = [
            path
            for path in (paths.run, paths.checkpoints, paths.tensorboard)
            if path.is_file() or (path.is_dir() and any(child.is_file() for child in path.rglob("*")))
        ]
        if occupied:
            raise FileExistsError(
                "training attempt paths are already in use; allocate the next version: "
                + ", ".join(str(path) for path in occupied)
            )
        return paths
    return TrainingPaths(
        run=output,
        checkpoints=output / "checkpoints",
        tensorboard=output / "tensorboard",
        config=output / "config.json",
        metrics=output / "metrics.jsonl",
        summary=output / "summary.json",
    )


def numbered_artifact_path(requested: Path, group: str) -> Path:
    """Map legacy ``rl_runs/<label>/evaluation`` into the separated archive tree."""
    if group != "evaluation":
        raise ValueError(f"unknown RL artifact group: {group}")
    resolved = requested.resolve()
    runs_root = RUNS_ROOT.resolve()
    if resolved.name != "evaluation" or resolved.parent.parent != runs_root:
        return requested
    experiment = resolved.parent
    if NUMBERED_ARTIFACT.match(experiment.name):
        return RUNS_ROOT / "evaluation" / experiment.name
    normalized = re.sub(r"[^a-z0-9_-]+", "-", experiment.name.lower()).strip("-")
    if EXPERIMENT_ROOT.is_dir():
        for child in EXPERIMENT_ROOT.iterdir():
            match = EXPERIMENT_DIR.match(child.name)
            if child.is_dir() and match is not None and match.group("label") == normalized:
                return RUNS_ROOT / "evaluation" / child.name
    allocated = next_experiment_path(experiment.name, EXPERIMENT_ROOT)
    return RUNS_ROOT / "evaluation" / allocated.name


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def next_experiment_path(label: str, runs_root: Path = EXPERIMENT_ROOT) -> Path:
    """Return the next globally numbered experiment directory."""
    normalized = re.sub(r"[^a-z0-9_-]+", "-", label.lower()).strip("-")
    if not normalized:
        raise ValueError("experiment label must contain an ASCII letter or number")
    numbers = [
        int(match.group("number"))
        for child in runs_root.iterdir()
        if child.is_dir() and (match := EXPERIMENT_DIR.match(child.name))
    ] if runs_root.is_dir() else []
    return runs_root / f"{max(numbers, default=0) + 1:04d}-{normalized}"


def initialize_experiment(
    label: str,
    *,
    objective: str,
    runs_root: Path = EXPERIMENT_ROOT,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Create a tracked run record without creating a supervised dataset."""
    root = next_experiment_path(label, runs_root)
    root.mkdir(parents=True)
    evaluation = RUNS_ROOT / "evaluation" / root.name
    evaluation.mkdir(parents=True)
    tensorboard = RUNS_ROOT / "tensorboard" / root.name
    tensorboard.mkdir(parents=True)
    checkpoint = CHECKPOINT_ROOT / root.name
    checkpoint.mkdir(parents=True)
    manifest = {
        "schema_version": "ptcg_experiment_v2",
        "experiment_id": root.name,
        "label": root.name.split("-", 1)[1],
        "objective": objective,
        "status": "initialized",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain": _git_value("status", "--short"),
        "paths": {
            "run": str(root.relative_to(RUNS_ROOT.parent)),
            "tensorboard": str(tensorboard.relative_to(RUNS_ROOT.parent)),
            "checkpoint": str(checkpoint.relative_to(RUNS_ROOT.parent)),
            "evaluation": str(evaluation.relative_to(RUNS_ROOT.parent)),
            "dataset": None,
        },
        "tracking": {
            "metric_schema": WANDB_METRIC_SCHEMA,
            "wandb_mode": "disabled",
            "wandb_project": WANDB_PROJECT,
        },
        **(metadata or {}),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _main() -> None:
    parser = argparse.ArgumentParser(description="Create a numbered RL experiment directory")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("label")
    create.add_argument("--objective", required=True)
    args = parser.parse_args()
    if args.command == "create":
        print(initialize_experiment(args.label, objective=args.objective))


if __name__ == "__main__":
    _main()

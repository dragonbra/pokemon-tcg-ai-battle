"""Helpers for keeping RL artifacts grouped by numbered experiment.

New experiments use one directory per experiment so that the data manifest,
source snapshot, training output, candidate package and frozen evaluation stay
together.  ``numbered_artifact_path`` remains for compatibility with the
older evaluation CLI and is intentionally not used by new experiments.
"""

from __future__ import annotations

import re
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
ARTIFACT_GROUPS = ("training", "research_candidates", "evaluation")
NUMBERED_ARTIFACT = re.compile(r"^(?P<number>\d{4})-(?P<label>.+)$")
EXPERIMENT_DIR = re.compile(r"^(?P<number>\d{4})-(?P<label>[a-z0-9][a-z0-9_-]*)$")
EXPERIMENT_GROUPS = ("data", "source", "training", "candidate", "evaluation", "notes")


def numbered_artifact_path(requested: Path, group: str) -> Path:
    """Number an unnumbered direct child of one RL artifact group.

    The next number is selected across training, research candidates, and
    evaluation so one experiment can reuse the same ID in each group.
    Paths outside ``rl/runs/<group>`` and already-numbered paths are unchanged.
    """
    if group not in ARTIFACT_GROUPS:
        raise ValueError(f"unknown RL artifact group: {group}")
    resolved = requested.resolve()
    group_root = (RUNS_ROOT / group).resolve()
    if resolved.parent != group_root or NUMBERED_ARTIFACT.match(resolved.name):
        return requested

    numbers: list[int] = []
    matching_numbers: list[int] = []
    for artifact_group in ARTIFACT_GROUPS:
        root = RUNS_ROOT / artifact_group
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = NUMBERED_ARTIFACT.match(child.name)
            if match is not None:
                number = int(match.group("number"))
                numbers.append(number)
                if match.group("label") == resolved.name:
                    matching_numbers.append(number)

    if matching_numbers:
        return group_root / f"{min(matching_numbers):04d}-{resolved.name}"

    number = max(numbers, default=0) + 1
    while True:
        candidate = group_root / f"{number:04d}-{resolved.name}"
        if not candidate.exists():
            return candidate
        number += 1


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def next_experiment_path(label: str, runs_root: Path = RUNS_ROOT) -> Path:
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
    runs_root: Path = RUNS_ROOT,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Create a numbered experiment directory and its immutable manifest."""
    root = next_experiment_path(label, runs_root)
    root.mkdir(parents=True)
    for group in EXPERIMENT_GROUPS:
        (root / group).mkdir()
    manifest = {
        "schema_version": "ptcg_experiment_v1",
        "experiment_id": root.name,
        "label": root.name.split("-", 1)[1],
        "objective": objective,
        "status": "initialized",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain": _git_value("status", "--short"),
        "artifact_groups": {group: group for group in EXPERIMENT_GROUPS},
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

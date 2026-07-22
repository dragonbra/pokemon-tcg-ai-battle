"""Helpers for keeping RL artifacts grouped by numbered experiment."""

from __future__ import annotations

import re
from pathlib import Path


RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
ARTIFACT_GROUPS = ("training", "research_candidates", "evaluation")
NUMBERED_ARTIFACT = re.compile(r"^(?P<number>\d{4})-(?P<label>.+)$")


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

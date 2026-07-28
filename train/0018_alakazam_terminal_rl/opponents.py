from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Literal

from evaluation.cli import DEFAULT_CATALOG, load_opponent_catalog
from evaluation.packages.loader import SubmissionPackage

from .constants import REPOSITORY_ROOT
from .rollout.protocol import RolloutJob


def load_frozen_pool() -> tuple[list[SubmissionPackage], dict[str, Any]]:
    packages = load_opponent_catalog(DEFAULT_CATALOG, REPOSITORY_ROOT / "evaluation")
    catalog_hash = hashlib.sha256(DEFAULT_CATALOG.read_bytes()).hexdigest()
    train_names = [package.name for package in packages]
    snapshot = {
        "schema_version": "0018_opponent_snapshot_v1",
        "catalog": str(DEFAULT_CATALOG.relative_to(REPOSITORY_ROOT)),
        "catalog_sha256": catalog_hash,
        "enabled_count": len(packages),
        "train_count": len(train_names),
        "holdout_count": 0,
        "train": train_names,
        "holdout": [],
        "packages": [
            {
                "name": package.name,
                "package_hash": package.package_hash,
                "deck_hash": package.deck_hash,
                "cg_manifest": package.cg_manifest,
            }
            for package in packages
        ],
    }
    return packages, snapshot


def select_slice(
    packages: list[SubmissionPackage],
    snapshot: dict[str, Any],
    slice_name: Literal["train", "holdout", "all"],
) -> list[SubmissionPackage]:
    if slice_name == "all":
        return packages
    names = set(snapshot[slice_name])
    return [package for package in packages if package.name in names]


def balanced_jobs(
    packages: list[SubmissionPackage],
    *,
    count: int,
    seed: int,
    prefix: str,
) -> list[RolloutJob]:
    if not packages or count < 1:
        raise ValueError("opponent jobs require a nonempty pool and positive count")
    randomizer = random.Random(seed)
    schedule: list[SubmissionPackage] = []
    while len(schedule) < count:
        block = list(packages)
        randomizer.shuffle(block)
        schedule.extend(block)
    schedule = schedule[:count]
    return [
        RolloutJob(
            episode_id=f"{prefix}-{index:06d}",
            opponent=opponent,
            candidate_first=index % 2 == 0,
            seed=seed + index,
        )
        for index, opponent in enumerate(schedule)
    ]


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["balanced_jobs", "load_frozen_pool", "select_slice", "write_snapshot"]

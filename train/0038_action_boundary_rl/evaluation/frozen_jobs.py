"""Translate the immutable 2,048 manifest into official-engine RolloutJobs."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.runtime.seeded import build_seeded_runtime

from ..league import load_frozen_catalog
from ..rollout.protocol import RolloutJob


def build_frozen_jobs(
    manifest_path: Path, *, focal_deck: tuple[int, ...], runtime_root: Path,
    source_policy_update: int,
) -> list[RolloutJob]:
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != "0038_frozen_panel_manifest_v1" \
            or payload.get("games") != 2048 or payload.get("unique_seeds") != 2048:
        raise ValueError("invalid 0038 Frozen panel manifest")
    decks = {item.deck_id: item.deck for item in load_frozen_catalog()}
    runtime = build_seeded_runtime()
    jobs = []
    for index, row in enumerate(payload["entries"]):
        seed = int(row["seed"])
        jobs.append(RolloutJob(
            game_id=f"frozen-{row['shard_id']}-{index:04d}",
            opponent_id=row["opponent_deck"], focal_first=bool(row["focal_first"]),
            seed=seed, source_policy_update=source_policy_update,
            focal_deck=focal_deck, opponent_deck=decks[row["opponent_deck"]],
            runtime_root=runtime_root, policy_seed=(seed + 1_700_000_009) & 0x7FFFFFFF or 1,
            search_seed=(seed + 900_000_007) & 0x7FFFFFFF or 1,
            engine_library=runtime.library_path, action_boundary_mode="enabled",
            trace_policy="errors_and_sample",
        ))
    if len({item.seed for item in jobs}) != 2048:
        raise RuntimeError("Frozen jobs lost seed uniqueness")
    return jobs


__all__ = ["build_frozen_jobs"]

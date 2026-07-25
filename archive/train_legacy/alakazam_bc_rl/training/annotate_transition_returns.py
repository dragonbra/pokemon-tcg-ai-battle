"""Annotate BC records with discounted, transition-level return targets.

The source BC dataset already stores the visible potential difference for each
decision.  This module turns those local rewards into a trajectory return
without introducing hidden cards or evaluator-only labels:

    r_t = shaping_scale * visible_potential_delta_t + terminal_reward_t
    G_t = r_t + gamma * G_{t+1}

The terminal reward is attached only to the last decision of each source game.
Returns are clipped to the model value contract ``[-1, 1]`` while the raw
transition reward remains available for audit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rl_environment.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .dataset import load_behavior_cloning_dataset


TRANSITION_RETURN_VERSION = "transition_return_v1"


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def annotate_records(
    records: Iterable[dict[str, Any]],
    *,
    gamma: float = 0.99,
    shaping_scale: float = 0.1,
) -> list[dict[str, Any]]:
    """Return records annotated with a backward discounted return per game.

    Records from a source trace are grouped by ``source`` (falling back to
    ``game_id``), sorted by the recorded simulator step, then restored to the
    input order.  This keeps JSONL diffs stable and makes the operation safe
    for datasets assembled from multiple evaluation games.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if shaping_scale < 0.0:
        raise ValueError("shaping_scale must be non-negative")
    annotated = [dict(record) for record in records]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(annotated):
        key = str(record.get("source") or record.get("game_id") or index)
        groups[key].append(index)

    for indices in groups.values():
        indices.sort(key=lambda index: int(annotated[index].get("step", index)))
        future_return = 0.0
        for position in range(len(indices) - 1, -1, -1):
            index = indices[position]
            record = annotated[index]
            shaping = record.get("potential_shaping") or {}
            if not isinstance(shaping, dict):
                shaping = {}
            visible_delta = float(shaping.get("total", 0.0) or 0.0)
            terminal = float(record.get("terminal_outcome", 0.0) or 0.0)
            terminal_reward = terminal if position == len(indices) - 1 else 0.0
            transition_reward = shaping_scale * visible_delta + terminal_reward
            future_return = transition_reward + gamma * future_return
            record["transition_reward"] = transition_reward
            record["visible_potential_delta"] = visible_delta
            record["transition_return"] = _clip(future_return)
            record["transition_return_unclipped"] = future_return
            record["reward_target_version"] = TRANSITION_RETURN_VERSION
            record["reward_gamma"] = gamma
            record["reward_shaping_scale"] = shaping_scale
            record["terminal_reward"] = terminal_reward
    return annotated


def annotate_dataset(
    input_path: Path,
    output_path: Path,
    *,
    gamma: float = 0.99,
    shaping_scale: float = 0.1,
    storage_path: Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, int | float | str]:
    storage = assert_storage_safe(storage_path, min_free_gib)
    records = load_behavior_cloning_dataset(input_path)
    annotated = annotate_records(records, gamma=gamma, shaping_scale=shaping_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in annotated:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return {
        "records": len(annotated),
        "gamma": gamma,
        "shaping_scale": shaping_scale,
        "reward_target_version": TRANSITION_RETURN_VERSION,
        "input": str(input_path),
        "output": str(output_path),
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--shaping-scale", type=float, default=0.1)
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    print(
        json.dumps(
            annotate_dataset(
                args.input,
                args.output,
                gamma=args.gamma,
                shaping_scale=args.shaping_scale,
                storage_path=args.storage_path,
                min_free_gib=args.min_free_gib,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

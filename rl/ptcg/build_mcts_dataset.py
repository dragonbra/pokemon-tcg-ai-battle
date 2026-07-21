"""Build small policy targets from official counterfactual search expansions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .dataset import DATASET_VERSION
from .features import encode_observation
from .inference import PTCGCandidatePolicy


def _load_deck(path: Path) -> list[int]:
    values = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != 60:
        raise ValueError(f"deck must contain 60 card IDs: {path}")
    return values


def _load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("trace"), list):
        raise ValueError(f"not an evaluation trace: {path}")
    return payload


def _load_teacher(root: Path) -> ModuleType:
    path = root.resolve() / "main.py"
    spec = importlib.util.spec_from_file_location("rl_mcts_rollout_teacher", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load rollout teacher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cg(cg_root: Path):
    sys.path.insert(0, str(cg_root.resolve()))
    from cg.api import search_begin, search_end, search_step, to_observation_class

    return search_begin, search_end, search_step, to_observation_class


def _perspective_observation(observation: object, player_index: int) -> dict[str, Any]:
    result = asdict(observation)
    current = result.get("current") or {}
    if int(current.get("yourIndex", player_index)) == player_index:
        return result
    players = current.get("players") or []
    if len(players) == 2:
        current["players"] = [players[1], players[0]]
    current["yourIndex"] = player_index
    first_player = current.get("firstPlayer")
    if first_player in (0, 1):
        current["firstPlayer"] = 1 - int(first_player)
    return result


def _leaf_value(
    observation: object,
    player_index: int,
    policy: PTCGCandidatePolicy,
) -> float:
    state = getattr(observation, "current", None)
    if state is None:
        return 0.0
    result = int(getattr(state, "result", -1))
    if result >= 0:
        return 1.0 if result == player_index else -1.0
    observation_dict = _perspective_observation(observation, player_index)
    try:
        _, value = policy.select(observation_dict)
        return float(value)
    except Exception:
        return 0.0


def _search_action_value(
    observation: dict[str, Any],
    action_index: int,
    player_index: int,
    deck: list[int],
    policy: PTCGCandidatePolicy,
    search_begin: Any,
    search_end: Any,
    search_step: Any,
    to_observation_class: Any,
    rng: random.Random,
    rollout_steps: int,
    rollout_teacher: ModuleType | None,
) -> float | None:
    current = observation.get("current") or {}
    players = current.get("players") or []
    if len(players) != 2:
        return None
    own = players[player_index]
    opponent = players[1 - player_index]
    active = opponent.get("active") or []
    try:
        root = search_begin(
            to_observation_class(observation),
            your_deck=rng.sample(deck, min(int(own.get("deckCount", 0) or 0), len(deck))),
            your_prize=rng.sample(deck, min(len(own.get("prize") or []), len(deck))),
            opponent_deck=[1072] * int(opponent.get("deckCount", 0) or 0),
            opponent_prize=[1] * len(opponent.get("prize") or []),
            opponent_hand=[1] * int(opponent.get("handCount", 0) or 0),
            opponent_active=[1072] if active and active[0] is None else [],
        )
        search_id = root.searchId
        leaf = search_step(search_id, [action_index]).observation
        if rollout_teacher is not None:
            rollout_teacher.agent({"select": None})
        for _ in range(rollout_steps):
            state = getattr(leaf, "current", None)
            if state is None or int(getattr(state, "result", -1)) >= 0:
                break
            leaf_dict = asdict(leaf)
            next_select = leaf_dict.get("select") or {}
            options = next_select.get("option") or []
            if not options:
                break
            minimum = int(next_select.get("minCount", 1) or 0)
            maximum = int(next_select.get("maxCount", 1) or 1)
            if rollout_teacher is not None:
                try:
                    proposed = rollout_teacher.agent(leaf_dict)
                    if (
                        isinstance(proposed, list)
                        and minimum <= len(proposed) <= maximum
                        and len(set(int(index) for index in proposed)) == len(proposed)
                        and all(0 <= int(index) < len(options) for index in proposed)
                    ):
                        selection = [int(index) for index in proposed]
                    else:
                        raise ValueError("teacher returned an invalid search selection")
                except Exception:
                    count = min(max(1, minimum), maximum, len(options))
                    selection = list(range(count))
            elif minimum == maximum == 1 and int(getattr(state, "yourIndex", -1)) == player_index:
                try:
                    next_index, _value = policy.select(leaf_dict)
                    selection = [next_index]
                except Exception:
                    selection = [0]
            else:
                count = min(max(1, minimum), maximum, len(options))
                selection = list(range(count))
            leaf = search_step(search_id, selection).observation
        return _leaf_value(leaf, player_index, policy)
    except Exception:
        return None
    finally:
        try:
            search_end()
        except Exception:
            pass


def build_records(
    traces: Iterable[Path],
    *,
    checkpoint: Path,
    deck: list[int],
    cg_root: Path,
    max_records: int,
    seed: int,
    rollout_steps: int,
    rollout_teacher: ModuleType | None,
) -> list[dict[str, Any]]:
    policy = PTCGCandidatePolicy.from_checkpoint(str(checkpoint), map_location="cpu")
    schema_by_width = {24: "ptcg_features_v1", 32: "ptcg_features_v2", 36: "ptcg_features_v3"}
    feature_schema_version = schema_by_width[policy.feature_config.state_numeric_dim]
    search_begin, search_end, search_step, to_observation_class = _load_cg(cg_root)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for trace_path in traces:
        payload = _load_trace(trace_path)
        result = payload.get("result") or {}
        player_index = int(result.get("candidate_physical_index", 0))
        winner = result.get("winner")
        outcome = 1.0 if winner == player_index else -1.0 if winner in (0, 1) else 0.0
        for entry in payload["trace"]:
            if len(records) >= max_records:
                return records
            if not isinstance(entry, dict):
                continue
            observation = entry.get("observation")
            if not isinstance(observation, dict):
                continue
            current = observation.get("current") or {}
            select = observation.get("select") or {}
            if int(current.get("yourIndex", -1)) != player_index:
                continue
            if int(select.get("type", -1)) != 0 or int(select.get("context", -1)) != 0:
                continue
            if int(select.get("minCount", 0) or 0) != 1 or int(select.get("maxCount", 0) or 0) != 1:
                continue
            if not select.get("option"):
                continue
            encoded = encode_observation(observation, policy.feature_config)
            values: list[float | None] = []
            for index, _option in enumerate(select.get("option") or []):
                values.append(
                    _search_action_value(
                        observation,
                        index,
                        player_index,
                        deck,
                        policy,
                        search_begin,
                        search_end,
                        search_step,
                        to_observation_class,
                        rng,
                        rollout_steps,
                        rollout_teacher,
                    )
                )
            valid = [index for index, value in enumerate(values) if value is not None]
            if not valid:
                continue
            target = max(valid, key=lambda index: float(values[index]))
            if not encoded["action_mask"][target]:
                continue
            records.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "feature_schema_version": feature_schema_version,
                    "source": str(trace_path.resolve()),
                    "game_id": str(result.get("game_id") or trace_path.stem),
                    "step": int(entry.get("step", -1)),
                    "player_index": player_index,
                    "selection_type": 0,
                    "selection_context": 0,
                    "target": target,
                    "terminal_outcome": outcome,
                    "mcts_action_values": values,
                    "teacher_or_rollout_action": entry.get("action"),
                    "encoded": encoded,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--cg-root", type=Path, required=True)
    parser.add_argument("--rollout-teacher", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    if args.max_records < 1 or args.rollout_steps < 0:
        raise ValueError("max-records must be positive and rollout-steps cannot be negative")
    storage = assert_storage_safe(args.storage_path, args.min_free_gib)
    rollout_teacher = _load_teacher(args.rollout_teacher) if args.rollout_teacher else None
    records = build_records(
        args.traces,
        checkpoint=args.checkpoint,
        deck=_load_deck(args.deck),
        cg_root=args.cg_root,
        max_records=args.max_records,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        rollout_teacher=rollout_teacher,
    )
    if not records:
        raise ValueError("MCTS target collection produced no records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "records": len(records),
                "output": str(args.output),
                "storage_path": storage.path,
                "storage_free_gib": round(storage.free_gib, 2),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

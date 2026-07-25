from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any

from rl_environment.streaming import iter_jsonl
from train.alakazam_sota_model.dataset import (
    ReplayArchiveResolver,
    _frame_observation,
    _record_identity,
    _visual_frames,
)


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _player(current: dict[str, Any], index: int) -> dict[str, Any]:
    players = _list(current.get("players"))
    return players[index] if 0 <= index < len(players) and isinstance(players[index], dict) else {}


def _pokemon(player: dict[str, Any]) -> list[dict[str, Any]]:
    values = [*_list(player.get("active")), *_list(player.get("bench"))]
    return [value for value in values if isinstance(value, dict)]


def _state_vector(observation: dict[str, Any], actor: int) -> tuple[int, ...]:
    current = observation.get("current") or {}
    own = _player(current, actor)
    opp = _player(current, 1 - actor)
    own_pokemon = _pokemon(own)
    opp_pokemon = _pokemon(opp)
    return (
        _int(own.get("handCount"), len(_list(own.get("hand")))),
        _int(own.get("deckCount")),
        _int(own.get("prizeCount"), len(_list(own.get("prize")))),
        len(_list(own.get("discard"))),
        len(_list(own.get("active"))),
        len(_list(own.get("bench"))),
        sum(max(0, _int(card.get("maxHp"), _int(card.get("hp"))) - _int(card.get("hp"))) for card in own_pokemon),
        sum(len(_list(card.get("energyCards", card.get("energies")))) for card in own_pokemon),
        len(_list(opp.get("active"))),
        len(_list(opp.get("bench"))),
        sum(max(0, _int(card.get("maxHp"), _int(card.get("hp"))) - _int(card.get("hp"))) for card in opp_pokemon),
    )


def audit_transitions(
    source_dataset: Path,
    archive_root: Path,
    *,
    max_groups: int | None = None,
) -> dict[str, Any]:
    names = (
        "own_hand",
        "own_deck",
        "own_prize",
        "own_discard",
        "own_active",
        "own_bench",
        "own_damage",
        "own_energy",
        "opp_active",
        "opp_bench",
        "opp_damage",
    )
    coverage = Counter()
    changed = Counter()
    selected_types = Counter()
    groups = 0
    decisions = 0
    next_available = 0
    with ReplayArchiveResolver(archive_root) as resolver:
        for _, grouped in groupby(iter_jsonl(source_dataset), key=_record_identity):
            rows = list(grouped)
            payload, _ = resolver.read(str(rows[0].get("source", "")))
            frames = _visual_frames(payload)
            for row in rows:
                decisions += 1
                step = int(row["episode_step"])
                if step + 1 >= len(frames):
                    coverage["missing_next_frame"] += 1
                    continue
                before = _frame_observation(frames[step])
                after = _frame_observation(frames[step + 1])
                current = before.get("current") or {}
                next_current = after.get("current") or {}
                actor = int(row["player_index"])
                before_vector = _state_vector(before, actor)
                after_vector = _state_vector(after, actor)
                delta = tuple(right - left for left, right in zip(before_vector, after_vector))
                next_available += 1
                same_turn = _int(current.get("turn")) == _int(next_current.get("turn"))
                same_actor = _int(next_current.get("yourIndex"), -1) == actor
                coverage["same_turn"] += int(same_turn)
                coverage["turn_advanced"] += int(not same_turn)
                coverage["same_actor"] += int(same_actor)
                coverage["actor_changed"] += int(not same_actor)
                coverage["any_observable_delta"] += int(any(delta))
                coverage["zero_observable_delta"] += int(not any(delta))
                coverage["same_turn_any_delta"] += int(same_turn and any(delta))
                for name, value in zip(names, delta):
                    changed[name] += int(value != 0)
                    changed[f"{name}_same_turn"] += int(same_turn and value != 0)
                select = before.get("select") or {}
                options = _list(select.get("option"))
                for index in row.get("targets") or []:
                    if isinstance(index, int) and 0 <= index < len(options):
                        option = options[index]
                        if isinstance(option, dict):
                            selected_types[str(option.get("type"))] += 1
            groups += 1
            if max_groups is not None and groups >= max_groups:
                break
    return {
        "schema_version": "alakazam_sota_transition_audit_v1",
        "groups": groups,
        "decisions": decisions,
        "next_frame_available": next_available,
        "coverage": dict(sorted(coverage.items())),
        "delta_changed": dict(sorted(changed.items())),
        "selected_option_types": dict(sorted(selected_types.items())),
        "recommended_contract": {
            "same_turn_resource_delta": "supervise only when current.turn is unchanged",
            "turn_transition": "separate classification target",
            "counterfactual": False,
            "label_scope": "chosen action and automatic engine consequences until next decision frame",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit E4 next-decision transition coverage")
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_transitions(
        args.source_dataset,
        args.archive_root,
        max_groups=args.max_groups,
    )
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

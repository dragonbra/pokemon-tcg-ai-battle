#!/usr/bin/env python3
"""分析 ``eval/alakazam_replay.py --save-traces`` 生成的 Alakazam replay。

这个脚本只读取 JSON，不运行引擎，也不修改评测结果。它统一按照
``alakazamPhysicalIndex`` 识别我方玩家，避免手工复盘时把交换先后手的对手状态
误当成我方状态。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ALAKAZAM = 743
ABRA = 741
KADABRA = 742
ATTACK_LINE = {ABRA, KADABRA, ALAKAZAM}
PSYCHIC_ENERGY = 5


def _field(player: dict[str, Any]) -> list[dict[str, Any]]:
    active = player.get("active")
    return [card for card in [active, *(player.get("bench") or [])] if card]


def _ready_count(player: dict[str, Any]) -> int:
    return sum(
        card.get("id") in ATTACK_LINE and PSYCHIC_ENERGY in (card.get("energies") or [])
        for card in _field(player)
    )


def _first_turn(
    states: Iterable[dict[str, Any]], predicate: Any, *, minimum_turn: int = 0
) -> int | None:
    for state in states:
        if state["turn"] >= minimum_turn and predicate(state):
            return int(state["turn"])
    return None


def _selected_options(step: dict[str, Any], player_index: int) -> list[dict[str, Any]]:
    observation = step.get("observation") or {}
    select = observation.get("select") or {}
    options = select.get("option") or []
    action = step.get("action") or []
    selected: list[dict[str, Any]] = []
    for index in action:
        if isinstance(index, int) and 0 <= index < len(options):
            option = options[index]
            if isinstance(option, dict):
                selected.append(option)
    return selected


def _hand_card_id(step: dict[str, Any], option: dict[str, Any], player_index: int) -> int | None:
    if option.get("type") not in {7, 8}:
        return None
    index = option.get("index")
    if not isinstance(index, int):
        return None
    players = ((step.get("observation") or {}).get("current") or {}).get("players") or []
    if not 0 <= player_index < len(players):
        return None
    hand = players[player_index].get("hand") or []
    if not 0 <= index < len(hand):
        return None
    card_id = hand[index].get("id")
    return int(card_id) if card_id is not None else None


def _analyze_game(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    player_index = int(record["alakazamPhysicalIndex"])
    trace = record.get("trace") or []
    states: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    attack_ids: dict[str, int] = {}

    for step in trace:
        players = step.get("players") or []
        player = players[player_index] if player_index < len(players) else {}
        states.append(
            {
                "turn": int(step.get("turn", 0)),
                "active": (player.get("active") or {}).get("id"),
                "field": _field(player),
                "ready": _ready_count(player),
                "deck": player.get("deckCount"),
                "hand": player.get("handCount"),
            }
        )
        if step.get("yourIndex") != player_index:
            continue
        for option in _selected_options(step, player_index):
            option_type = option.get("type")
            if option_type == 13:
                key = f"attack:{option.get('attackId', 'unknown')}"
                attack_ids[key] = attack_ids.get(key, 0) + 1
                continue
            card_id = _hand_card_id(step, option, player_index)
            if card_id is not None:
                key = f"card:{card_id}"
                action_counts[key] = action_counts.get(key, 0) + 1

    first_alakazam = _first_turn(
        states, lambda state: any(card.get("id") == ALAKAZAM for card in state["field"])
    )
    first_active_alakazam = _first_turn(states, lambda state: state["active"] == ALAKAZAM)
    first_attack = _first_turn(
        [
            {
                "turn": int(step.get("turn", 0)),
                "selected": _selected_options(step, player_index),
            }
            for step in trace
            if step.get("yourIndex") == player_index
        ],
        lambda state: any(option.get("type") == 13 for option in state["selected"]),
    )
    empty_state = next(
        (state for state in states if state["turn"] > 0 and state["active"] is None), None
    )

    return {
        "opponent": record.get("opponent", path.parent.name),
        "game": record.get("game", path.stem),
        "winner": record.get("winner"),
        "steps": record.get("steps"),
        "first_alakazam_turn": first_alakazam,
        "first_active_alakazam_turn": first_active_alakazam,
        "first_attack_turn": first_attack,
        "first_empty_active_turn": empty_state["turn"] if empty_state else None,
        "ready_at_first_empty": empty_state["ready"] if empty_state else None,
        "max_ready_attackers": max((state["ready"] for state in states), default=0),
        "action_counts": action_counts,
        "attack_ids": attack_ids,
    }


def _iter_games(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/game_*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path, help="包含各 opponent/game_*.json 的目录")
    args = parser.parse_args()
    paths = list(_iter_games(args.trace_dir))
    if not paths:
        raise SystemExit(f"没有找到 replay: {args.trace_dir}")
    for path in paths:
        print(json.dumps(_analyze_game(path), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

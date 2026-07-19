#!/usr/bin/env python3
"""从 Kaggle 官方 replay 提取 Alakazam 策略指标。

输入目录应直接包含 ``episode-*-replay.json``。脚本不运行引擎，也不修改 replay，
只生成可复核的 ``games.json`` 和 ``summary.json``。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ALAKAZAM = 743
ATTACK_LINE = {741, 742, 743}
PSYCHIC_ENERGY = {5, 19}
OUR_NAME_HINTS = ("糕手",)
TRACKED_CARDS = {
    13: "Enriching Energy",
    66: "Dudunsparce",
    1079: "Rare Candy",
    1081: "Enhanced Hammer",
    1086: "Buddy-Buddy Poffin",
    1097: "Night Stretcher",
    1129: "Sacred Ash",
    1146: "Wondrous Patch",
    1152: "Poké Pad",
    1182: "Boss’s Orders",
    1197: "Xerosic’s Machinations",
    1225: "Hilda",
    1231: "Dawn",
    1264: "Battle Cage",
}


def cards(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [card for card in value if isinstance(card, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def card_id(card: Any) -> int | None:
    if isinstance(card, dict) and isinstance(card.get("id"), int):
        return int(card["id"])
    return None


def field_cards(player: dict[str, Any]) -> list[dict[str, Any]]:
    return cards(player.get("active")) + cards(player.get("bench"))


def card_names(record: dict[str, Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    official = Path(__file__).resolve().parents[1] / "data/official/EN_Card_Data.csv"
    if official.exists():
        with official.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    names[int(row["Card ID"])] = row["Card Name"]
                except (KeyError, TypeError, ValueError):
                    continue
    for step in record.get("steps", []):
        for item in step:
            current = (item.get("observation") or {}).get("current") or {}
            for player in current.get("players") or []:
                for card in cards(player.get("deck")):
                    if isinstance(card.get("id"), int) and card.get("name"):
                        names.setdefault(int(card["id"]), str(card["name"]))
            if names:
                return names
    return names


def initial_visualized_decks(record: dict[str, Any]) -> list[Counter[int]] | None:
    """从 replay 的初始 visualize 快照恢复双方 60 张牌的 ID 多重集。"""
    for step in record.get("steps", []):
        for item in step:
            for visual in item.get("visualize") or []:
                action = visual.get("action") if isinstance(visual, dict) else None
                if not isinstance(action, list) or len(action) != 2:
                    continue
                if not all(
                    isinstance(deck, list)
                    and len(deck) == 60
                    and all(isinstance(card, int) for card in deck)
                    for deck in action
                ):
                    continue
                return [Counter(deck) for deck in action]
    return None


def find_self_index(record: dict[str, Any]) -> int:
    names = (record.get("info") or {}).get("TeamNames") or []
    candidates = [
        index
        for index, name in enumerate(names)
        if isinstance(name, str) and any(hint in name for hint in OUR_NAME_HINTS)
    ]
    if len(candidates) == 1:
        return candidates[0]

    required = Counter({743: 4, 742: 4, 741: 4, 305: 3, 66: 3})
    for step in record.get("steps", []):
        for item in step:
            current = (item.get("observation") or {}).get("current") or {}
            for index, player in enumerate(current.get("players") or []):
                counts = Counter(card_id(card) for card in cards(player.get("deck")))
                if all(counts[card] >= amount for card, amount in required.items()):
                    return index
    raise ValueError(f"无法识别我方玩家: {record.get('id')}")


def player_state(current: dict[str, Any], index: int) -> dict[str, Any]:
    players = current.get("players") or []
    if not 0 <= index < len(players):
        return {}
    player = players[index]
    if not isinstance(player, dict):
        return {}
    return player


def active_card(player: dict[str, Any]) -> dict[str, Any] | None:
    active = cards(player.get("active"))
    return active[0] if active else None


def ready_attackers(player: dict[str, Any]) -> int:
    count = 0
    for pokemon in field_cards(player):
        if card_id(pokemon) in ATTACK_LINE and cards(pokemon.get("energyCards")):
            count += 1
    return count


def snapshot(current: dict[str, Any], self_index: int) -> dict[str, Any]:
    opponent_index = 1 - self_index
    me = player_state(current, self_index)
    opponent = player_state(current, opponent_index)
    active = active_card(me)
    opponent_active = active_card(opponent)
    bench = [card_id(card) for card in cards(me.get("bench"))]
    opponent_bench = [card_id(card) for card in cards(opponent.get("bench"))]
    return {
        "turn": int(current.get("turn", 0) or 0),
        "first_player": current.get("firstPlayer"),
        "deck": int(me.get("deckCount", len(cards(me.get("deck")))) or 0),
        "hand": int(me.get("handCount", len(cards(me.get("hand")))) or 0),
        "opponent_deck": int(
            opponent.get("deckCount", len(cards(opponent.get("deck")))) or 0
        ),
        "opponent_hand": int(
            opponent.get("handCount", len(cards(opponent.get("hand")))) or 0
        ),
        # Kaggle replay 会把 Prize 卡牌内容脱敏为 null，但保留槽位数量。
        "prizes": len(me.get("prize") or []),
        "opponent_prizes": len(opponent.get("prize") or []),
        "active": card_id(active),
        "active_hp": active.get("hp") if active else None,
        "bench": bench,
        "opponent_active": card_id(opponent_active),
        "opponent_active_hp": opponent_active.get("hp") if opponent_active else None,
        "opponent_bench": opponent_bench,
        "ready_attackers": ready_attackers(me),
    }


def option_cards(
    current: dict[str, Any], option: dict[str, Any], self_index: int
) -> list[int]:
    """尽量解析 selection option 对应的卡；无法解析时返回空列表。"""
    area = option.get("area")
    index = option.get("index")
    if not isinstance(index, int):
        return []
    target_index = option.get("playerIndex", self_index)
    if not isinstance(target_index, int):
        target_index = self_index
    player = player_state(current, target_index)
    area_cards = {
        1: player.get("hand"),
        2: player.get("deck"),
        3: player.get("active"),
        4: player.get("bench"),
        6: player.get("discard"),
        12: player.get("prize"),
    }
    if area not in area_cards:
        return []
    selected = cards(area_cards[area])
    if 0 <= index < len(selected):
        selected_id = card_id(selected[index])
        return [selected_id] if selected_id is not None else []
    return []


def selected_options(item: dict[str, Any]) -> list[dict[str, Any]]:
    observation = item.get("observation") or {}
    select = observation.get("select") or {}
    options = select.get("option") or []
    action = item.get("action") or []
    selected: list[dict[str, Any]] = []
    for index in action:
        if isinstance(index, int) and 0 <= index < len(options):
            option = options[index]
            if isinstance(option, dict):
                selected.append(option)
    return selected


def canonical_event(event: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(event.items()))


def unique_logs(record: dict[str, Any], self_index: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for step in record.get("steps", []):
        for item in step:
            logs = ((item.get("observation") or {}).get("logs") or [])
            for event in logs:
                if not isinstance(event, dict) or event.get("playerIndex") != self_index:
                    continue
                key = canonical_event(event)
                if key not in seen:
                    seen.add(key)
                    events.append(event)
    return events


def analyze_game(path: Path, variant: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    self_index = find_self_index(record)
    opponent_index = 1 - self_index
    names = (record.get("info") or {}).get("TeamNames") or []
    team_names = [str(name) for name in names]
    own_name = team_names[self_index] if self_index < len(team_names) else "unknown"
    opponent_name = team_names[opponent_index] if opponent_index < len(team_names) else "unknown"
    name_map = card_names(record)
    rows: list[dict[str, Any]] = []
    attack_events: list[dict[str, Any]] = []
    seen_attacks: set[tuple[Any, ...]] = set()

    for outer_index, step in enumerate(record.get("steps", [])):
        for inner_index, item in enumerate(step):
            observation = item.get("observation") or {}
            current = observation.get("current")
            if not isinstance(current, dict) or current.get("yourIndex") != self_index:
                continue
            state = snapshot(current, self_index)
            state["outer_step"] = outer_index
            state["inner_step"] = inner_index
            rows.append(state)
            for option in selected_options(item):
                if option.get("type") != 13:
                    continue
                active = active_card(player_state(current, self_index)) or {}
                key = (
                    state["turn"],
                    option.get("attackId"),
                    active.get("serial"),
                )
                if key in seen_attacks:
                    continue
                seen_attacks.add(key)
                attack_events.append(
                    {
                        "turn": state["turn"],
                        "attack_id": option.get("attackId"),
                        "active_id": active.get("id"),
                        "active_serial": active.get("serial"),
                    }
                )

    if not rows:
        raise ValueError(f"没有找到我方状态: {path}")
    logs = unique_logs(record, self_index)
    played = Counter(
        int(event["cardId"])
        for event in logs
        if event.get("type") == 10 and isinstance(event.get("cardId"), int)
    )
    energy_attachments = [event for event in logs if event.get("type") == 11]
    switches = [event for event in logs if event.get("type") == 8]
    first_alakazam = next(
        (state["turn"] for state in rows if ALAKAZAM in state["bench"] or state["active"] == ALAKAZAM),
        None,
    )
    first_active_alakazam = next(
        (state["turn"] for state in rows if state["active"] == ALAKAZAM), None
    )
    first_ready = next((state["turn"] for state in rows if state["ready_attackers"] > 0), None)
    first_alakazam_attack = next(
        (event["turn"] for event in attack_events if event["active_id"] == ALAKAZAM),
        None,
    )
    attack_line_events = [
        event for event in attack_events if event["active_id"] in ATTACK_LINE
    ]
    first_empty_active = next(
        (
            state["turn"]
            for state in rows
            if state["turn"] > 0 and state["active"] is None
        ),
        None,
    )
    no_active = any(
        state["turn"] > 0 and state["active"] is None and not state["bench"] for state in rows
    )
    final = rows[-1]
    rewards = record.get("rewards") or []
    reward = rewards[self_index] if self_index < len(rewards) else None
    if reward == 1:
        result = "win"
    elif reward == -1:
        result = "loss"
    else:
        result = "draw_or_unknown"
    reasons: list[str] = []
    if result == "loss":
        if final["deck"] == 0:
            reasons.append("deck_exhaustion")
        if no_active:
            reasons.append("no_available_active")
        if final["prizes"] > final["opponent_prizes"]:
            reasons.append("prize_race_loss")
        if first_alakazam is None:
            reasons.append("no_alakazam_setup")
        if not reasons:
            reasons.append("combat_or_resource_loss")

    played_named = {
        name_map.get(card, TRACKED_CARDS.get(card, str(card))): count
        for card, count in sorted(played.items())
    }
    opponent_initial = Counter()
    visualized_decks = initial_visualized_decks(record)
    if visualized_decks is not None:
        opponent_initial.update(visualized_decks[opponent_index])
    else:
        first_current = None
        for step in record.get("steps", []):
            for item in step:
                current = (item.get("observation") or {}).get("current")
                if isinstance(current, dict):
                    first_current = current
                    break
            if first_current is not None:
                break
        if first_current is not None:
            opponent_initial.update(
                card_id(card)
                for card in cards(player_state(first_current, opponent_index).get("deck"))
            )
    opponent_initial_named = {
        name_map.get(card, str(card)): count
        for card, count in opponent_initial.most_common(12)
        if card is not None
    }
    return {
        "variant": variant,
        "episode_id": record.get("info", {}).get("EpisodeId"),
        "replay": str(path),
        "team": own_name,
        "opponent": opponent_name,
        "self_index": self_index,
        "first_player": rows[0].get("first_player"),
        "result": result,
        "reward": reward,
        "steps": len(record.get("steps", [])),
        "turns": final["turn"],
        "first_alakazam_turn": first_alakazam,
        "first_active_alakazam_turn": first_active_alakazam,
        "first_ready_attacker_turn": first_ready,
        "first_attack_turn": attack_events[0]["turn"] if attack_events else None,
        "first_alakazam_attack_turn": first_alakazam_attack,
        "attack_turns": [event["turn"] for event in attack_events],
        "attack_count": len(attack_events),
        "attack_line_attack_count": len(attack_line_events),
        "attackers": Counter(event["active_id"] for event in attack_events),
        "max_ready_attackers": max(state["ready_attackers"] for state in rows),
        "first_empty_active_turn": first_empty_active,
        "no_available_active": no_active,
        "final": final,
        "failure_reasons": reasons,
        "played_cards": played_named,
        "energy_attachments": len(energy_attachments),
        "switches": len(switches),
        "opponent_initial_cards": opponent_initial_named,
        "log_event_count": len(logs),
        "name_map": {str(card): name for card, name in name_map.items()},
    }


def numeric_stats(games: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [game[key] for game in games if isinstance(game.get(key), (int, float))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(games: list[dict[str, Any]]) -> dict[str, Any]:
    result_counts = Counter(game["result"] for game in games)
    failures = Counter(
        reason for game in games for reason in game.get("failure_reasons", [])
    )
    card_counts: Counter[str] = Counter()
    opponents = Counter(game["opponent"] for game in games)
    for game in games:
        card_counts.update(game.get("played_cards", {}))
    return {
        "games": len(games),
        "results": dict(result_counts),
        "wins": result_counts.get("win", 0),
        "losses": result_counts.get("loss", 0),
        "win_rate": round(result_counts.get("win", 0) / len(games), 3) if games else None,
        "first_alakazam": numeric_stats(games, "first_alakazam_turn"),
        "first_ready_attacker": numeric_stats(games, "first_ready_attacker_turn"),
        "first_attack": numeric_stats(games, "first_attack_turn"),
        "first_alakazam_attack": numeric_stats(games, "first_alakazam_attack_turn"),
        "attack_count": numeric_stats(games, "attack_count"),
        "attack_line_attack_count": numeric_stats(games, "attack_line_attack_count"),
        "max_ready_attackers": numeric_stats(games, "max_ready_attackers"),
        "final_deck": numeric_stats([game["final"] for game in games], "deck"),
        "final_prizes": numeric_stats([game["final"] for game in games], "prizes"),
        "empty_active_games": sum(game["first_empty_active_turn"] is not None for game in games),
        "no_available_active_games": sum(game["no_available_active"] for game in games),
        "failure_reasons": dict(failures),
        "played_cards": dict(card_counts),
        "opponents": dict(opponents),
    }


def analyze_directory(root: Path, variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(root.glob("episode-*-replay.json"))
    if not paths:
        raise SystemExit(f"没有找到 replay: {root}")
    games = [analyze_game(path, variant) for path in paths]
    for game in games:
        game.pop("name_map", None)
    return games, aggregate(games)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-root", type=Path, required=True)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    auto_games, auto_summary = analyze_directory(args.auto_root, "alakazam_v5_auto_iter")
    v5_games, v5_summary = analyze_directory(args.v5_root, "alakazam_v5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "games.json").write_text(
        json.dumps(auto_games + v5_games, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "scope": {
            "selection": "每个 submission 按 Kaggle API 返回的最新优先顺序取 10 场 public replay",
            "validation_excluded": True,
            "auto_root": str(args.auto_root),
            "v5_root": str(args.v5_root),
        },
        "alakazam_v5_auto_iter": auto_summary,
        "alakazam_v5": v5_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

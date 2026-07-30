"""Compile actor-relative, option-order-free tensors and causal features."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..knowledge.state import CausalSnapshot
from .. import base_model as _m0010
from .. import card_semantics as _card_semantics


COMPILER_VERSION = "0021_actor_relative_option_set_compiler_v1"
A0_MAX_ACTION_STEPS = 64
FULL_CACHE_MAX_ACTION_STEPS = 64
SEMANTIC_WIDTH = 64
ZONE_NUM_WIDTH = 16
LEDGER_NUM_WIDTH = 15
LEDGER_CAT_WIDTH = 4
EVENT_CAT_WIDTH = 8
EVENT_NUM_WIDTH = 4
_SOURCES = (
    Path(__file__),
    Path(__file__).parents[1] / "knowledge/state.py",
    Path(__file__).parents[1] / "config.py",
    Path(_m0010.__file__),
    Path(_card_semantics.__file__),
)


def compiler_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(_SOURCES, key=str):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _deck_cards(row: Mapping[str, Any]) -> list[int]:
    cards: list[int] = []
    for card_id, count in row["deck_manifest"]["counts"]:
        cards.extend([int(card_id)] * int(count))
    if len(cards) != 60:
        raise ValueError("registered deck must contain exactly 60 cards")
    return cards


def _children(card: Mapping[str, Any], field: str) -> Sequence[Any]:
    value = card.get(field, ())
    return value if isinstance(value, Sequence) else ()


def _zone_row(
    player: Mapping[str, Any],
    *,
    known_hand: int,
    unknown_hand: int,
    stadium_count: int,
) -> list[float]:
    active = player.get("active") or []
    bench = player.get("bench") or []
    discard = player.get("discard") or []
    board = [item for item in (*active, *bench) if isinstance(item, Mapping)]
    energy = sum(len(_children(item, "energyCards")) for item in board)
    tools = sum(len(_children(item, "tools")) for item in board)
    evolutions = sum(len(_children(item, "preEvolution")) for item in board)
    damage = sum(
        max(0.0, float(item.get("maxHp", 0) or 0) - float(item.get("hp", 0) or 0))
        for item in board
    )
    prize = player.get("prize") or []
    return [
        float(player.get("deckCount", 0)), float(player.get("handCount", 0)),
        float(len(prize)), float(len(active)), float(len(bench)), float(len(discard)),
        float(energy), float(tools), float(evolutions), damage,
        float(max(0, int(player.get("benchMax", 0)) - len(bench))),
        float(len(board)), float(known_hand), float(unknown_hand),
        float(stadium_count), float(len(board) + energy + tools + evolutions),
    ]


def _actor_relative_players(
    players: Sequence[Mapping[str, Any]], actor: int
) -> tuple[tuple[int, Mapping[str, Any]], tuple[int, Mapping[str, Any]]]:
    if actor not in (0, 1) or len(players) != 2:
        raise ValueError("0021 requires exactly two players and a valid Actor index")
    return ((actor, players[actor]), (1 - actor, players[1 - actor]))


def _ledger_features(
    deck_ids: Sequence[int], snapshot: CausalSnapshot
) -> tuple[list[list[int]], list[list[float]]]:
    unique = sorted(set(deck_ids))
    cats: list[list[int]] = []
    nums: list[list[float]] = []
    for card_id in unique:
        item = snapshot.self_ledger[card_id]
        visible = item.visible
        deck, prize = item.deck, item.prize
        cats.append([
            card_id,
            list(type(deck.state)).index(deck.state) + 1,
            list(type(prize.state)).index(prize.state) + 1,
            int(snapshot.deck_order_known),
        ])
        nums.append([
            float(item.initial), float(visible.get("active", 0)),
            float(visible.get("bench", 0)), float(visible.get("hand", 0)),
            float(visible.get("discard", 0)), float(visible.get("stadium", 0)),
            float(visible.get("playing", 0)),
            float(deck.value if deck.value is not None else 0),
            float(deck.lower), float(deck.upper),
            float(prize.value if prize.value is not None else 0),
            float(prize.lower), float(prize.upper),
            float(deck.age), float(prize.age),
        ])
    return cats, nums


def _event_features(snapshot: CausalSnapshot):
    cats: list[list[int]] = []
    nums: list[list[float]] = []
    newest = snapshot.recent_events[-1].source_event if snapshot.recent_events else -1
    for event in snapshot.recent_events:
        relative_actor = (
            0
            if event.actor is None
            else (1 if event.actor == snapshot.perspective_actor else 2)
        )
        payload = event.payload
        cats.append([
            event.log_type + 1, relative_actor, int(event.card_id or 0),
            int(event.from_area if event.from_area is not None else -1) + 1,
            int(event.to_area if event.to_area is not None else -1) + 1,
            int(event.identity_visible), int("serial" in payload),
            int("cardIdTarget" in payload),
        ])
        nums.append([
            float(newest - event.source_event), float(payload.get("value", 0) or 0),
            float(payload.get("putDamageCounter", 0) or 0),
            float(payload.get("head", 0) or 0),
        ])
    return cats, nums


def compile_row(row: Mapping[str, Any], snapshot: CausalSnapshot, registry: Any) -> dict[str, Any]:
    observation = row["actor_observation"]
    action = [int(value) for value in row["ordered_action"]]
    strict_codec = _m0010.IDOnlyCodec(_m0010.IDOnlyConfig())
    codec = strict_codec
    base = codec.encode(observation, action)
    if base is None:
        codec = _m0010.IDOnlyCodec(
            _m0010.IDOnlyConfig(max_action_steps=FULL_CACHE_MAX_ACTION_STEPS)
        )
        base = codec.encode(observation, action)
    if base is None:
        raise ValueError("full-cache legacy codec rejected an accepted raw decision")
    current = observation["current"]
    actor = current["yourIndex"]
    players = current["players"]
    deck_ids = _deck_cards(row)
    counts: dict[int, int] = {}
    for card_id in deck_ids:
        counts[card_id] = counts.get(card_id, 0) + 1
    registered_ids = sorted(counts)
    ledger_cat, ledger_num = _ledger_features(deck_ids, snapshot)
    event_cat, event_num = _event_features(snapshot)
    zone_inventory_num = []
    stadium = current.get("stadium", ())
    for player_index, player in _actor_relative_players(players, actor):
        if player_index == 1 - actor:
            known = len(snapshot.known_opponent_hand)
            unknown = snapshot.unknown_opponent_hand
        else:
            known = int(player.get("handCount", 0))
            unknown = 0
        stadium_count = sum(
            1
            for item in stadium
            if isinstance(item, Mapping) and item.get("playerIndex") == player_index
        ) if isinstance(stadium, Sequence) else 0
        zone_inventory_num.append(
            _zone_row(
                player,
                known_hand=known,
                unknown_hand=unknown,
                stadium_count=stadium_count,
            )
        )
    return {
        **base,
        "source_id": int(row["source_id"]),
        "a0_eligible": len(action) <= A0_MAX_ACTION_STEPS,
        "action_termination": row["action_termination"],
        "zone_inventory_num": zone_inventory_num,
        "registered_card_ids": registered_ids,
        "registered_multiplicity": [counts[item] for item in registered_ids],
        "ledger_cat": ledger_cat,
        "ledger_num": ledger_num,
        "event_cat": event_cat,
        "event_num": event_num,
        "known_opponent_hand_card_ids": [item.card_id for item in snapshot.known_opponent_hand],
        "known_opponent_hand_count": len(snapshot.known_opponent_hand),
        "unknown_opponent_hand_count": snapshot.unknown_opponent_hand,
    }


__all__ = [
    "A0_MAX_ACTION_STEPS",
    "COMPILER_VERSION",
    "FULL_CACHE_MAX_ACTION_STEPS",
    "compile_row",
    "compiler_sha256",
]

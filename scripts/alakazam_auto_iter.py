#!/usr/bin/env python3
"""Analyze Alakazam evaluator traces and compare AutoIter candidates.

This module deliberately does not choose simulator actions. It turns the
adjacent evaluator's JSON output into rule-aware metrics and compact cases so
strategy changes can be reviewed one hypothesis at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from evaluation.cli import main as evaluation_cli_main


ALAKAZAM = 743
KADABRA = 742
ABRA = 741
DUNSPARCE = 305
DUDUNSPARCE = 66
FEZANDIPITI_EX = 140
POWERFUL_HAND = 1072
PSYCHIC_ENERGY = 5
POFFIN = 1086
WONDROUS_PATCH = 1146
TELEPATH_ENERGY = 19
ENRICHING_ENERGY = 13
BASIC_PSYCHIC = 5
LANAS_AID = 1184
RARE_CANDY = 1079
ROCK_FIGHTING_ENERGY = 20
NIGHT_STRETCHER = 1097
ATTACK_LINE = {ABRA, KADABRA, ALAKAZAM}
READY_ATTACKERS = {KADABRA, ALAKAZAM}
THREE_PRIZE_POKEMON = {678, 723, 756}
TWO_PRIZE_POKEMON = {
    30,
    40,
    63,
    75,
    80,
    96,
    108,
    121,
    130,
    140,
    150,
    153,
    154,
    176,
    184,
    190,
    207,
    210,
    269,
    306,
    320,
    337,
    340,
    389,
    481,
    990,
    997,
    1071,
}
RULE_BOX_POKEMON = {FEZANDIPITI_EX, 306, 389, 481, 723, 997}

# 固定 catalog 的历史 meta 权重；未知 opponent 使用 0.05 fallback。
META_WEIGHTS = {
    "romanrozen_v9": 0.10,
    "pilkwang_v2": 0.08,
    "kokinn_search": 0.06,
    "penguin_915": 0.06,
    "crustle_wall": 0.08,
    "crustle_v1": 0.05,
    "kiyotah_lucario": 0.08,
    "kiyotah_dragapult": 0.06,
    "kiyotah_iono": 0.04,
    "kiyotah_abomasnow": 0.04,
    "kacchan_anti_wall": 0.06,
    "nursrijan_lucario": 0.05,
    "zoli_dragapult": 0.04,
    "sue_alakazam": 0.05,
    "maktha_1084": 0.08,
    "Agent_Lucario": 0.05,
    "Agent_Aluxian": 0.05,
}


@dataclass(frozen=True)
class EvaluationMetrics:
    games: int
    wins: int
    losses: int
    draws: int
    errors: int
    win_rate: float
    meta_weighted_win_rate: float
    second_turn_powerful_hand_games: int
    second_turn_powerful_hand_rate: float
    post_ko_count: int
    post_ko_zero_ready_count: int
    post_ko_zero_ready_event_rate: float
    games_with_post_ko_break: int
    games_with_post_ko_break_rate: float
    empty_bench_run_away_draw_count: int
    first_alakazam_turns: tuple[int, ...]


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    source: dict[str, Any]
    failure_class: str
    state_summary: dict[str, Any]
    legal_options: list[dict[str, Any]]
    actual_action: list[int]
    expected_action: list[int]
    expected_reason: str
    case_status: str


@dataclass(frozen=True)
class AnalysisResult:
    metrics: EvaluationMetrics
    cases: tuple[CaseRecord, ...]
    source_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    reasons: tuple[str, ...]
    regressions: tuple[str, ...]


def _as_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _trace_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("trace")
    return trace if isinstance(trace, list) else []


def _infer_agent_label(records: Iterable[dict[str, Any]], explicit: str | None) -> str | None:
    records = list(records)
    if explicit and any(
        step.get("role") == explicit
        for record in records
        for step in _trace_for_record(record)
    ):
        return explicit
    # Evaluator traces store the exact role label on every agent step. If a
    # stale CLI label is supplied (for example a submission directory name
    # without the evaluator's date suffix), silently treating all agent steps
    # as opponent steps produces a plausible but false 0% metric. Fall back
    # to the observed role instead.
    for record in records:
        label = record.get("label")
        if isinstance(label, str) and label and any(
            step.get("role") == label for step in _trace_for_record(record)
        ):
            return label
        for step in _trace_for_record(record):
            role = step.get("role")
            if isinstance(role, str) and role not in {"opponent", "finished"}:
                return role
    return None


def _observation(step: dict[str, Any]) -> dict[str, Any]:
    observation = step.get("observation")
    return observation if isinstance(observation, dict) else {}


def _current(step: dict[str, Any]) -> dict[str, Any]:
    current = _observation(step).get("current")
    if isinstance(current, dict):
        return current
    return {
        "turn": step.get("turn", 0),
        "yourIndex": step.get("yourIndex", 0),
        "players": step.get("players") or [],
    }


def _players(step: dict[str, Any]) -> list[dict[str, Any]]:
    players = _current(step).get("players")
    return players if isinstance(players, list) else []


def _player_at(step: dict[str, Any], index: int) -> dict[str, Any]:
    players = _players(step)
    if 0 <= index < len(players) and isinstance(players[index], dict):
        return players[index]
    return {}


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    cards = player.get(area) or []
    return [card for card in cards if isinstance(card, dict)]


def _active(player: dict[str, Any]) -> dict[str, Any] | None:
    cards = _cards(player, "active")
    return cards[0] if cards else None


def _bench(player: dict[str, Any]) -> list[dict[str, Any]]:
    return _cards(player, "bench")


def _field_pokemon(player: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the visible Active and Bench Pokémon in trace order."""
    active = _active(player)
    return ([active] if active is not None else []) + _bench(player)


def _option_list(step: dict[str, Any]) -> list[dict[str, Any]]:
    select = step.get("select")
    if not isinstance(select, dict):
        select = _observation(step).get("select")
    if not isinstance(select, dict):
        return []
    options = select.get("options")
    if options is None:
        options = select.get("option")
    return [option for option in options or [] if isinstance(option, dict)]


def _action_list(step: dict[str, Any]) -> list[int]:
    action = step.get("action")
    if not isinstance(action, list):
        return []
    return [value for value in action if isinstance(value, int)]


def _selected_options(step: dict[str, Any]) -> list[dict[str, Any]]:
    options = _option_list(step)
    return [
        options[index]
        for index in _action_list(step)
        if 0 <= index < len(options)
    ]


def _option_type(option: dict[str, Any]) -> int | str | None:
    value = option.get("type")
    if isinstance(value, (int, str)):
        return value
    return None


def _option_card_id(option: dict[str, Any]) -> int | None:
    for key in ("cardId", "card_id", "id"):
        value = _as_int(option.get(key))
        if value is not None:
            return value
    return None


def _option_card_id_in_step(
    option: dict[str, Any], step: dict[str, Any], player_index: int
) -> int | None:
    """Resolve field-target options whose card id is implicit in the area."""
    card_id = _option_card_id(option)
    if card_id is not None:
        return card_id
    area = _as_int(option.get("area", option.get("inPlayArea")))
    if area not in {4, 5}:
        return None
    owner = _as_int(option.get("playerIndex"), player_index)
    index = next(
        (
            _as_int(option.get(key))
            for key in ("indexInArea", "inPlayIndex")
            if _as_int(option.get(key)) is not None
        ),
        None,
    )
    if index is None:
        return None
    player = _player_at(step, owner if owner is not None else player_index)
    cards = _cards(player, "active" if area == 4 else "bench")
    return _as_int(cards[index].get("id")) if 0 <= index < len(cards) else None


def _option_attack_id(option: dict[str, Any]) -> int | None:
    return _as_int(option.get("attackId", option.get("attack_id")))


def _selected_attack_id(step: dict[str, Any]) -> int | None:
    for option in _selected_options(step):
        attack_id = _option_attack_id(option)
        if attack_id is not None:
            return attack_id
    return None


def _logs(step: dict[str, Any]) -> list[dict[str, Any]]:
    logs = _observation(step).get("logs")
    return logs if isinstance(logs, list) else []


def _field_state(step: dict[str, Any], player_index: int) -> dict[int, dict[str, Any]]:
    """Return the last visible field snapshot keyed by Pokémon serial."""
    player = _player_at(step, player_index)
    state: dict[int, dict[str, Any]] = {}
    for pokemon in [_active(player), *_bench(player)]:
        if not pokemon:
            continue
        serial = _as_int(pokemon.get("serial"))
        if serial is not None:
            state[serial] = pokemon
    return state


def _damage_counter_serials(step: dict[str, Any], player_index: int) -> set[int]:
    """Return Pokémon that received a real damage counter in this observation."""
    return {
        serial
        for log in _logs(step)
        if _as_int(log.get("type")) == 16
        and _as_int(log.get("playerIndex")) == player_index
        and log.get("putDamageCounter") is True
        for serial in [_as_int(log.get("serial"))]
        if serial is not None
    }


def _knockout_is_confirmed(
    step: dict[str, Any],
    log: dict[str, Any],
    player_index: int,
    previous_field: dict[int, dict[str, Any]],
) -> bool:
    """Require damage or an observed zero-HP state before calling a move KO."""
    serial = _as_int(log.get("serial"))
    if serial is None:
        return False
    if serial in _damage_counter_serials(step, player_index):
        return True
    previous = previous_field.get(serial)
    previous_hp = _as_int(previous.get("hp")) if previous is not None else None
    if previous_hp is not None and previous_hp <= 0:
        return True
    current = _field_state(step, player_index).get(serial)
    current_hp = _as_int(current.get("hp")) if current is not None else None
    return current_hp is not None and current_hp <= 0


def _agent_index(record: dict[str, Any]) -> int:
    value = _as_int(record.get("alakazamPhysicalIndex"), 0)
    return value if value in (0, 1) else 0


def _agent_turn_target(record: dict[str, Any]) -> int | None:
    agent_index = _agent_index(record)
    for step in _trace_for_record(record):
        first_player = _as_int(_current(step).get("firstPlayer"))
        if first_player in (0, 1):
            return 3 if first_player == agent_index else 4
    return None


def _is_agent_step(step: dict[str, Any], agent_label: str | None) -> bool:
    if agent_label is None:
        return step.get("role") not in {"opponent", "finished"}
    return step.get("role") == agent_label


def _pokemon_summary(card: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(card, dict):
        return None
    return {
        "id": card.get("id"),
        "serial": card.get("serial"),
        "hp": card.get("hp"),
        "maxHp": card.get("maxHp"),
        "energies": list(card.get("energies") or []),
        "energy_count": len(card.get("energies") or card.get("energyCards") or []),
    }


def _state_summary(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "active": _pokemon_summary(_active(player)),
        "bench": [_pokemon_summary(card) for card in _bench(player)],
        "hand_count": _as_int(player.get("handCount"), len(player.get("hand") or [])) or 0,
        "deck_count": _as_int(player.get("deckCount"), 0) or 0,
        "prize_count": len(player.get("prize") or []),
    }


def _ready_attacker_count(player: dict[str, Any]) -> int:
    count = 0
    for pokemon in [_active(player), *_bench(player)]:
        if pokemon and _as_int(pokemon.get("id")) in READY_ATTACKERS:
            energies = pokemon.get("energies") or []
            if PSYCHIC_ENERGY in energies:
                count += 1
    return count


def _attack_line_state(player: dict[str, Any]) -> dict[str, int]:
    field = [_active(player), *_bench(player)]
    return {
        "abra_count": sum(_as_int(card.get("id")) == ABRA for card in field if card),
        "kadabra_count": sum(_as_int(card.get("id")) == KADABRA for card in field if card),
        "alakazam_count": sum(_as_int(card.get("id")) == ALAKAZAM for card in field if card),
        "ready_attacker_count": _ready_attacker_count(player),
    }


def _case_source(record: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    return {
        "opponent": record.get("opponent", "unknown"),
        "game": record.get("game"),
        "turn": _as_int(step.get("turn"), _as_int(_current(step).get("turn"), 0)) or 0,
    }


def _case(
    record: dict[str, Any],
    step: dict[str, Any],
    failure_class: str,
    reason: str,
    status: str = "diagnostic",
) -> CaseRecord:
    source = _case_source(record, step)
    case_id = f"{source['opponent']}-game-{source['game']}-turn-{source['turn']}-{failure_class}"
    agent_player = _player_at(step, _agent_index(record))
    return CaseRecord(
        case_id=case_id,
        source=source,
        failure_class=failure_class,
        state_summary={
            **_state_summary(agent_player),
            "attack_line": _attack_line_state(agent_player),
        },
        legal_options=_option_list(step),
        actual_action=_action_list(step),
        expected_action=[],
        expected_reason=reason,
        case_status=status,
    )


def _hand_option_card_id(step: dict[str, Any], option: dict[str, Any], player_index: int) -> int | None:
    """Resolve a main-action card option from the visible hand snapshot."""
    option_type = _as_int(option.get("type"))
    if option_type not in {7, 8}:
        return None
    explicit_card_id = _as_int(option.get("cardId"))
    if explicit_card_id is not None:
        return explicit_card_id
    hand_index = _as_int(option.get("indexInArea"))
    if hand_index is None:
        observation = _observation(step)
        raw_options = (observation.get("select") or {}).get("option") or []
        option_index = _as_int(option.get("index"))
        if option_index is not None and 0 <= option_index < len(raw_options):
            hand_index = _as_int(raw_options[option_index].get("index"))
    if hand_index is None:
        return None
    hand = _player_at(step, player_index).get("hand") or []
    if 0 <= hand_index < len(hand) and isinstance(hand[hand_index], dict):
        return _as_int(hand[hand_index].get("id"))
    return None


def _field_target_id(step: dict[str, Any], option: dict[str, Any], player_index: int) -> int | None:
    """Resolve a main-action field target from the compact trace option."""
    area = _as_int(option.get("inPlayArea"))
    index = _as_int(option.get("inPlayIndex"))
    if area not in {4, 5} or index is None:
        return None
    player = _player_at(step, player_index)
    cards = _cards(player, "active" if area == 4 else "bench")
    if 0 <= index < len(cards):
        return _as_int(cards[index].get("id"))
    return None


def _powerful_hand_damage(
    player: dict[str, Any], active: dict[str, Any] | None, extra_hand: int = 0
) -> int:
    """Use the evaluator's existing hand-size damage model consistently."""
    hand_count = _as_int(player.get("handCount"), len(player.get("hand") or [])) or 0
    hand_count += extra_hand
    return hand_count * 20 if _as_int((active or {}).get("id")) == ALAKAZAM else 30


def _selected_action_closes_prize(step: dict[str, Any], player_index: int) -> bool:
    """Recognize a selected action that can finish the remaining prize count."""
    selected = _selected_options(step)
    selected_attack = _selected_attack_id(step) == POWERFUL_HAND
    selected_enriching = any(
        _option_type(option) == 8
        and _hand_option_card_id(step, option, player_index) == ENRICHING_ENERGY
        and _field_target_id(step, option, player_index) == DUDUNSPARCE
        for option in selected
    )
    if not selected_attack and not selected_enriching:
        return False

    player = _player_at(step, player_index)
    active = _active(player)
    target = _active(_player_at(step, 1 - player_index))
    if not active or not target:
        return False
    remaining_prizes = len(player.get("prize") or [])
    target_hp = _as_int(target.get("hp"), 9999) or 9999
    target_prizes = _prize_value(_as_int(target.get("id")))
    if remaining_prizes > target_prizes:
        return False

    # Dudunsparce's selected Enriching Energy route is evaluated after its
    # documented net hand-size gain, but only when a Powerful Hand option is
    # still exposed in the same decision state.
    extra_hand = 3 if selected_enriching and not selected_attack else 0
    if selected_enriching and not any(
        _option_type(option) == 13 and _option_attack_id(option) == POWERFUL_HAND
        for option in _option_list(step)
    ):
        return False
    return _powerful_hand_damage(player, active, extra_hand) >= target_hp


def _handoff_preparation_missed(step: dict[str, Any], player_index: int) -> bool:
    """Detect the visible handoff case that V4 is intended to repair."""
    player = _player_at(step, player_index)
    active = _active(player)
    if not active or _as_int(active.get("id")) != ALAKAZAM:
        return False
    if PSYCHIC_ENERGY not in (active.get("energies") or []):
        return False
    hand = {
        _as_int(card.get("id"))
        for card in player.get("hand") or []
        if isinstance(card, dict) and _as_int(card.get("id")) is not None
    }
    bench_targets = {
        index
        for index, card in enumerate(_bench(player))
        if _as_int(card.get("id")) in {KADABRA, ALAKAZAM}
        and PSYCHIC_ENERGY not in (card.get("energies") or [])
        and not card.get("appearThisTurn", False)
    }
    if not bench_targets:
        return False

    options = _option_list(step)
    has_direct_attachment = any(
        _option_type(option) == 8
        and _hand_option_card_id(step, option, player_index)
        in {BASIC_PSYCHIC, TELEPATH_ENERGY}
        and _as_int(option.get("inPlayArea")) == 5
        and _as_int(option.get("inPlayIndex")) in bench_targets
        for option in options
    )
    has_lana_route = any(
        _option_type(option) == 7
        and _hand_option_card_id(step, option, player_index) == LANAS_AID
        and BASIC_PSYCHIC in {
            _as_int(card.get("id"))
            for card in player.get("discard") or []
            if isinstance(card, dict)
        }
        for option in options
    )
    if not (has_direct_attachment or has_lana_route):
        return False

    selected = _selected_options(step)
    selected_handoff = any(
        _option_type(option) == 8
        and _hand_option_card_id(step, option, player_index)
        in {BASIC_PSYCHIC, TELEPATH_ENERGY}
        and _as_int(option.get("inPlayArea")) == 5
        and _as_int(option.get("inPlayIndex")) in bench_targets
        for option in selected
    ) or any(
        _option_type(option) == 7
        and _hand_option_card_id(step, option, player_index) == LANAS_AID
        for option in selected
    )
    if selected_handoff:
        return False

    selected_enriching_to_draw_engine = any(
        _option_type(option) == 8
        and _hand_option_card_id(step, option, player_index) == ENRICHING_ENERGY
        and _field_target_id(step, option, player_index) == DUDUNSPARCE
        for option in selected
    )
    selected_attack = _selected_attack_id(step) == POWERFUL_HAND
    if _selected_action_closes_prize(step, player_index):
        return False
    return selected_enriching_to_draw_engine or selected_attack


def _prize_value(card_id: int | None) -> int:
    if card_id in THREE_PRIZE_POKEMON:
        return 3
    if card_id in TWO_PRIZE_POKEMON or card_id in RULE_BOX_POKEMON:
        return 2
    return 1


def _has_visible_bench_handoff(step: dict[str, Any], player_index: int) -> bool:
    player = _player_at(step, player_index)
    hand_ids = {
        _as_int(card.get("id"))
        for card in player.get("hand") or []
        if isinstance(card, dict) and _as_int(card.get("id")) is not None
    }
    for pokemon in _bench(player):
        pokemon_id = _as_int(pokemon.get("id"))
        energies = pokemon.get("energies") or []
        if pokemon_id in {KADABRA, ALAKAZAM} and PSYCHIC_ENERGY in energies:
            return True
        if (
            pokemon_id == ABRA
            and PSYCHIC_ENERGY in energies
            and (KADABRA in hand_ids or (ALAKAZAM in hand_ids and RARE_CANDY in hand_ids))
        ):
            return True
    return False


def _lanas_aid_successor_route_available(step: dict[str, Any], player_index: int) -> bool:
    """Recognize Lana's Aid as a complete recovery-to-Bench handoff route."""
    player = _player_at(step, player_index)
    bench_space = _as_int(player.get("benchMax"), 5) - len(_bench(player))
    discard_ids = {
        _as_int(card.get("id"))
        for card in player.get("discard") or []
        if isinstance(card, dict)
    }
    return bench_space > 0 and ABRA in discard_ids and BASIC_PSYCHIC in discard_ids


def _night_stretcher_handoff_route_available(
    step: dict[str, Any], player_index: int
) -> bool:
    """Recognize a Night Stretcher route that can actually seed the Bench.

    Night Stretcher returns one Pokémon or one basic Energy to hand.  For an
    empty Bench, the useful immediate route is therefore either a discarded
    Abra plus a Psychic Energy already in hand, or an Abra already in hand
    plus a recoverable Psychic Energy.  Merely holding Night Stretcher, or
    recovering a Stage 1/Stage 2 Pokémon, is not enough to count as Bench
    insurance.
    """
    player = _player_at(step, player_index)
    bench_space = _as_int(player.get("benchMax"), 5) - len(_bench(player))
    if bench_space <= 0:
        return False
    if any(_as_int(pokemon.get("id")) in ATTACK_LINE for pokemon in _bench(player)):
        return False

    hand_ids = {
        _as_int(card.get("id"))
        for card in player.get("hand") or []
        if isinstance(card, dict)
    }
    discard_ids = {
        _as_int(card.get("id"))
        for card in player.get("discard") or []
        if isinstance(card, dict)
    }
    psychic_energy_ids = {BASIC_PSYCHIC, TELEPATH_ENERGY}
    return (
        ABRA in discard_ids and bool(hand_ids & psychic_energy_ids)
    ) or (
        ABRA in hand_ids and bool(discard_ids & psychic_energy_ids)
    )


def _selected_night_stretcher_handoff(
    step: dict[str, Any], player_index: int
) -> bool:
    return any(
        _option_type(option) == 7
        and _hand_option_card_id(step, option, player_index) == NIGHT_STRETCHER
        and _night_stretcher_handoff_route_available(step, player_index)
        for option in _selected_options(step)
    )


def _wondrous_patch_handoff_route_available(
    step: dict[str, Any], player_index: int
) -> bool:
    """Require a visible evolution route before treating Patch as handoff work."""
    player = _player_at(step, player_index)
    hand_ids = {
        _as_int(card.get("id"))
        for card in player.get("hand") or []
        if isinstance(card, dict)
    }
    discard_ids = {
        _as_int(card.get("id"))
        for card in player.get("discard") or []
        if isinstance(card, dict)
    }
    if BASIC_PSYCHIC not in discard_ids:
        return False
    for pokemon in _bench(player):
        pokemon_id = _as_int(pokemon.get("id"))
        if PSYCHIC_ENERGY in (pokemon.get("energies") or []):
            continue
        if pokemon_id in {KADABRA, ALAKAZAM}:
            return True
        if pokemon_id == ABRA and (
            KADABRA in hand_ids
            or (ALAKAZAM in hand_ids and RARE_CANDY in hand_ids)
        ):
            return True
    return False


def _selected_bench_anchor(step: dict[str, Any], player_index: int) -> bool:
    player = _player_at(step, player_index)
    for option in _selected_options(step):
        card_id = _hand_option_card_id(step, option, player_index)
        if _option_type(option) == 7 and card_id in {POFFIN, ABRA, DUNSPARCE}:
            return True
        if (
            _option_type(option) == 7
            and card_id == WONDROUS_PATCH
            and _wondrous_patch_handoff_route_available(step, player_index)
        ):
            return True
        if (
            _option_type(option) == 7
            and card_id == LANAS_AID
            and (
                _lanas_aid_successor_route_available(step, player_index)
                or (
                    any(
                        _as_int(card.get("id")) in ATTACK_LINE
                        for card in _bench(player)
                        if PSYCHIC_ENERGY not in (card.get("energies") or [])
                    )
                    and BASIC_PSYCHIC in {
                        _as_int(card.get("id"))
                        for card in player.get("discard") or []
                        if isinstance(card, dict)
                    }
                )
            )
        ):
            return True
        if _selected_night_stretcher_handoff(step, player_index):
            return True
        if (
            _option_type(option) == 8
            and card_id == TELEPATH_ENERGY
            and _field_target_id(step, option, player_index) in ATTACK_LINE
        ):
            return True
    return False


def _selected_draw_engine_progress(step: dict[str, Any], player_index: int) -> bool:
    """Recognize a valid draw-engine action without calling it a Bench miss.

    The strategy contract explicitly allows Enriching Energy to go to a
    visible Dunsparce/Dudunsparce route. That action does not itself create a
    ready Abra-line attacker, but it is still intentional progress when the
    Bench is non-empty; reporting it as a missed Bench anchor would turn a
    valid resource choice into a false failure case. An empty Bench remains a
    hard insurance issue and is deliberately not covered here.
    """
    player = _player_at(step, player_index)
    if not _bench(player):
        return False
    for option in _selected_options(step):
        if _option_type(option) != 8:
            continue
        if _hand_option_card_id(step, option, player_index) != ENRICHING_ENERGY:
            continue
        if _field_target_id(step, option, player_index) in {DUNSPARCE, DUDUNSPARCE}:
            return True
    return False


def _has_direct_handoff_option(step: dict[str, Any], player_index: int) -> bool:
    """Do not call Bench insurance a miss when handoff work is exposed."""
    player = _player_at(step, player_index)
    for option in _option_list(step):
        target = _field_target_id(step, option, player_index)
        if _option_type(option) == 8:
            energy_id = _hand_option_card_id(step, option, player_index)
            if energy_id in {BASIC_PSYCHIC, TELEPATH_ENERGY} and target in ATTACK_LINE:
                area = _as_int(option.get("inPlayArea"))
                target_index = _as_int(option.get("inPlayIndex"))
                if area == 4:
                    target_card = _active(player)
                elif area == 5 and target_index is not None:
                    bench = _bench(player)
                    target_card = (
                        bench[target_index]
                        if 0 <= target_index < len(bench)
                        else None
                    )
                else:
                    target_card = None
                if target_card and PSYCHIC_ENERGY not in (target_card.get("energies") or []):
                    return True
        if _option_type(option) == 9 and target in {ABRA, KADABRA}:
            return True
    return False


def _bench_insurance_due(step: dict[str, Any], player_index: int) -> bool:
    """Detect a non-terminal state without a visible Abra-line handoff."""
    player = _player_at(step, player_index)
    active = _active(player)
    if not active or _as_int(active.get("id")) not in {ALAKAZAM, KADABRA}:
        return False
    hand = player.get("hand") or []
    bench_space = _as_int(player.get("benchMax"), 5) - len(_bench(player))
    if (
        bench_space <= 0
        or _has_visible_bench_handoff(step, player_index)
        or _has_direct_handoff_option(step, player_index)
    ):
        return False
    bench_attack_line = [
        pokemon
        for pokemon in _bench(player)
        if _as_int(pokemon.get("id")) in ATTACK_LINE
    ]
    if bench_attack_line:
        # A non-empty, already-established Abra-line is still Bench
        # continuity even when it is not immediately ready to attack.  Only
        # the narrow fresh-Bench state is a genuine insurance gate: every
        # visible line piece entered this turn and is still unenergized.
        freshly_unready = all(
            pokemon.get("appearThisTurn", False)
            and PSYCHIC_ENERGY not in (pokemon.get("energies") or [])
            for pokemon in bench_attack_line
        )
        selected_valid_progress = _selected_bench_anchor(step, player_index) or _selected_draw_engine_progress(
            step, player_index
        )
        if not freshly_unready and not selected_valid_progress:
            return False
    options = _option_list(step)
    if not any(_option_type(option) == 13 for option in options):
        return False
    has_anchor_option = any(
        (
            _option_type(option) == 7
            and _hand_option_card_id(step, option, player_index)
            in {POFFIN, ABRA, DUNSPARCE}
        )
        or (
            _option_type(option) == 7
            and _hand_option_card_id(step, option, player_index) == WONDROUS_PATCH
            and _wondrous_patch_handoff_route_available(step, player_index)
        )
        or (
            _option_type(option) == 7
            and _hand_option_card_id(step, option, player_index) == LANAS_AID
            and _lanas_aid_successor_route_available(step, player_index)
        )
        or (
            _option_type(option) == 7
            and _hand_option_card_id(step, option, player_index) == NIGHT_STRETCHER
            and _night_stretcher_handoff_route_available(step, player_index)
        )
        or (
            _option_type(option) == 8
            and _hand_option_card_id(step, option, player_index) == TELEPATH_ENERGY
            and _field_target_id(step, option, player_index) in ATTACK_LINE
        )
        for option in options
    )
    if not has_anchor_option:
        return False

    target = _active(_player_at(step, 1 - player_index))
    hand_count = _as_int(player.get("handCount"), len(hand)) or 0
    damage = hand_count * 20 if _as_int(active.get("id")) == ALAKAZAM else 30
    remaining_prizes = len(player.get("prize") or [])
    target_is_final_ko = bool(
        target
        and _as_int(target.get("hp"), 9999) <= damage
        and remaining_prizes <= _prize_value(_as_int(target.get("id")))
    )
    return not target_is_final_ko


def _future_same_turn_bench_anchor_selected(
    trace: list[dict[str, Any]],
    step_index: int,
    player_index: int,
    agent_label: str | None,
) -> bool:
    """Treat a later same-turn anchor as validation of an earlier search step."""
    if not (0 <= step_index < len(trace)):
        return False
    current_turn = _as_int(
        trace[step_index].get("turn"), _as_int(_current(trace[step_index]).get("turn"))
    )
    if current_turn is None:
        return False
    for future_step in trace[step_index + 1 :]:
        future_turn = _as_int(
            future_step.get("turn"), _as_int(_current(future_step).get("turn"))
        )
        if future_turn != current_turn:
            break
        if not _is_agent_step(future_step, agent_label):
            continue
        if _selected_bench_anchor(future_step, player_index) or _selected_draw_engine_progress(
            future_step, player_index
        ):
            return True
        if _selected_attack_id(future_step) is not None or any(
            _option_type(option) == 14 for option in _selected_options(future_step)
        ):
            return False
    return False


def _record_error_is_agent(record: dict[str, Any], agent_label: str | None) -> bool:
    """Attribute an evaluator error to the last side that acted."""
    if not record.get("error"):
        return False
    for step in reversed(_trace_for_record(record)):
        role = step.get("role")
        if role == "finished":
            continue
        if role == "opponent":
            return False
        return _is_agent_step(step, agent_label)
    return True


def _record_win(record: dict[str, Any]) -> bool:
    return _as_int(record.get("winner"), -1) == 0


def _weighted_win_rate(records: list[dict[str, Any]]) -> float:
    wins_by_opponent: Counter[str] = Counter()
    games_by_opponent: Counter[str] = Counter()
    for record in records:
        opponent = str(record.get("opponent", "unknown"))
        games_by_opponent[opponent] += 1
        if _record_win(record):
            wins_by_opponent[opponent] += 1
    weighted_sum = 0.0
    total_weight = 0.0
    for opponent, games in games_by_opponent.items():
        weight = META_WEIGHTS.get(opponent, 0.05)
        weighted_sum += (wins_by_opponent[opponent] / games) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def analyze_records(
    records: list[dict[str, Any]], agent_label: str | None = None
) -> AnalysisResult:
    """Analyze already-loaded evaluator records without touching the engine."""
    label = _infer_agent_label(records, agent_label)
    wins = sum(_record_win(record) for record in records)
    losses = sum(_as_int(record.get("winner"), -1) == 1 for record in records)
    draws = len(records) - wins - losses
    errors = sum(_record_error_is_agent(record, label) for record in records)
    powerful_games = 0
    post_ko_count = 0
    post_ko_zero_ready_count = 0
    games_with_post_ko_break = 0
    empty_bench_draws = 0
    first_alakazam_turns: list[int] = []
    cases: list[CaseRecord] = []

    for record in records:
        trace = _trace_for_record(record)
        target_turn = _agent_turn_target(record)
        second_turn_seen = False
        second_turn_step: dict[str, Any] | None = None
        powerful_hand_step: dict[str, Any] | None = None
        powerful_hand_available = False
        seen_knockouts: set[tuple[Any, Any]] = set()
        game_has_break = False
        first_alakazam: int | None = None
        agent_index = _agent_index(record)
        previous_field: dict[int, dict[str, Any]] = {}
        seen_strategy_cases: set[tuple[str, str]] = set()

        for step_index, step in enumerate(trace):
            player = _player_at(step, agent_index)
            if first_alakazam is None and any(
                _as_int(card.get("id")) == ALAKAZAM
                for card in [_active(player), *_bench(player)]
                if card
            ):
                first_alakazam = _as_int(step.get("turn"), 0) or 0

            if not _is_agent_step(step, label):
                # KO logs are attached to the first observation after the
                # opponent action, even if the next selection belongs to us.
                pass

            if _is_agent_step(step, label):
                if _bench_insurance_due(step, agent_index):
                    anchor_selected = (
                        _selected_bench_anchor(step, agent_index)
                        or _selected_draw_engine_progress(step, agent_index)
                        or _future_same_turn_bench_anchor_selected(
                            trace, step_index, agent_index, label
                        )
                    )
                    signature = json.dumps(
                        _state_summary(player), ensure_ascii=False, sort_keys=True
                    )
                    case_key = ("bench_insurance_missed", signature)
                    if case_key not in seen_strategy_cases:
                        seen_strategy_cases.add(case_key)
                        cases.append(
                            _case(
                                record,
                                step,
                                "bench_insurance_missed",
                                "当前攻击前没有可验证的 Abra-line handoff；应先用 Poffin、Telepath 或直接放下 Basic 建立 Bench，除非攻击完成最后奖赏闭环",
                                status="pass" if anchor_selected else "fail",
                            )
                        )
                if _handoff_preparation_missed(step, agent_index):
                    signature = json.dumps(
                        _state_summary(player), ensure_ascii=False, sort_keys=True
                    )
                    case_key = ("handoff_preparation_missed", signature)
                    if case_key not in seen_strategy_cases:
                        seen_strategy_cases.add(case_key)
                        cases.append(
                            _case(
                                record,
                                step,
                                "handoff_preparation_missed",
                                "Active Alakazam 已能攻击，但可见 Bench 接力线有 Psychic 附能或 Lana's Aid 回收路径；不应先走 Enriching Energy 或直接攻击",
                                status="fail",
                            )
                        )
                selected = _selected_options(step)
                attack_id = _selected_attack_id(step)
                is_second_turn = (
                    target_turn in (3, 4) and _as_int(step.get("turn")) == target_turn
                ) or (
                    target_turn is None and _as_int(step.get("turn")) in (3, 4)
                )
                if is_second_turn:
                    second_turn_step = step
                    if any(
                        _option_type(option) == 13
                        and _option_attack_id(option) == POWERFUL_HAND
                        for option in _option_list(step)
                    ):
                        powerful_hand_available = True
                        if powerful_hand_step is None:
                            powerful_hand_step = step
                    if attack_id == POWERFUL_HAND and not second_turn_seen:
                        second_turn_seen = True
                        powerful_games += 1

                active = _active(player)
                if (
                    active
                    and _as_int(active.get("id")) == DUDUNSPARCE
                    and not _bench(player)
                    and any(
                        _option_type(option) == 10
                        and _option_card_id_in_step(option, step, agent_index)
                        == DUDUNSPARCE
                        for option in selected
                    )
                ):
                    empty_bench_draws += 1
                    cases.append(
                        _case(
                            record,
                            step,
                            "empty_bench_run_away_draw",
                            "Active Dudunsparce 在空 Bench 时不应选择 Run Away Draw",
                            status="fail",
                        )
                    )

            for log in _logs(step):
                if not isinstance(log, dict):
                    continue
                if (
                    _as_int(log.get("type")) != 6
                    or _as_int(log.get("playerIndex")) != agent_index
                    or _as_int(log.get("fromArea")) not in {4, 5}
                    or _as_int(log.get("toArea")) != 3
                    or _as_int(log.get("cardId")) not in ATTACK_LINE
                    or not _knockout_is_confirmed(step, log, agent_index, previous_field)
                ):
                    continue
                key = (log.get("serial"), log.get("toArea"))
                if key in seen_knockouts:
                    continue
                seen_knockouts.add(key)
                post_ko_count += 1
                ready_count = _ready_attacker_count(player)
                if ready_count == 0:
                    post_ko_zero_ready_count += 1
                    game_has_break = True
                    cases.append(
                        _case(
                            record,
                            step,
                            "post_ko_no_ready_attacker",
                            "我方 Pokémon 被击倒后没有可立即接班的 Kadabra/Alakazam",
                        )
                    )

            previous_field = _field_state(step, agent_index)

        if first_alakazam is not None:
            first_alakazam_turns.append(first_alakazam)
        if game_has_break:
            games_with_post_ko_break += 1
        if not second_turn_seen:
            failure_class = (
                "second_turn_powerful_hand_missing"
                if powerful_hand_available
                else "second_turn_powerful_hand_unavailable"
            )
            reason = (
                "第二回合存在合法的 Alakazam Powerful Hand，但实际 action 没有选择它；"
                "需要结合当时的铺场、进化、过牌和终局判断复盘"
                if powerful_hand_available
                else "目标第二回合没有合法的 Alakazam Powerful Hand option；"
                "这属于资源或规则条件不可用，不计为策略漏攻"
            )
            cases.append(
                _case(
                    record,
                    powerful_hand_step or second_turn_step or (trace[-1] if trace else {}),
                    failure_class,
                    reason,
                )
            )

    games = len(records)
    return AnalysisResult(
        metrics=EvaluationMetrics(
            games=games,
            wins=wins,
            losses=losses,
            draws=draws,
            errors=errors,
            win_rate=wins / games if games else 0.0,
            meta_weighted_win_rate=_weighted_win_rate(records),
            second_turn_powerful_hand_games=powerful_games,
            second_turn_powerful_hand_rate=powerful_games / games if games else 0.0,
            post_ko_count=post_ko_count,
            post_ko_zero_ready_count=post_ko_zero_ready_count,
            post_ko_zero_ready_event_rate=(
                post_ko_zero_ready_count / post_ko_count if post_ko_count else 0.0
            ),
            games_with_post_ko_break=games_with_post_ko_break,
            games_with_post_ko_break_rate=(
                games_with_post_ko_break / games if games else 0.0
            ),
            empty_bench_run_away_draw_count=empty_bench_draws,
            first_alakazam_turns=tuple(first_alakazam_turns),
        ),
        cases=tuple(cases),
    )


def _iter_game_files(report_dir: Path) -> list[Path]:
    old_game_files = report_dir.rglob("game_*.json")
    retained_trace_files = (report_dir / "traces").glob("*.json")
    return sorted(
        {
            path
            for path in [*old_game_files, *retained_trace_files]
            if path.is_file() and path.name != "summary.json"
        }
    )


def _record_from_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten native TraceStore's nested result without changing old records."""
    record = dict(payload)
    nested_result = payload.get("result")
    if isinstance(nested_result, dict):
        for key, value in nested_result.items():
            record.setdefault(key, value)
    return record


def analyze_report(report_dir: Path, agent_label: str | None = None) -> AnalysisResult:
    """Load summary/game JSON files recursively and analyze their full traces."""
    report_dir = report_dir.resolve()
    files = _iter_game_files(report_dir)
    records: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records.append(_record_from_report_payload(payload))

    if not records:
        summary_path = report_dir / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            records = payload.get("results") or []
            if isinstance(payload.get("label"), str) and agent_label is None:
                agent_label = payload["label"]

    result = analyze_records(records, agent_label=agent_label)
    return replace(
        result,
        source_files=tuple(str(path.relative_to(report_dir)) for path in files),
    )


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def _metrics_payload(result: AnalysisResult) -> dict[str, Any]:
    return {
        "metrics": asdict(result.metrics),
        "source_files": list(result.source_files),
        "case_count": len(result.cases),
    }


def _render_analysis(result: AnalysisResult, manifest: dict[str, Any]) -> str:
    metrics = result.metrics
    by_failure = Counter(case.failure_class for case in result.cases)
    lines = [
        "# Alakazam AutoIter 分析",
        "",
        "## 样本与结果",
        "",
        f"- 实际读取 trace 文件：{len(result.source_files)}",
        f"- 对局：{metrics.games}；胜 / 负 / 平：{metrics.wins} / {metrics.losses} / {metrics.draws}",
        f"- 我方错误：{metrics.errors}",
        f"- 胜率：{_percentage(metrics.win_rate)}",
        f"- Meta 加权胜率：{_percentage(metrics.meta_weighted_win_rate)}",
        f"- 第二回合 Powerful Hand：{metrics.second_turn_powerful_hand_games}/{metrics.games} ({_percentage(metrics.second_turn_powerful_hand_rate)})",
        f"- 我方被击倒事件：{metrics.post_ko_count}",
        f"- 击倒后无 ready attacker：{metrics.post_ko_zero_ready_count}/{metrics.post_ko_count} ({_percentage(metrics.post_ko_zero_ready_event_rate)})",
        f"- 出现过打手断档的对局：{metrics.games_with_post_ko_break}/{metrics.games} ({_percentage(metrics.games_with_post_ko_break_rate)})",
        f"- 空 Bench Run Away Draw：{metrics.empty_bench_run_away_draw_count}",
        "",
        "## Case 摘要",
        "",
        "| failure_class | 数量 |",
        "|---|---:|",
    ]
    for failure_class, count in sorted(by_failure.items()):
        lines.append(f"| {failure_class} | {count} |")
    if not by_failure:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## 评测配置",
            "",
            "```json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(
    result: AnalysisResult, output_dir: Path, manifest: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(_metrics_payload(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in result.cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    (output_dir / "analysis.md").write_text(
        _render_analysis(result, manifest), encoding="utf-8"
    )


def _load_metrics(path: Path) -> EvaluationMetrics:
    payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    values = payload.get("metrics", payload)
    if not isinstance(values, dict):
        raise ValueError(f"metrics.json must contain an object: {path}")
    if "games" in values:
        legacy_values = dict(values)
        legacy_values["first_alakazam_turns"] = tuple(
            legacy_values.get("first_alakazam_turns") or []
        )
        fields = set(EvaluationMetrics.__dataclass_fields__)
        return EvaluationMetrics(
            **{key: value for key, value in legacy_values.items() if key in fields}
        )

    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise ValueError("native metrics.json requires sibling summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"summary.json must contain an object: {summary_path}")
    games = _as_int(summary.get("total_games"), 0) or 0
    wins = _as_int(summary.get("wins"), 0) or 0
    losses = _as_int(summary.get("losses"), 0) or 0
    draws = _as_int(summary.get("draws"), 0) or 0
    errors = _as_int(summary.get("errors"), 0) or 0

    def metric(metric_id: str) -> dict[str, Any]:
        value = values.get(metric_id, {})
        return value if isinstance(value, dict) else {}

    powerful_hand = metric("powerful_hand")
    powerful_games = _as_int(powerful_hand.get("numerator"), 0) or 0
    post_ko_relay = metric("post_ko_relay")
    post_ko_zero_ready = _as_int(post_ko_relay.get("numerator"), 0) or 0
    post_ko_count = _as_int(post_ko_relay.get("denominator"), 0) or 0
    run_away_draw = metric("run_away_draw")
    empty_bench_draws = _as_int(run_away_draw.get("numerator"), 0) or 0
    games_with_post_ko_break = sum(
        _as_int(group.get("games_with_zero_ready"), 0) or 0
        for group in post_ko_relay.get("by_opponent", {}).values()
        if isinstance(group, dict)
    )

    return EvaluationMetrics(
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        errors=errors,
        win_rate=wins / games if games else 0.0,
        meta_weighted_win_rate=wins / games if games else 0.0,
        second_turn_powerful_hand_games=powerful_games,
        second_turn_powerful_hand_rate=powerful_games / games if games else 0.0,
        post_ko_count=post_ko_count,
        post_ko_zero_ready_count=post_ko_zero_ready,
        post_ko_zero_ready_event_rate=(
            post_ko_zero_ready / post_ko_count if post_ko_count else 0.0
        ),
        games_with_post_ko_break=games_with_post_ko_break,
        games_with_post_ko_break_rate=(
            games_with_post_ko_break / games if games else 0.0
        ),
        empty_bench_run_away_draw_count=empty_bench_draws,
        first_alakazam_turns=(),
    )


def _load_cases(path: Path) -> list[dict[str, Any]]:
    case_path = path / "cases.jsonl"
    if not case_path.exists():
        return []
    cases: list[dict[str, Any]] = []
    for line in case_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def decide_promotion(
    control: EvaluationMetrics,
    candidate: EvaluationMetrics,
    target_case_improved: bool,
    independent_pairs: list[tuple[float, float]] | None = None,
) -> PromotionDecision:
    """Apply correctness, case, signal and soft outcome gates."""
    reasons: list[str] = []
    regressions: list[str] = []
    if candidate.errors > 0:
        reasons.append("correctness")
    if candidate.empty_bench_run_away_draw_count > 0:
        reasons.append("empty_bench_run_away_draw")
    if not target_case_improved:
        reasons.append("target_case")

    if candidate.second_turn_powerful_hand_rate < control.second_turn_powerful_hand_rate:
        regressions.append("second_turn_powerful_hand")
    if candidate.post_ko_zero_ready_event_rate > control.post_ko_zero_ready_event_rate:
        regressions.append("post_ko_zero_ready")
    if candidate.meta_weighted_win_rate < control.meta_weighted_win_rate:
        regressions.append("meta_weighted_win_rate")

    repeated_win_decline = bool(independent_pairs) and all(
        candidate_rate < control_rate
        for control_rate, candidate_rate in independent_pairs or []
    )
    if repeated_win_decline:
        reasons.append("win_rate")

    if reasons and any(reason in {"correctness", "empty_bench_run_away_draw", "win_rate"} for reason in reasons):
        status = "reject"
    elif not target_case_improved or regressions:
        status = "observe"
    else:
        status = "accept"
    return PromotionDecision(status, tuple(reasons), tuple(regressions))


def compare_reports(control_dir: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write comparison.json and decision.md for one candidate/control pair."""
    control_metrics = _load_metrics(control_dir)
    candidate_metrics = _load_metrics(candidate_dir)
    control_cases = _load_cases(control_dir)
    candidate_cases = _load_cases(candidate_dir)
    control_failures = Counter(
        case.get("failure_class")
        for case in control_cases
        if case.get("case_status") == "fail"
    )
    candidate_failures = Counter(
        case.get("failure_class")
        for case in candidate_cases
        if case.get("case_status") == "fail"
    )
    target_case_improved = sum(candidate_failures.values()) < sum(control_failures.values())
    decision = decide_promotion(
        control_metrics,
        candidate_metrics,
        target_case_improved=target_case_improved,
    )
    payload = {
        "status": decision.status,
        "reasons": list(decision.reasons),
        "regressions": list(decision.regressions),
        "control": asdict(control_metrics),
        "candidate": asdict(candidate_metrics),
        "control_failure_counts": dict(control_failures),
        "candidate_failure_counts": dict(candidate_failures),
        "target_case_improved": target_case_improved,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    decision_lines = [
        "# AutoIter Candidate/Control 决策",
        "",
        f"- 状态：**{decision.status}**",
        f"- 目标 case 改善：`{target_case_improved}`",
        f"- 原因：{', '.join(decision.reasons) or 'none'}",
        f"- 回退信号：{', '.join(decision.regressions) or 'none'}",
        "",
        "本文件只代表当前两份报告的比较；完整晋级仍需要独立随机批次复核。",
        "",
    ]
    (output_dir / "decision.md").write_text("\n".join(decision_lines), encoding="utf-8")
    return payload


def legacy_candidate_root(agent_path: Path) -> Path:
    """Resolve the old ``--agent`` value to a standard submission package."""
    root = agent_path.resolve().parent if agent_path.name == "main.py" else agent_path.resolve()
    if not (root / "main.py").is_file() or not (root / "deck.csv").is_file():
        raise ValueError("legacy --agent must point to a standard submission package")
    return root


def build_replay_command(
    evaluator_root: Path,
    agent: Path,
    label: str,
    opponents: list[str],
    games: int,
    output: Path,
    cg_path: Path,
    save_traces: bool = False,
) -> list[str]:
    """Build the native evaluation CLI command for callers retaining this helper.

    ``evaluator_root``, ``label`` and ``cg_path`` remain in the signature for
    import compatibility. They are intentionally not used for external lookup;
    the runtime contract is now the candidate's own standard package.
    """
    del evaluator_root, label, cg_path
    command = [
        sys.executable,
        "-m",
        "evaluation",
        "run",
        "--candidate",
        str(agent.resolve().parent if agent.name == "main.py" else agent.resolve()),
        "--opponents",
        ",".join(opponents),
        "--games",
        str(games),
        "--output",
        str(output.resolve()),
    ]
    if save_traces:
        command.append("--save-traces")
    return command


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="分析已有 evaluator report")
    analyze.add_argument("--report-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--agent-label", default=None)

    compare = subparsers.add_parser("compare", help="比较 control 与 candidate")
    compare.add_argument("--control", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="通过原生 evaluation CLI 运行旧兼容入口")
    run.add_argument(
        "--evaluator-root",
        type=Path,
        help="已废弃；仅记录兼容调用，不再用于导入或路径查找",
    )
    run.add_argument("--agent", type=Path, required=True)
    run.add_argument("--cg-path", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--opponents", required=True)
    run.add_argument("--games", type=int, default=10)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--save-traces",
        dest="save_traces",
        action="store_true",
        help="保存完整逐局 trace；仅用于需要复盘的最新评测轮次",
    )
    run.add_argument(
        "--no-save-traces",
        dest="save_traces",
        action="store_false",
        help="只保存 evaluator summary，避免完整 trace 占用大量空间",
    )
    run.set_defaults(save_traces=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _command_parser().parse_args(argv)
    if args.command == "analyze":
        result = analyze_report(args.report_dir, agent_label=args.agent_label)
        manifest = {
            "command": "analyze",
            "report_dir": str(args.report_dir.resolve()),
            "agent_label": args.agent_label,
        }
        write_analysis(result, args.output_dir, manifest)
        return 0
    if args.command == "compare":
        compare_reports(args.control, args.candidate, args.output_dir)
        return 0

    try:
        candidate_root = legacy_candidate_root(args.agent)
        expected_cg_path = (candidate_root / "cg").resolve()
        if args.cg_path.resolve() != expected_cg_path:
            raise ValueError("legacy --cg-path must point to candidate/cg")
    except ValueError as exc:
        _command_parser().error(str(exc))

    if args.evaluator_root is not None:
        print(
            "warning: --evaluator-root is deprecated and ignored; "
            "the candidate package provides its own runtime",
            file=sys.stderr,
        )
    if args.save_traces:
        print(
            "warning: --save-traces maps to --keep-temp; persistent reports retain at most three traces",
            file=sys.stderr,
        )

    cli_args = [
        "run",
        "--candidate",
        str(candidate_root),
        "--opponents",
        args.opponents,
        "--games",
        str(args.games),
        "--output",
        str(args.output_dir),
    ]
    if args.save_traces:
        cli_args.append("--keep-temp")
    return evaluation_cli_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())

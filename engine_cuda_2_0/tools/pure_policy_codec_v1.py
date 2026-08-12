from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


CODEC_VERSION = "pure_policy_codec_v1"

GLOBAL_CAT_DIM = 8
GLOBAL_NUM_DIM = 16
ENTITY_CAT_DIM = 6
ENTITY_NUM_DIM = 10
OPTION_CAT_DIM = 12
OPTION_NUM_DIM = 4


OWNER_NONE = 0
OWNER_SELF = 1
OWNER_OPP = 2

ZONE_UNKNOWN = 0
ZONE_OWN_ACTIVE = 1
ZONE_OWN_BENCH = 2
ZONE_OWN_HAND = 3
ZONE_OWN_DISCARD = 4
ZONE_OWN_PRIZE = 5
ZONE_OPP_ACTIVE = 6
ZONE_OPP_BENCH = 7
ZONE_OPP_DISCARD = 8
ZONE_OPP_PRIZE = 9
ZONE_STADIUM = 10
ZONE_LOOKING = 11
ZONE_SELECT_DECK = 12
ZONE_OWN_ENERGY = 13
ZONE_OPP_ENERGY = 14
ZONE_OWN_TOOL = 15
ZONE_OPP_TOOL = 16
ZONE_OWN_EVOLUTION = 17
ZONE_OPP_EVOLUTION = 18

KIND_CARD = 1
KIND_POKEMON = 2
KIND_ENERGY = 3
KIND_TOOL = 4
KIND_EVOLUTION = 5
KIND_STADIUM = 6
KIND_LOOKING = 7

FAMILY_NAMES = (
    "other",
    "attack",
    "pass_end",
    "evolve",
    "play_card",
    "attach_energy",
    "discard_energy",
    "board_select",
    "search_select",
    "number_select",
    "multi_select",
)
FAMILY_TO_ID = {name: idx for idx, name in enumerate(FAMILY_NAMES)}


def _getv(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: Any) -> int:
    if card is None:
        return 0
    if isinstance(card, (int, float)):
        return max(0, _as_int(card, 0))
    return max(0, _as_int(_getv(card, "id", _getv(card, "cardId", 0)), 0))


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _player(current: dict[str, Any], player_index: int) -> dict[str, Any]:
    players = _list(current.get("players"))
    if 0 <= player_index < len(players) and isinstance(players[player_index], dict):
        return players[player_index]
    return {}


def _count_prizes(player: dict[str, Any]) -> int:
    prizes = player.get("prize")
    if isinstance(prizes, list):
        return len(prizes)
    return max(0, _as_int(player.get("prizeCount"), 0))


def _status_bits(player: dict[str, Any]) -> int:
    flags = ("asleep", "burned", "confused", "paralyzed", "poisoned")
    bits = 0
    for idx, flag in enumerate(flags):
        if bool(player.get(flag)):
            bits |= 1 << idx
    return bits


def _relative_owner(player_index: int, actor_index: int) -> int:
    if player_index == actor_index:
        return OWNER_SELF
    if player_index in (0, 1):
        return OWNER_OPP
    return OWNER_NONE


def _zone_for(owner: int, self_zone: int, opp_zone: int) -> int:
    return self_zone if owner == OWNER_SELF else opp_zone


def _positive_enum(value: Any, cap: int) -> int:
    raw = _as_int(value, -1)
    if raw < 0:
        return 0
    return min(raw + 1, cap)


def stable_percent(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 100


def split_for_episode(
    episode_id: str,
    opponent: str,
    *,
    valid_percent: int = 5,
    test_percent: int = 5,
    permanent_holdouts: Iterable[str] = (),
) -> str:
    if opponent in set(permanent_holdouts):
        return "test"
    bucket = stable_percent(episode_id)
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + valid_percent:
        return "valid"
    return "train"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _flatten(rows: list[list[Any]], width: int) -> list[Any]:
    out: list[Any] = []
    for row in rows:
        if len(row) != width:
            raise ValueError(f"row width {len(row)} != expected {width}")
        out.extend(row)
    return out


def _unflatten(values: Iterable[Any], width: int) -> list[list[Any]]:
    flat = list(values)
    if len(flat) % width:
        raise ValueError(f"flat length {len(flat)} is not divisible by {width}")
    return [flat[i : i + width] for i in range(0, len(flat), width)]


@dataclass
class EncodedObservation:
    global_cat: list[int]
    global_num: list[float]
    entity_cat: list[list[int]]
    entity_num: list[list[float]]
    entity_parent: list[int]
    option_cat: list[list[int]]
    option_num: list[list[float]]
    option_equiv: list[int]
    min_count: int
    max_count: int
    action_family: int
    actor_index: int
    turn: int
    own_prizes: int
    opp_prizes: int

    def validate(self) -> None:
        if len(self.global_cat) != GLOBAL_CAT_DIM:
            raise ValueError(f"global_cat dim={len(self.global_cat)}")
        if len(self.global_num) != GLOBAL_NUM_DIM:
            raise ValueError(f"global_num dim={len(self.global_num)}")
        if not self.option_cat:
            raise ValueError("encoded observation has no legal options")
        if not (
            len(self.entity_cat)
            == len(self.entity_num)
            == len(self.entity_parent)
        ):
            raise ValueError("entity arrays have different lengths")
        if not (
            len(self.option_cat)
            == len(self.option_num)
            == len(self.option_equiv)
        ):
            raise ValueError("option arrays have different lengths")
        if any(len(row) != ENTITY_CAT_DIM for row in self.entity_cat):
            raise ValueError("bad entity categorical width")
        if any(len(row) != ENTITY_NUM_DIM for row in self.entity_num):
            raise ValueError("bad entity numeric width")
        if any(len(row) != OPTION_CAT_DIM for row in self.option_cat):
            raise ValueError("bad option categorical width")
        if any(len(row) != OPTION_NUM_DIM for row in self.option_num):
            raise ValueError("bad option numeric width")
        if self.min_count < 0 or self.max_count < self.min_count:
            raise ValueError(f"invalid select bounds {self.min_count}/{self.max_count}")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "codec_version": CODEC_VERSION,
            "global_cat": self.global_cat,
            "global_num": self.global_num,
            "entity_count": len(self.entity_cat),
            "entity_cat": _flatten(self.entity_cat, ENTITY_CAT_DIM),
            "entity_num": _flatten(self.entity_num, ENTITY_NUM_DIM),
            "entity_parent": self.entity_parent,
            "option_count": len(self.option_cat),
            "option_cat": _flatten(self.option_cat, OPTION_CAT_DIM),
            "option_num": _flatten(self.option_num, OPTION_NUM_DIM),
            "option_equiv": self.option_equiv,
            "min_count": self.min_count,
            "max_count": self.max_count,
            "action_family": self.action_family,
            "actor_index": self.actor_index,
            "turn": self.turn,
            "own_prizes": self.own_prizes,
            "opp_prizes": self.opp_prizes,
        }

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> "EncodedObservation":
        encoded = cls(
            global_cat=[int(x) for x in row["global_cat"]],
            global_num=[float(x) for x in row["global_num"]],
            entity_cat=[[int(v) for v in values] for values in _unflatten(row["entity_cat"], ENTITY_CAT_DIM)],
            entity_num=[[float(v) for v in values] for values in _unflatten(row["entity_num"], ENTITY_NUM_DIM)],
            entity_parent=[int(x) for x in row["entity_parent"]],
            option_cat=[[int(v) for v in values] for values in _unflatten(row["option_cat"], OPTION_CAT_DIM)],
            option_num=[[float(v) for v in values] for values in _unflatten(row["option_num"], OPTION_NUM_DIM)],
            option_equiv=[int(x) for x in row["option_equiv"]],
            min_count=int(row["min_count"]),
            max_count=int(row["max_count"]),
            action_family=int(row.get("action_family", 0)),
            actor_index=int(row.get("actor_index", 0)),
            turn=int(row.get("turn", 0)),
            own_prizes=int(row.get("own_prizes", 0)),
            opp_prizes=int(row.get("opp_prizes", 0)),
        )
        encoded.validate()
        return encoded


class PolicyCodecV1:
    """One deterministic observation codec shared by collection, training and runtime."""

    def encode(self, obs: dict[str, Any], selected_action: list[int] | None = None) -> EncodedObservation:
        current = obs.get("current") or {}
        select = obs.get("select") or {}
        options = _list(select.get("option"))
        if not options:
            raise ValueError("observation has no legal options")

        actor_index = _as_int(current.get("yourIndex"), 0)
        opponent_index = 1 - actor_index
        own = _player(current, actor_index)
        opp = _player(current, opponent_index)
        turn = max(0, _as_int(current.get("turn"), 0))
        min_count = max(0, _as_int(select.get("minCount"), 0))
        max_count = max(min_count, _as_int(select.get("maxCount"), min_count))
        own_prizes = _count_prizes(own)
        opp_prizes = _count_prizes(opp)

        flags = 0
        for idx, name in enumerate(("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated", "turnEnd")):
            if bool(current.get(name)):
                flags |= 1 << idx

        global_cat = [
            _positive_enum(select.get("type"), 127),
            _positive_enum(select.get("context"), 255),
            _positive_enum(current.get("firstPlayer"), 2),
            _positive_enum(actor_index, 2),
            _positive_enum(current.get("phase"), 63),
            min(min_count, 15),
            min(max_count, 15),
            min(flags, 63),
        ]
        looking = _list(current.get("looking"))
        global_num = [
            min(turn / 20.0, 2.0),
            min(max(0, _as_int(current.get("turnActionCount"), 0)) / 50.0, 2.0),
            min(max(0, _as_int(current.get("effectActionCount"), 0)) / 50.0, 2.0),
            min(max(0, _as_int(current.get("turnAttackCount"), 0)) / 10.0, 2.0),
            min(max(0, _as_int(own.get("deckCount"), 0)) / 60.0, 1.0),
            min(max(0, _as_int(opp.get("deckCount"), 0)) / 60.0, 1.0),
            min(max(0, _as_int(own.get("handCount"), len(_list(own.get("hand"))))) / 20.0, 2.0),
            min(max(0, _as_int(opp.get("handCount"), len(_list(opp.get("hand"))))) / 20.0, 2.0),
            min(own_prizes / 6.0, 1.0),
            min(opp_prizes / 6.0, 1.0),
            min(len(options) / 128.0, 2.0),
            min(max(0, _as_int(select.get("remainDamageCounter"), 0)) / 300.0, 2.0),
            min(max(0, _as_int(select.get("remainEnergyCost"), 0)) / 10.0, 2.0),
            min(len(looking) / 60.0, 1.0),
            min(len(_list(own.get("bench"))) / 5.0, 1.0),
            min(len(_list(opp.get("bench"))) / 5.0, 1.0),
        ]

        entity_cat: list[list[int]] = []
        entity_num: list[list[float]] = []
        entity_parent: list[int] = []
        location_map: dict[tuple[int, int, int], int] = {}

        def add_entity(
            card: Any,
            owner: int,
            zone: int,
            slot: int,
            kind: int,
            *,
            parent: int = -1,
            player_status: int = 0,
            location: tuple[int, int, int] | None = None,
        ) -> int:
            cid = min(_card_id(card), 4096)
            hp = max(0.0, _as_float(_getv(card, "hp", 0), 0.0))
            max_hp = max(hp, _as_float(_getv(card, "maxHp", hp), hp), 0.0)
            damage = max(0.0, max_hp - hp)
            energies = _list(_getv(card, "energyCards", [])) or _list(_getv(card, "energies", []))
            tools = _list(_getv(card, "tools", []))
            pre_evolution = _list(_getv(card, "preEvolution", []))
            idx = len(entity_cat)
            entity_cat.append([
                cid,
                min(max(owner, 0), 3),
                min(max(zone, 0), 31),
                min(max(slot + 1, 0), 64),
                min(max(kind, 0), 7),
                min(max(player_status, 0), 63),
            ])
            entity_num.append([
                min(hp / 400.0, 2.0),
                min(max_hp / 400.0, 2.0),
                min(damage / 400.0, 2.0),
                min(len(energies) / 10.0, 2.0),
                min(len(tools) / 4.0, 2.0),
                min(len(pre_evolution) / 4.0, 2.0),
                1.0 if bool(_getv(card, "appearThisTurn", _getv(card, "appear", False))) else 0.0,
                1.0 if bool(_getv(card, "evolved", False)) else 0.0,
                1.0 if bool(_getv(card, "reverse", False)) else 0.0,
                min(max(slot, 0) / 64.0, 1.0),
            ])
            entity_parent.append(parent)
            if location is not None:
                location_map[location] = idx
            return idx

        def add_board_zone(player_index: int, zone_name: str, area: int, self_zone: int, opp_zone: int) -> None:
            player = _player(current, player_index)
            owner = _relative_owner(player_index, actor_index)
            zone = _zone_for(owner, self_zone, opp_zone)
            status = _status_bits(player) if zone_name == "active" else 0
            for slot, card in enumerate(_list(player.get(zone_name))):
                if card is None:
                    continue
                kind = KIND_POKEMON if zone_name in ("active", "bench") else KIND_CARD
                parent = add_entity(
                    card,
                    owner,
                    zone,
                    slot,
                    kind,
                    player_status=status,
                    location=(player_index, area, slot),
                )
                if zone_name not in ("active", "bench"):
                    continue
                energy_zone = ZONE_OWN_ENERGY if owner == OWNER_SELF else ZONE_OPP_ENERGY
                tool_zone = ZONE_OWN_TOOL if owner == OWNER_SELF else ZONE_OPP_TOOL
                evo_zone = ZONE_OWN_EVOLUTION if owner == OWNER_SELF else ZONE_OPP_EVOLUTION
                energies = _list(_getv(card, "energyCards", [])) or _list(_getv(card, "energies", []))
                for child_slot, child in enumerate(energies):
                    add_entity(child, owner, energy_zone, child_slot, KIND_ENERGY, parent=parent)
                for child_slot, child in enumerate(_list(_getv(card, "tools", []))):
                    add_entity(child, owner, tool_zone, child_slot, KIND_TOOL, parent=parent)
                for child_slot, child in enumerate(_list(_getv(card, "preEvolution", []))):
                    add_entity(child, owner, evo_zone, child_slot, KIND_EVOLUTION, parent=parent)

        for player_index in (actor_index, opponent_index):
            add_board_zone(player_index, "active", 4, ZONE_OWN_ACTIVE, ZONE_OPP_ACTIVE)
            add_board_zone(player_index, "bench", 5, ZONE_OWN_BENCH, ZONE_OPP_BENCH)
            if player_index == actor_index:
                add_board_zone(player_index, "hand", 2, ZONE_OWN_HAND, ZONE_UNKNOWN)
            add_board_zone(player_index, "discard", 3, ZONE_OWN_DISCARD, ZONE_OPP_DISCARD)
            for slot, card in enumerate(_list(_player(current, player_index).get("prize"))):
                if card is None:
                    continue
                owner = _relative_owner(player_index, actor_index)
                zone = _zone_for(owner, ZONE_OWN_PRIZE, ZONE_OPP_PRIZE)
                add_entity(card, owner, zone, slot, KIND_CARD, location=(player_index, 6, slot))

        for slot, card in enumerate(_list(current.get("stadium"))):
            if card is not None:
                add_entity(card, OWNER_NONE, ZONE_STADIUM, slot, KIND_STADIUM, location=(-1, 7, slot))
        for slot, card in enumerate(looking):
            if card is not None:
                add_entity(card, OWNER_SELF, ZONE_LOOKING, slot, KIND_LOOKING, location=(actor_index, 12, slot))
        for slot, card in enumerate(_list(select.get("deck"))):
            if card is not None:
                add_entity(card, OWNER_SELF, ZONE_SELECT_DECK, slot, KIND_CARD, location=(actor_index, 1, slot))

        def option_player_rel(raw_player: Any) -> int:
            pi = _as_int(raw_player, actor_index)
            return _relative_owner(pi, actor_index)

        option_cat: list[list[int]] = []
        option_num: list[list[float]] = []
        signatures: list[tuple[Any, ...]] = []
        context = _as_int(select.get("context"), -1)
        for option_index, option in enumerate(options):
            if not isinstance(option, dict):
                option = {}
            opt_type = _as_int(option.get("type"), -1)
            source_player = _as_int(option.get("playerIndex"), actor_index)
            source_area = _as_int(option.get("area"), -1)
            if opt_type == 7 and source_area < 0:
                source_area = 2
            source_slot = _as_int(option.get("index"), -1)
            source_entity = location_map.get((source_player, source_area, source_slot), -1)
            if source_area == 7:
                source_entity = location_map.get((-1, 7, source_slot), source_entity)
            explicit_card = _as_int(option.get("cardId"), 0)
            source_card = explicit_card
            if source_card <= 0 and 0 <= source_entity < len(entity_cat):
                source_card = entity_cat[source_entity][0]

            target_area = _as_int(option.get("inPlayArea"), -1)
            target_slot = _as_int(option.get("inPlayIndex"), -1)
            target_player = _as_int(
                option.get("inPlayPlayerIndex", option.get("targetPlayerIndex", option.get("playerIndex", actor_index))),
                actor_index,
            )
            target_entity = location_map.get((target_player, target_area, target_slot), -1)
            target_card = entity_cat[target_entity][0] if 0 <= target_entity < len(entity_cat) else 0
            attack_id = min(max(0, _as_int(option.get("attackId"), 0)), 4096)
            number = _as_int(option.get("number"), -1)
            row = [
                _positive_enum(opt_type, 63),
                _positive_enum(source_area, 31),
                _positive_enum(target_area, 31),
                option_player_rel(source_player),
                min(max(source_card, 0), 4096),
                min(max(target_card, 0), 4096),
                attack_id,
                _positive_enum(number, 127),
                source_entity + 1,
                target_entity + 1,
                _positive_enum(source_slot, 127),
                _positive_enum(target_slot, 127),
            ]
            option_cat.append(row)
            option_num.append([
                min(max(0, _as_int(select.get("remainDamageCounter"), 0)) / 300.0, 2.0),
                min(max(0, _as_int(select.get("remainEnergyCost"), 0)) / 10.0, 2.0),
                min(option_index / 128.0, 2.0),
                min(len(options) / 128.0, 2.0),
            ])

            source_zone = entity_cat[source_entity][2] if 0 <= source_entity < len(entity_cat) else 0
            source_pos = entity_cat[source_entity][3] if 0 <= source_entity < len(entity_cat) else 0
            if source_zone in {ZONE_OWN_HAND, ZONE_OWN_DISCARD, ZONE_OPP_DISCARD, ZONE_LOOKING, ZONE_SELECT_DECK}:
                source_pos = 0
            target_zone = entity_cat[target_entity][2] if 0 <= target_entity < len(entity_cat) else 0
            target_pos = entity_cat[target_entity][3] if 0 <= target_entity < len(entity_cat) else 0
            signatures.append(
                (
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                    source_zone, source_pos, target_zone, target_pos,
                )
            )

        signature_to_group: dict[tuple[Any, ...], int] = {}
        option_equiv: list[int] = []
        for signature in signatures:
            if signature not in signature_to_group:
                signature_to_group[signature] = len(signature_to_group)
            option_equiv.append(signature_to_group[signature])

        summaries = [self._summary_from_option(row, context) for row in option_cat]
        chosen = [idx for idx in (selected_action or []) if isinstance(idx, int) and 0 <= idx < len(summaries)]
        chosen_summaries = [summaries[idx] for idx in chosen] if chosen else summaries[:1]
        family_name = self._action_family(context, chosen_summaries, min_count, max_count)
        encoded = EncodedObservation(
            global_cat=global_cat,
            global_num=global_num,
            entity_cat=entity_cat,
            entity_num=entity_num,
            entity_parent=entity_parent,
            option_cat=option_cat,
            option_num=option_num,
            option_equiv=option_equiv,
            min_count=min_count,
            max_count=max_count,
            action_family=FAMILY_TO_ID[family_name],
            actor_index=actor_index,
            turn=turn,
            own_prizes=own_prizes,
            opp_prizes=opp_prizes,
        )
        encoded.validate()
        return encoded

    @staticmethod
    def _summary_from_option(row: list[int], context: int) -> dict[str, int]:
        return {
            "type": row[0] - 1,
            "area": row[1] - 1,
            "in_play_area": row[2] - 1,
            "attack_id": row[6],
            "context": context,
        }

    @staticmethod
    def _action_family(context: int, summaries: list[dict[str, int]], min_count: int, max_count: int) -> str:
        if max_count > 1:
            return "multi_select"
        option_types = {row["type"] for row in summaries}
        areas = {row["area"] for row in summaries}
        in_play_areas = {row["in_play_area"] for row in summaries}
        if any(row["attack_id"] > 0 for row in summaries) or 13 in option_types or context == 35:
            return "attack"
        if 12 in option_types or 14 in option_types:
            return "pass_end"
        if 9 in option_types or context == 37:
            return "evolve"
        if 7 in option_types:
            return "play_card"
        if context == 30:
            return "discard_energy"
        if 5 in option_types or 6 in option_types:
            return "attach_energy"
        if context in {1, 2, 4, 13, 21, 22} or 4 in areas or 5 in areas or 4 in in_play_areas or 5 in in_play_areas:
            return "board_select"
        if 1 in areas or context in {5, 7}:
            return "search_select"
        if 0 in option_types or context == 8:
            return "number_select"
        return "other"


def normalize_model_action(action: Any, option_count: int, min_count: int, max_count: int) -> list[int]:
    if option_count <= 0:
        return []
    out: list[int] = []
    if isinstance(action, list):
        for value in action:
            if isinstance(value, int) and 0 <= value < option_count and value not in out:
                out.append(value)
                if len(out) >= max_count:
                    break
    if len(out) < min_count:
        for idx in range(option_count):
            if idx not in out:
                out.append(idx)
                if len(out) >= min_count:
                    break
    return out[:max_count]


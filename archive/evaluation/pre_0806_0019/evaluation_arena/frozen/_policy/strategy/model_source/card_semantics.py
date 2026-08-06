"""Deterministic structured semantics from the official card CSV."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Generic, TypeVar

UNK_CARD_ID = -1
PADDING_CARD_ID = -2
T = TypeVar("T")


class FieldState(str, Enum):
    OBSERVED = "observed"
    UNKNOWN = "unknown"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    PADDING = "padding"


@dataclass(frozen=True, slots=True)
class SemanticField(Generic[T]):
    value: T | None
    state: FieldState


@dataclass(frozen=True, slots=True)
class EffectPrimitive:
    kind: str
    target: str = "unspecified"
    source_zone: str = "unspecified"
    destination_zone: str = "unspecified"
    count: int | None = None
    condition: str = "none"


@dataclass(frozen=True, slots=True)
class MoveSemantics:
    name: str
    energy_cost: tuple[str, ...]
    damage: SemanticField[int]
    effects: tuple[EffectPrimitive, ...]


@dataclass(frozen=True, slots=True)
class CardSemantics:
    card_id: int
    identity_state: FieldState
    name: str
    card_kind: str
    stage: str
    rule: str
    category: str
    previous_stage: SemanticField[str]
    hp: SemanticField[int]
    pokemon_type: SemanticField[str]
    weakness: SemanticField[str]
    resistance: SemanticField[str]
    retreat: SemanticField[int]
    moves: tuple[MoveSemantics, ...]
    capabilities: frozenset[str]


@dataclass(frozen=True, slots=True)
class CoverageAudit:
    identity_total: int
    identity_known: int
    effect_instances_total: int
    effect_instances_known: int

    @property
    def identity_coverage(self) -> float:
        return self.identity_known / self.identity_total if self.identity_total else 1.0

    @property
    def effect_coverage(self) -> float:
        return self.effect_instances_known / self.effect_instances_total if self.effect_instances_total else 1.0


_NA = {"n/a", "na", "not applicable"}
_ENERGY = re.compile(r"\{[A-Z]}|●")
_EFFECT_RULES = (
    ("draw", ("draw ",)), ("search", ("search your deck", "search their deck")),
    ("reveal", ("reveal",)), ("discard", ("discard",)),
    ("recover", ("put", "from your discard")), ("attach", ("attach",)),
    ("evolve", ("evolve",)), ("switch", ("switch", "bench")),
    ("heal", ("heal", "remove damage")),
    ("status", ("poisoned", "burned", "paralyzed", "asleep", "confused")),
    ("prize", ("prize card",)), ("shuffle", ("shuffle",)),
    ("energy_access", ("energy card", "basic energy")),
    ("protection", ("prevent all damage", "prevent damage")),
    ("disruption", ("opponent’s hand", "opponent's hand", "opponent’s deck")),
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _text(value: object) -> str:
    return str(value or "").strip()


def _field(value: object, parser=lambda x: x) -> SemanticField:
    raw = _text(value)
    if not raw:
        return SemanticField(None, FieldState.MISSING)
    if raw.casefold() in _NA:
        return SemanticField(None, FieldState.NOT_APPLICABLE)
    try:
        return SemanticField(parser(raw), FieldState.OBSERVED)
    except (TypeError, ValueError):
        return SemanticField(None, FieldState.UNKNOWN)


def _stage(value: str) -> str:
    normalized = value.casefold()
    if "basic pokémon" in normalized:
        return "basic"
    if "stage 1" in normalized:
        return "stage_1"
    if "stage 2" in normalized:
        return "stage_2"
    if "energy" in normalized:
        return "energy"
    if normalized in {"item", "supporter", "stadium", "pokémon tool"}:
        return normalized.replace("é", "e").replace(" ", "_")
    return "unknown"


def _kind(stage_type: str) -> str:
    value = stage_type.casefold()
    if "pokémon" in value:
        return "pokemon"
    if "energy" in value:
        return "energy"
    return "trainer"


def _primitives(text: str, move_name: str, damage: SemanticField[int]) -> tuple[EffectPrimitive, ...]:
    lower = text.casefold()
    result: list[EffectPrimitive] = []
    if damage.state is FieldState.OBSERVED and damage.value and damage.value > 0:
        result.append(EffectPrimitive("damage", target="opponent_active"))
    for kind, needles in _EFFECT_RULES:
        if any(needle in lower for needle in needles):
            result.append(EffectPrimitive(kind))
    if text and not result:
        result.append(EffectPrimitive("rule", condition="text_residual"))
    if move_name.casefold().startswith("[ability]"):
        result.append(EffectPrimitive("rule", condition="ability"))
    return tuple(dict.fromkeys(result))


def _capabilities(kind: str, stage: str, moves: Sequence[MoveSemantics]) -> frozenset[str]:
    caps: set[str] = set()
    if stage in {"basic", "stage_1", "stage_2"}:
        caps.add("setup" if stage == "basic" else "evolution")
    mapping = {
        "damage": {"attacker", "prize_progress"}, "search": {"search", "setup"},
        "draw": {"draw"}, "recover": {"recovery"}, "switch": {"switching"},
        "retreat": {"switching"}, "energy_access": {"energy_access"},
        "attach": {"energy_access"}, "disruption": {"disruption"},
        "protection": {"survival"}, "heal": {"survival"}, "status": {"disruption"},
        "prize": {"prize_progress"},
    }
    for move in moves:
        for effect in move.effects:
            caps.update(mapping.get(effect.kind, ()))
    return frozenset(caps)


class CardSemanticRegistry:
    def __init__(self, cards: Mapping[int, CardSemantics]) -> None:
        self._cards = MappingProxyType(dict(cards))
        self.unknown = CardSemantics(UNK_CARD_ID, FieldState.UNKNOWN, "UNK_CARD", "unknown", "unknown", "unknown", "unknown", SemanticField(None, FieldState.UNKNOWN), SemanticField(None, FieldState.UNKNOWN), SemanticField(None, FieldState.UNKNOWN), SemanticField(None, FieldState.UNKNOWN), SemanticField(None, FieldState.UNKNOWN), SemanticField(None, FieldState.UNKNOWN), (), frozenset())
        self.padding = CardSemantics(PADDING_CARD_ID, FieldState.PADDING, "PADDING", "padding", "padding", "padding", "padding", SemanticField(None, FieldState.PADDING), SemanticField(None, FieldState.PADDING), SemanticField(None, FieldState.PADDING), SemanticField(None, FieldState.PADDING), SemanticField(None, FieldState.PADDING), SemanticField(None, FieldState.PADDING), (), frozenset())
        serial = [self._as_dict(self._cards[key]) for key in sorted(self._cards)]
        self.sha256 = hashlib.sha256(_canonical(serial)).hexdigest()

    @classmethod
    def from_official_csv(cls, path: Path | str) -> CardSemanticRegistry:
        import csv
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            return cls.from_rows(csv.DictReader(handle))

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, str]]) -> CardSemanticRegistry:
        grouped: dict[int, list[Mapping[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["Card ID"])].append(dict(row))
        cards: dict[int, CardSemantics] = {}
        structural = ("Card Name", "Stage (Pokémon)/Type (Energy and Trainer)", "Rule", "Category", "Previous stage", "HP", "Type", "Weakness", "Resistance (Type)", "Retreat")
        for card_id, members in grouped.items():
            ordered = sorted(members, key=lambda row: (len(_ENERGY.findall(_text(row.get("Cost")))), _text(row.get("Move Name")), _text(row.get("Effect Explanation"))))
            for field in structural:
                if len({_text(row.get(field)) for row in ordered}) != 1:
                    raise ValueError(f"card {card_id} has conflicting structural fields: {field}")
            base = ordered[0]
            stage_type = _text(base["Stage (Pokémon)/Type (Energy and Trainer)"])
            moves = []
            for row in ordered:
                name = _text(row.get("Move Name"))
                explanation = _text(row.get("Effect Explanation"))
                damage = _field(row.get("Damage"), lambda value: int(re.match(r"\d+", value).group()) if re.match(r"\d+", value) else (_ for _ in ()).throw(ValueError()))
                if not name and not explanation:
                    continue
                moves.append(MoveSemantics(name, tuple(_ENERGY.findall(_text(row.get("Cost")))), damage, _primitives(explanation, name, damage)))
            kind, stage = _kind(stage_type), _stage(stage_type)
            card = CardSemantics(card_id, FieldState.OBSERVED, _text(base["Card Name"]), kind, stage, _text(base["Rule"]), _text(base["Category"]), _field(base["Previous stage"]), _field(base["HP"], int), _field(base["Type"]), _field(base["Weakness"]), _field(base["Resistance (Type)"]), _field(base["Retreat"], int), tuple(moves), frozenset())
            cards[card_id] = CardSemantics(**{**card.__dict__, "capabilities": _capabilities(kind, stage, card.moves)}) if hasattr(card, "__dict__") else CardSemantics(card.card_id, card.identity_state, card.name, card.card_kind, card.stage, card.rule, card.category, card.previous_stage, card.hp, card.pokemon_type, card.weakness, card.resistance, card.retreat, card.moves, _capabilities(kind, stage, card.moves))
        return cls(cards)

    def lookup(self, card_id: int) -> CardSemantics:
        return self._cards.get(card_id, self.unknown)

    def require(self, card_id: int) -> CardSemantics:
        card = self.lookup(card_id)
        if card.identity_state is not FieldState.OBSERVED:
            raise KeyError(card_id)
        return card

    def audit_coverage(self, registered: Sequence[int], option_ids: Sequence[int]) -> CoverageAudit:
        identities = tuple(registered) + tuple(option_ids)
        known = [self.lookup(card_id) for card_id in identities]
        effects = [effect for card in known for move in card.moves for effect in move.effects]
        return CoverageAudit(len(known), sum(card.identity_state is FieldState.OBSERVED for card in known), len(effects), sum(effect.kind != "UNKNOWN_EFFECT" for effect in effects))

    @staticmethod
    def _as_dict(card: CardSemantics) -> dict[str, object]:
        def field(value: SemanticField) -> dict[str, object]:
            return {"value": value.value, "state": value.state.value}
        return {"card_id": card.card_id, "name": card.name, "card_kind": card.card_kind, "stage": card.stage, "rule": card.rule, "category": card.category, "previous_stage": field(card.previous_stage), "hp": field(card.hp), "pokemon_type": field(card.pokemon_type), "weakness": field(card.weakness), "resistance": field(card.resistance), "retreat": field(card.retreat), "moves": [{"name": move.name, "energy_cost": list(move.energy_cost), "damage": field(move.damage), "effects": [{"kind": item.kind, "target": item.target, "source_zone": item.source_zone, "destination_zone": item.destination_zone, "count": item.count, "condition": item.condition} for item in move.effects]} for move in card.moves], "capabilities": sorted(card.capabilities)}


__all__ = ["CardSemanticRegistry", "CardSemantics", "CoverageAudit", "EffectPrimitive", "FieldState", "MoveSemantics", "PADDING_CARD_ID", "SemanticField", "UNK_CARD_ID"]

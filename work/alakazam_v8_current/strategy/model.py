from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Mapping


class Area(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7


class PlanKind(str, Enum):
    VICTORY = "victory"
    ATTACK = "attack"
    BUILD_SURVIVE = "build_survive"


class RouteCertainty(str, Enum):
    CONFIRMED = "confirmed"
    POSSIBLE = "possible"
    BLOCKED = "blocked"


class ActionKind(str, Enum):
    PLAY = "play"
    ATTACH = "attach"
    EVOLVE = "evolve"
    ABILITY = "ability"
    RETREAT = "retreat"
    ATTACK = "attack"
    END = "end"
    SELECT = "select"
    YES = "yes"
    NO = "no"
    COUNT = "count"
    UNKNOWN = "unknown"


class DecisionPhase(IntEnum):
    HARD_RULES = 0
    VICTORY = 1
    CURRENT_ATTACKER = 2
    MUST_PREPARE = 3
    RESOURCE_ALLOCATION = 4
    OPTIONAL_PREPARE = 5
    ATTACK_COMMIT = 6
    END = 7


@dataclass(frozen=True)
class PokemonRef:
    card_id: int
    serial: int | None
    owner: int
    area: Area
    index: int
    hp: int
    max_hp: int
    energy_types: tuple[int, ...] = ()
    attached_energy_ids: tuple[int, ...] = ()
    appear_this_turn: bool = False
    pre_evolution_ids: tuple[int, ...] = ()
    pre_evolution_serials: tuple[int, ...] = ()
    can_evolve: bool = False

    @property
    def key(self) -> tuple[int, int, int | None]:
        return self.owner, int(self.area), self.serial

    def has_energy_type(self, energy_type: int) -> bool:
        return energy_type in self.energy_types


@dataclass(frozen=True)
class PlayerFacts:
    index: int
    active: PokemonRef | None
    bench: tuple[PokemonRef, ...]
    hand_ids: tuple[int, ...]
    hand_count: int
    deck_count: int
    discard_ids: tuple[int, ...]
    prize_count: int
    bench_max: int

    @property
    def field(self) -> tuple[PokemonRef, ...]:
        return ((self.active,) if self.active else ()) + self.bench


@dataclass(frozen=True)
class TurnBudget:
    supporter_used: bool
    stadium_used: bool
    energy_used: bool
    retreat_used: bool
    attack_submitted: bool


@dataclass(frozen=True)
class ResourceLedger:
    visible_by_zone: Mapping[str, Mapping[int, int]]
    known_in_deck: Mapping[int, int]
    known_in_prize: Mapping[int, int]
    unknown_deck_or_prize: Mapping[int, int]


@dataclass(frozen=True)
class TurnFacts:
    turn: int
    own_turn: int
    your_index: int
    opponent_index: int
    first_player: int
    yours: PlayerFacts
    opponent: PlayerFacts
    budget: TurnBudget
    resources: ResourceLedger
    item_lock: bool
    previous_opponent_turn_had_ko: bool
    stadium_ids: tuple[int, ...]
    logs: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class AttackRoute:
    attacker: PokemonRef | None
    target: PokemonRef | None
    expected_damage: int
    prize_value: int
    certainty: RouteCertainty
    needs_evolution: bool = False
    needs_energy: bool = False
    needs_switch: bool = False


@dataclass(frozen=True)
class HandoffRoute:
    attacker: PokemonRef | None
    certainty: RouteCertainty
    needs_evolution: bool = False
    needs_energy: bool = False
    needs_search: bool = False
    via_dudunsparce: bool = False


@dataclass(frozen=True)
class PlanGoal:
    rule_id: str
    purpose: str
    required_before_attack: bool = True
    satisfied: bool = False


@dataclass(frozen=True)
class TurnPlan:
    plan_id: str
    revision: int
    kind: PlanKind
    primary_attack: AttackRoute
    handoff: HandoffRoute
    must_goals: tuple[PlanGoal, ...] = ()
    supporter_purpose: str | None = None
    energy_purpose: str | None = None
    deck_budget: int = 0
    attack_blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticOption:
    index: int
    action_kind: ActionKind
    raw_type: int
    card_id: int | None = None
    source: PokemonRef | None = None
    target: PokemonRef | None = None
    energy_id: int | None = None
    attack_id: int | None = None
    effect_id: int | None = None
    owner: int | None = None
    area: int | None = None
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class ActionIntent:
    rule_id: str
    phase: DecisionPhase
    action_kind: ActionKind
    purpose: str
    card_id: int | None = None
    target_key: tuple[int, int, int | None] | None = None
    attack_id: int | None = None
    required_before_attack: bool = False


@dataclass(frozen=True)
class Decision:
    option_indexes: tuple[int, ...]
    rule_id: str
    plan_kind: PlanKind | None
    purpose: str

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class AttackPreparation(str, Enum):
    COMPLETE_BEFORE_ATTACK = "complete_before_attack"
    ATTACK_WHEN_LETHAL = "attack_when_lethal"


class DudunsparcePolicy(str, Enum):
    HANDOFF_FIRST = "handoff_first"
    DRAW_WHEN_SAFE = "draw_when_safe"


class FezandipitiPolicy(str, Enum):
    DRAW_AFTER_KO = "draw_after_ko"
    CONSERVATIVE = "conservative"


class HammerPolicy(str, Enum):
    ALWAYS_LEGAL_TARGET = "always_legal_target"
    PROTECTIVE_ONLY = "protective_only"


class RecoveryPolicy(str, Enum):
    COMPLETE_LINES = "complete_lines"
    STAGE_WEIGHTED = "stage_weighted"


class StadiumPolicy(str, Enum):
    IMMEDIATE_VALUE = "immediate_value"
    BOARD_ONLY = "board_only"


class DeckSafetyPolicy(str, Enum):
    THREE_BANDS = "three_bands"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class StrategyProfile:
    attack_preparation: AttackPreparation = AttackPreparation.COMPLETE_BEFORE_ATTACK
    dudunsparce: DudunsparcePolicy = DudunsparcePolicy.HANDOFF_FIRST
    fezandipiti: FezandipitiPolicy = FezandipitiPolicy.DRAW_AFTER_KO
    hammer: HammerPolicy = HammerPolicy.ALWAYS_LEGAL_TARGET
    recovery: RecoveryPolicy = RecoveryPolicy.COMPLETE_LINES
    stadium: StadiumPolicy = StadiumPolicy.IMMEDIATE_VALUE
    deck_safety: DeckSafetyPolicy = DeckSafetyPolicy.THREE_BANDS


BASELINE_PROFILE = StrategyProfile()


def with_variant(profile: StrategyProfile, **changes: Any) -> StrategyProfile:
    """Create a typed profile variant without executing strategy code from data."""
    unknown = set(changes) - set(profile.__dataclass_fields__)
    if unknown:
        raise TypeError(f"unknown strategy profile fields: {sorted(unknown)}")
    for name, value in changes.items():
        expected = type(getattr(profile, name))
        if not isinstance(value, expected):
            raise TypeError(f"{name} must be {expected.__name__}, got {type(value).__name__}")
    return replace(profile, **changes)

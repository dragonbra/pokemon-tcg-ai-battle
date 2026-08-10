"""Fail-closed Phase 3 classifier for one official CPU SearchStep branch.

This module deliberately does not infer safety from selectType/selectContext.
Callers must supply evidence collected from the official CPU engine and the
production feature path. It is research scaffolding, not an inference API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Eligibility(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    REJECT_NOT_ACTUAL_CHOICE = "REJECT_NOT_ACTUAL_CHOICE"
    REJECT_PERSPECTIVE_FLIP = "REJECT_PERSPECTIVE_FLIP"
    REJECT_TURN_CHANGE = "REJECT_TURN_CHANGE"
    REJECT_TERMINAL = "REJECT_TERMINAL"
    REJECT_RNG = "REJECT_RNG"
    REJECT_HIDDEN_SENSITIVE = "REJECT_HIDDEN_SENSITIVE"
    REJECT_FEATURE_INCOMPATIBLE = "REJECT_FEATURE_INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CandidateEvidence:
    focal_player: int | None
    root_select_player: int | None
    legal_selection_count: int | None
    distinct_value_visible_afterstates: int | None
    branch_terminal: bool | None
    branch_select_player: int | None
    root_turn: int | None
    branch_turn: int | None
    root_active_player: int | None
    branch_active_player: int | None
    root_main_phase: bool | None
    branch_main_phase: bool | None
    rng_unchanged: bool | None
    hidden_invariant: bool | None
    feature_compatible: bool | None


def is_v0_value_search_candidate(evidence: CandidateEvidence) -> Eligibility:
    """Classify a candidate using only explicit, already-audited evidence."""

    if evidence.focal_player not in (0, 1):
        return Eligibility.UNKNOWN
    if evidence.root_select_player is None:
        return Eligibility.UNKNOWN
    if evidence.root_select_player != evidence.focal_player:
        return Eligibility.REJECT_PERSPECTIVE_FLIP

    if evidence.legal_selection_count is None:
        return Eligibility.UNKNOWN
    if evidence.legal_selection_count < 2:
        return Eligibility.REJECT_NOT_ACTUAL_CHOICE
    if evidence.distinct_value_visible_afterstates is None:
        return Eligibility.UNKNOWN
    if evidence.distinct_value_visible_afterstates < 2:
        return Eligibility.REJECT_NOT_ACTUAL_CHOICE

    if evidence.branch_terminal is None:
        return Eligibility.UNKNOWN
    if evidence.branch_terminal:
        return Eligibility.REJECT_TERMINAL
    if evidence.branch_select_player is None:
        return Eligibility.UNKNOWN
    if evidence.branch_select_player != evidence.focal_player:
        return Eligibility.REJECT_PERSPECTIVE_FLIP

    turn_fields = (
        evidence.root_turn,
        evidence.branch_turn,
        evidence.root_active_player,
        evidence.branch_active_player,
        evidence.root_main_phase,
        evidence.branch_main_phase,
    )
    if any(value is None for value in turn_fields):
        return Eligibility.UNKNOWN
    if (
        evidence.root_turn != evidence.branch_turn
        or evidence.root_active_player != evidence.branch_active_player
        or evidence.root_active_player != evidence.focal_player
        or evidence.branch_active_player != evidence.focal_player
        or not evidence.root_main_phase
        or not evidence.branch_main_phase
    ):
        return Eligibility.REJECT_TURN_CHANGE

    if evidence.rng_unchanged is None:
        return Eligibility.UNKNOWN
    if not evidence.rng_unchanged:
        return Eligibility.REJECT_RNG
    if evidence.hidden_invariant is None:
        return Eligibility.UNKNOWN
    if not evidence.hidden_invariant:
        return Eligibility.REJECT_HIDDEN_SENSITIVE
    if evidence.feature_compatible is None:
        return Eligibility.UNKNOWN
    if not evidence.feature_compatible:
        return Eligibility.REJECT_FEATURE_INCOMPATIBLE
    return Eligibility.ELIGIBLE


__all__ = ["CandidateEvidence", "Eligibility", "is_v0_value_search_candidate"]

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .official_ir import validate_official_ir


DOMAIN_SIZES = {
    "effects": 245,
    "targets": 102,
    "conditions": 24,
    "triggers": 21,
    "select_types": 12,
    "select_contexts": 50,
    "select_options": 17,
}

# Entity inventories are derived from the frozen IR and intentionally do not
# claim semantic parity.  Their explicit status makes missing per-card work
# visible instead of allowing an enum-only matrix to look complete.
ENTITY_DOMAINS = ("cards", "skills", "attacks")

# These have host/device state primitives, not complete target/condition dispatch.
PRIMITIVE_EFFECTS = {
    0: "NoEffect",
    2: "ForEach",
    3: "Ko",
    4: "ToHand",
    6: "ToHandReverse",
    7: "ToHandWithAttach",
    8: "ToTrash",
    9: "ToDeck",
    10: "ToDeckWithAttach",
    11: "ToDeckReverse",
    12: "LookToDeckReverse",
    13: "ToDeckAndShuffle",
    14: "ToDeckReverseAndShuffle",
    15: "ToDeckBottom",
    16: "ToDeckBottomReverse",
    19: "ToBench",
    20: "ToPrize",
    21: "ToLooking",
    22: "ToPlayingFirst",
    30: "DamageCounter",
    41: "RemoveDamageCounter",
    42: "RemoveDamageCounterAll",
    43: "Heal",
    44: "HealAll",
    46: "ResetHp",
    71: "Coin",
    72: "CoinUntilTail",
    96: "Burn",
    97: "Poison",
    98: "Poison8",
    99: "Poison16",
    100: "Sleep",
    101: "Confuse",
    102: "Paralyze",
    103: "RecoverSpecialCondition",
    105: "Draw",
    106: "DrawTargetCount",
    107: "DrawPrizeCount",
    108: "DrawUntil",
    110: "DrawMirror",
    111: "DeckToTrash",
    113: "DeckBottomToTrash",
    114: "DeckToPrize",
    115: "Shuffle",
    116: "EffectWin",
    117: "FailRetreat",
    118: "TurnEnd",
}
_EFFECT_NAMES = (
    "AttackDamageChange AttackDamageChangeTargetCount EffectDamageChangeTargetCount "
    "AttackDamageChangeEnergyCount EffectDamageChangeEnergyCount "
    "AttackDamageChangeTypeEnergyCount EffectDamageChangeTypeEnergyCount "
    "AttackDamageChangeEnergyCountCoin AttackDamageChangeTypeEnergyCountCoin "
    "AttackDamageChangeCoin AttackDamageChangeCoinUntilTail "
    "AttackDamageChangeTargetCountCoin AttackDamageChangeTargetCountEnemyCoin "
    "AttackDamageChangeTakenPrize EffectDamageChangeTakenPrize "
    "AttackDamageChangeDamageCounter EffectDamageChangeDamageCounter "
    "AttackDamageChangePutDamageCounter AttackDamageChangeRetreatCost "
    "AttackDamageChangeTypeCount AttackDamageChangeSpecialConditionCount "
    "AttackDamageChangeTakeAttackDamagePreTurn AttackDamageChangePreTurnTakePrizeCount"
).split()
PRIMITIVE_EFFECTS.update({73 + index: name for index, name in enumerate(_EFFECT_NAMES)})
PRIMITIVE_EFFECTS[109] = "DrawUntilPsychic"
PRIMITIVE_EFFECTS.update({
    1: "SelectCard",
    5: "PrizeToHand",
    17: "ToDeckBottomClose",
    18: "ToActiveAndTrashActive",
    23: "Switch",
    24: "SwitchDeck",
    25: "NotMove",
    26: "LookDeck",
    27: "LookDeckReverse",
    28: "LookDeckBottom",
    29: "LookAndReturn",
    31: "DamageCounterRemoved",
    32: "DamageCounterDamaged",
    33: "DamageCounterAny",
    34: "DamageCounterDouble",
    35: "DamageCounterHp",
    36: "DamageCounterSwitchAny",
    37: "DamageCounterTypeEnergyCountMe",
    38: "AttackDamage",
    39: "AttackDamageMulti",
    40: "AttackDamageCoin",
    45: "HealSand",
    47: "Drain",
    48: "Devolve",
    49: "DevolveAny",
    50: "TransformDeck",
    51: "TransformTrash",
    52: "ExchangeSelected",
    53: "KoPrizeChangeAlways",
    54: "KoPrizeChange",
    55: "KoPrizeDecreaseOnce",
    56: "SelectEvolvesFrom",
    57: "EvolvesToEach",
    58: "SelectEvolvesTo",
    59: "EvolvesFromEach",
    60: "SelectAttachFrom",
    61: "AttachToEach",
    62: "SelectAttachTo",
    63: "AttachEnergyMe",
    64: "AttachSelectedCard",
    65: "SwitchSelectedCard",
    66: "AttachFromEach",
    67: "SelectSwitchEnergy",
    68: "SelectSwitchEnergyCard",
    69: "EnergySwitchEach",
    70: "DelayEffect",
    104: "RecoverSpecialConditionSingle",
    112: "DeckToTrashCoinUntilTail",
})
PRIMITIVE_EFFECTS.update({
    effect_id: f"OfficialEffectType{effect_id}" for effect_id in range(119, 245)
})
PRIMITIVE_TARGETS = set(range(DOMAIN_SIZES["targets"]))
PRIMITIVE_CONDITIONS = set(range(DOMAIN_SIZES["conditions"]))
PAIRED_EFFECTS = {172, 174, 186, 207, 222, 234, 238, 244}
PAIRED_TARGETS = {79, 85, 92}
PAIRED_CONDITIONS = {9}


@dataclass(frozen=True)
class CoverageSummary:
    cards: int
    skills: int
    attacks: int
    branch_records: int
    effect_occurrences: int
    target_records: int
    target_type_occurrences: int
    target_condition_occurrences: int
    trigger_occurrences: int
    used_effect_types: int
    used_target_types: int
    used_condition_types: int
    used_trigger_types: int
    primitive_effect_types: int
    primitive_effect_occurrences: int
    primitive_target_types: int
    primitive_target_occurrences: int
    primitive_condition_types: int
    primitive_condition_occurrences: int
    paired_effect_types: int
    paired_effect_occurrences: int
    paired_target_types: int
    paired_target_occurrences: int
    paired_condition_types: int
    paired_condition_occurrences: int
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effects(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for skill in payload["skills"]:
        yield from skill["effects"]
    for attack in payload["attacks"]:
        yield from attack["pre_effects"]
        yield from attack["post_effects"]


def _targets(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for effect in _effects(payload):
        yield effect["target"]
    for skill in payload["skills"]:
        for trigger in skill["triggers"]:
            yield trigger["subject"]


def _entries(
    size: int,
    counts: Counter[int],
    status: str = "not_started",
) -> list[dict[str, Any]]:
    return [
        {"id": value, "occurrences": counts[value], "status": status}
        for value in range(size)
    ]


def _entity_entries(
    rows: Iterable[dict[str, Any]],
    *,
    row_id: str,
    metadata: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "id": int(row[row_id]),
            "occurrences": 1,
            "status": "inventory_only",
        }
        entry.update(metadata(row))
        entries.append(entry)
    return entries


def _branch_obligations(effect: dict[str, Any]) -> list[str]:
    obligations: list[str] = []
    if int(effect["select_type"]) != 0:
        obligations.extend(("selection_min", "selection_max"))
        if int(effect["flags"]) & ((1 << 11) | (1 << 12)):
            obligations.append("selection_zero")
    if int(effect["condition_type"]) != 0 or effect["target"]["conditions"]:
        obligations.extend(("condition_false", "condition_true"))
    if int(effect["flags"]) & (1 << 2) or int(effect["type"]) in {
        40,
        71,
        72,
        80,
        81,
        82,
        83,
        84,
        85,
        112,
    }:
        obligations.extend(("rng_false", "rng_true"))
    return obligations


def _branch_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def append_effect(
        *,
        branch_id: str,
        owner_card_id: int,
        owner_kind: str,
        owner_id: int,
        effect: dict[str, Any],
    ) -> None:
        entries.append(
            {
                "branch_id": branch_id,
                "kind": "effect",
                "owner_card_id": owner_card_id,
                "owner_kind": owner_kind,
                "owner_id": owner_id,
                "effect_type": int(effect["type"]),
                "select_type": int(effect["select_type"]),
                "select_context": int(effect["select_context"]),
                "target_condition_count": len(effect["target"]["conditions"]),
                "required_outcomes": _branch_obligations(effect),
                "status": "inventory_only",
            }
        )

    for skill in payload["skills"]:
        skill_id = int(skill["id"])
        card_id = int(skill["card_id"])
        for index, trigger in enumerate(skill["triggers"]):
            entries.append(
                {
                    "branch_id": f"skill:{skill_id}:trigger:{index}",
                    "kind": "trigger",
                    "owner_card_id": card_id,
                    "owner_kind": "skill",
                    "owner_id": skill_id,
                    "trigger_type": int(trigger["type"]),
                    "target_condition_count": len(trigger["subject"]["conditions"]),
                    "required_outcomes": (
                        ["condition_false", "condition_true"]
                        if trigger["subject"]["conditions"]
                        else []
                    ),
                    "status": "inventory_only",
                }
            )
        for index, effect in enumerate(skill["effects"]):
            append_effect(
                branch_id=f"skill:{skill_id}:effect:{index}",
                owner_card_id=card_id,
                owner_kind="skill",
                owner_id=skill_id,
                effect=effect,
            )
    for attack in payload["attacks"]:
        attack_id = int(attack["id"])
        card_id = int(attack["card_id"])
        for phase in ("pre_effects", "post_effects"):
            for index, effect in enumerate(attack[phase]):
                append_effect(
                    branch_id=f"attack:{attack_id}:{phase}:{index}",
                    owner_card_id=card_id,
                    owner_kind="attack",
                    owner_id=attack_id,
                    effect=effect,
                )
    return entries


def build_official_semantic_coverage(
    payload: dict[str, Any],
    oracle_id: str,
) -> tuple[dict[str, Any], CoverageSummary]:
    ir_summary = validate_official_ir(payload)
    effects = list(_effects(payload))
    targets = list(_targets(payload))
    branches = _branch_entries(payload)

    effect_counts = Counter(int(effect["type"]) for effect in effects)
    target_counts: Counter[int] = Counter()
    target_condition_counts: Counter[int] = Counter()
    effect_condition_counts = Counter(int(effect["condition_type"]) for effect in effects)
    select_type_counts = Counter(int(effect["select_type"]) for effect in effects)
    select_context_counts = Counter(int(effect["select_context"]) for effect in effects)
    trigger_counts: Counter[int] = Counter()

    for target in targets:
        for condition in target["conditions"]:
            condition_type = int(condition["type"])
            target_counts[condition_type] += 1
            target_condition_counts[condition_type] += 1
    for skill in payload["skills"]:
        for trigger in skill["triggers"]:
            trigger_counts[int(trigger["type"])] += 1

    for name, counts in (
        ("effects", effect_counts),
        ("targets", target_counts),
        ("conditions", effect_condition_counts),
        ("triggers", trigger_counts),
        ("select_types", select_type_counts),
        ("select_contexts", select_context_counts),
    ):
        invalid = sorted(value for value in counts if not 0 <= value < DOMAIN_SIZES[name])
        if invalid:
            raise ValueError(f"{name} values outside frozen domain: {invalid}")

    effect_entries = _entries(DOMAIN_SIZES["effects"], effect_counts)
    for entry in effect_entries:
        effect_id = int(entry["id"])
        if effect_id in PAIRED_EFFECTS:
            entry["name"] = PRIMITIVE_EFFECTS[effect_id]
            entry["status"] = "paired"
        elif effect_id in PRIMITIVE_EFFECTS:
            entry["name"] = PRIMITIVE_EFFECTS[effect_id]
            entry["status"] = "core_primitive_ready"

    continuation_entries = [
        {
            "id": int(row["id"]),
            "status": "registered_only",
        }
        for row in payload["continuations"]
    ]
    if len({entry["id"] for entry in continuation_entries}) != len(continuation_entries):
        raise ValueError("continuation coverage contains duplicate IDs")

    target_entries = _entries(DOMAIN_SIZES["targets"], target_counts)
    for entry in target_entries:
        target_id = int(entry["id"])
        if target_id in PAIRED_TARGETS:
            entry["status"] = "paired"
        elif target_id in PRIMITIVE_TARGETS:
            entry["status"] = "core_primitive_ready"
    condition_entries = _entries(
        DOMAIN_SIZES["conditions"], effect_condition_counts
    )
    for entry in condition_entries:
        condition_id = int(entry["id"])
        if condition_id in PAIRED_CONDITIONS:
            entry["status"] = "paired"
        elif condition_id in PRIMITIVE_CONDITIONS:
            entry["status"] = "core_primitive_ready"

    matrix: dict[str, Any] = {
        "version": 2,
        "oracle_id": oracle_id,
        "source_ir_sha256": ir_summary.canonical_sha256,
        "status_contract": {
            "not_started": "No independent CPU POD semantics exist.",
            "core_primitive_ready": (
                "A host/device primitive exists; target, condition, selection, trigger, "
                "and official differential gates are still required."
            ),
            "registered_only": (
                "The frozen ID is represented by the VM ABI, but its handler is not promoted."
            ),
            "inventory_only": (
                "The entity is present in the frozen IR; no independent official/POD/CUDA "
                "semantic replay has promoted it."
            ),
            "paired": "Official CPU vs CPU POD micro-fixture is exact.",
            "cuda_paired": "Official CPU, CPU POD, and CUDA replay are exact.",
        },
        "domains": {
            "cards": _entity_entries(
                payload["cards"],
                row_id="id",
                metadata=lambda row: {
                    "attack_ids": list(row["attack_ids"]),
                    "ability_id": int(row["ability_id"]),
                    "play_id": int(row["play_id"]),
                    "delay_id": int(row["delay_id"]),
                },
            ),
            "skills": _entity_entries(
                payload["skills"],
                row_id="id",
                metadata=lambda row: {
                    "card_id": int(row["card_id"]),
                    "effect_count": len(row["effects"]),
                    "trigger_count": len(row["triggers"]),
                },
            ),
            "attacks": _entity_entries(
                payload["attacks"],
                row_id="id",
                metadata=lambda row: {
                    "card_id": int(row["card_id"]),
                    "pre_effect_count": len(row["pre_effects"]),
                    "post_effect_count": len(row["post_effects"]),
                },
            ),
            "effects": effect_entries,
            "targets": target_entries,
            "conditions": condition_entries,
            "triggers": _entries(DOMAIN_SIZES["triggers"], trigger_counts),
            "select_types": _entries(
                DOMAIN_SIZES["select_types"], select_type_counts
            ),
            "select_contexts": _entries(
                DOMAIN_SIZES["select_contexts"], select_context_counts
            ),
            "select_options": _entries(
                DOMAIN_SIZES["select_options"], Counter()
            ),
            "continuations": continuation_entries,
            "branches": branches,
        },
        "target_condition_occurrences": sum(target_condition_counts.values()),
    }
    summary = CoverageSummary(
        cards=len(payload["cards"]),
        skills=len(payload["skills"]),
        attacks=len(payload["attacks"]),
        branch_records=len(branches),
        effect_occurrences=sum(effect_counts.values()),
        target_records=len(targets),
        target_type_occurrences=sum(target_counts.values()),
        target_condition_occurrences=sum(target_condition_counts.values()),
        trigger_occurrences=sum(trigger_counts.values()),
        used_effect_types=len(effect_counts),
        used_target_types=len(target_counts),
        used_condition_types=len(effect_condition_counts),
        used_trigger_types=len(trigger_counts),
        primitive_effect_types=sum(
            1 for effect_id in effect_counts if effect_id in PRIMITIVE_EFFECTS
        ),
        primitive_effect_occurrences=sum(
            count for effect_id, count in effect_counts.items() if effect_id in PRIMITIVE_EFFECTS
        ),
        primitive_target_types=sum(
            1 for target_id in target_counts if target_id in PRIMITIVE_TARGETS
        ),
        primitive_target_occurrences=sum(
            count for target_id, count in target_counts.items() if target_id in PRIMITIVE_TARGETS
        ),
        primitive_condition_types=sum(
            1 for condition_id in effect_condition_counts
            if condition_id in PRIMITIVE_CONDITIONS
        ),
        primitive_condition_occurrences=sum(
            count for condition_id, count in effect_condition_counts.items()
            if condition_id in PRIMITIVE_CONDITIONS
        ),
        paired_effect_types=sum(
            1 for effect_id in effect_counts if effect_id in PAIRED_EFFECTS
        ),
        paired_effect_occurrences=sum(
            count for effect_id, count in effect_counts.items()
            if effect_id in PAIRED_EFFECTS
        ),
        paired_target_types=sum(
            1 for target_id in target_counts if target_id in PAIRED_TARGETS
        ),
        paired_target_occurrences=sum(
            count for target_id, count in target_counts.items()
            if target_id in PAIRED_TARGETS
        ),
        paired_condition_types=sum(
            1 for condition_id in effect_condition_counts
            if condition_id in PAIRED_CONDITIONS
        ),
        paired_condition_occurrences=sum(
            count for condition_id, count in effect_condition_counts.items()
            if condition_id in PAIRED_CONDITIONS
        ),
        payload_sha256="",
    )
    matrix["summary"] = summary.to_dict()
    payload_hash = _canonical_hash(matrix)
    summary = CoverageSummary(**{**summary.to_dict(), "payload_sha256": payload_hash})
    matrix["summary"] = summary.to_dict()
    return matrix, summary


def validate_official_coverage_inventory(
    payload: dict[str, Any], matrix: dict[str, Any]
) -> dict[str, Any]:
    """Validate that a coverage file is a lossless inventory of the frozen IR.

    This intentionally does not promote any semantic status.  It proves that
    every card, skill, attack, and effect/trigger branch is represented and
    that the matrix was built from this exact IR digest.
    """
    ir_summary = validate_official_ir(payload)
    if matrix.get("version") != 2:
        raise ValueError("unsupported coverage matrix version")
    if matrix.get("source_ir_sha256") != ir_summary.canonical_sha256:
        raise ValueError("coverage source IR hash mismatch")
    domains = matrix.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("coverage domains are missing")

    expected_entities = {
        "cards": [int(row["id"]) for row in payload["cards"]],
        "skills": [int(row["id"]) for row in payload["skills"]],
        "attacks": [int(row["id"]) for row in payload["attacks"]],
    }
    for name, expected_ids in expected_entities.items():
        rows = domains.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"coverage domain {name} is missing")
        actual_ids = [row.get("id") for row in rows]
        if actual_ids != expected_ids:
            raise ValueError(f"coverage domain {name} IDs do not match IR")
        if any(row.get("status") not in {"inventory_only", "paired", "cuda_paired"} for row in rows):
            raise ValueError(f"coverage domain {name} contains an invalid status")

    expected_branches = _branch_entries(payload)
    actual_branches = domains.get("branches")
    if not isinstance(actual_branches, list) or len(actual_branches) != len(expected_branches):
        raise ValueError("coverage branch inventory does not match IR")
    for actual, expected in zip(actual_branches, expected_branches):
        if actual.get("status") not in {"inventory_only", "paired", "cuda_paired"}:
            raise ValueError("coverage branch inventory contains an invalid status")
        actual_contract = dict(actual)
        expected_contract = dict(expected)
        actual_contract.pop("status", None)
        expected_contract.pop("status", None)
        if actual_contract != expected_contract:
            raise ValueError("coverage branch inventory does not match IR")
    summary = matrix.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("coverage summary is missing")
    if summary.get("cards") != len(expected_entities["cards"]):
        raise ValueError("coverage card summary mismatch")
    if summary.get("skills") != len(expected_entities["skills"]):
        raise ValueError("coverage skill summary mismatch")
    if summary.get("attacks") != len(expected_entities["attacks"]):
        raise ValueError("coverage attack summary mismatch")
    if summary.get("branch_records") != len(expected_branches):
        raise ValueError("coverage branch summary mismatch")
    return {
        "cards": len(expected_entities["cards"]),
        "skills": len(expected_entities["skills"]),
        "attacks": len(expected_entities["attacks"]),
        "branch_records": len(expected_branches),
        "source_ir_sha256": ir_summary.canonical_sha256,
    }

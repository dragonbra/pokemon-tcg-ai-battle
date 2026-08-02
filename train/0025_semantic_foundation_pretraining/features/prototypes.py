"""Finite typed card prototypes exported from the read-only official runtime API."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "0025_official_public_prototypes_v1"


class FieldState(IntEnum):
    """A value of zero remains distinct from missing and padding."""

    PAD = 0
    PRESENT = 1
    UNKNOWN = 2
    NOT_APPLICABLE = 3


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _optional(value: Any, *, applicable: bool = True) -> dict[str, Any]:
    if not applicable:
        return {"state": int(FieldState.NOT_APPLICABLE), "value": 0}
    if value is None:
        return {"state": int(FieldState.UNKNOWN), "value": 0}
    return {"state": int(FieldState.PRESENT), "value": int(value)}


def export_public_runtime_prototypes(output: Path) -> dict[str, Any]:
    """Export all public CardData/Attack records; this never edits engine sources."""
    from evaluation.arena.frozen._policy.cg import api

    cards = []
    for card in sorted(api.all_card_data(), key=lambda item: item.cardId):
        is_pokemon = int(card.cardType) == int(api.CardType.POKEMON)
        skills = [
            {
                "local_skill_ref": int(card.cardId) * 8 + ordinal + 1,
                "ordinal": ordinal,
                "name": skill.name,
                "text": skill.text,
                "text_sha256": _sha256_bytes(skill.text.encode("utf-8")),
                "identity_scope": "card_local_derived_ref",
            }
            for ordinal, skill in enumerate(card.skills)
        ]
        cards.append(
            {
                "card_id": int(card.cardId),
                "name": card.name,
                "card_type": int(card.cardType),
                "hp": _optional(card.hp, applicable=is_pokemon),
                "retreat_cost": _optional(card.retreatCost, applicable=is_pokemon),
                "weakness_type": _optional(card.weakness, applicable=is_pokemon),
                "resistance_type": _optional(card.resistance, applicable=is_pokemon),
                "energy_type": _optional(card.energyType),
                "stage": {
                    "basic": bool(card.basic),
                    "stage1": bool(card.stage1),
                    "stage2": bool(card.stage2),
                },
                "rule_flags": {
                    "ex": bool(card.ex),
                    "mega_ex": bool(card.megaEx),
                    "tera": bool(card.tera),
                    "ace_spec": bool(card.aceSpec),
                },
                "evolves_from": card.evolvesFrom,
                "skills": skills,
                "attack_ids": [int(value) for value in card.attacks],
            }
        )
    attacks = [
        {
            "attack_id": int(attack.attackId),
            "name": attack.name,
            "text": attack.text,
            "text_sha256": _sha256_bytes(attack.text.encode("utf-8")),
            "base_damage": _optional(attack.damage),
            "energy_types": [int(value) for value in attack.energies],
            "energy_count": len(attack.energies),
        }
        for attack in sorted(api.all_attack(), key=lambda item: item.attackId)
    ]
    attack_ids = {item["attack_id"] for item in attacks}
    dangling = sorted(
        attack_id for card in cards for attack_id in card["attack_ids"] if attack_id not in attack_ids
    )
    if dangling:
        raise ValueError(f"official card table references missing attacks: {dangling[:8]}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "evidence_scope": "official runtime public CardData and Attack tables",
        "limitations": [
            "Skill IDs are not exposed by the public API; local refs are card-scoped and derived.",
            "Structured Effect/Target/Condition chains require the separate full-engine extractor revision.",
        ],
        "field_state_codes": {item.name.lower(): int(item) for item in FieldState},
        "cards": cards,
        "attacks": attacks,
    }
    body["content_sha256"] = _sha256_bytes(canonical_json(body))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(body))
    return body


@dataclass(frozen=True)
class PrototypeIndex:
    payload: Mapping[str, Any]
    cards: Mapping[int, Mapping[str, Any]]
    attacks: Mapping[int, Mapping[str, Any]]
    engine_payload: Mapping[str, Any]
    engine_cards: Mapping[int, Mapping[str, Any]]
    engine_attacks: Mapping[int, Mapping[str, Any]]
    skills: Mapping[int, Mapping[str, Any]]
    effects: Mapping[int, Mapping[str, Any]]
    skill_effect_refs: Mapping[int, tuple[int, ...]]
    attack_effect_refs: Mapping[int, tuple[int, ...]]

    @classmethod
    def load(cls, path: Path | str, full_engine_path: Path | str | None = None) -> "PrototypeIndex":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported prototype schema")
        claimed = payload.get("content_sha256")
        unhashed = dict(payload)
        unhashed.pop("content_sha256", None)
        if claimed != _sha256_bytes(canonical_json(unhashed)):
            raise ValueError("prototype content commitment mismatch")
        cards = {int(item["card_id"]): item for item in payload["cards"]}
        attacks = {int(item["attack_id"]): item for item in payload["attacks"]}
        if len(cards) != len(payload["cards"]) or len(attacks) != len(payload["attacks"]):
            raise ValueError("duplicate prototype identity")
        if full_engine_path is None:
            full_engine_path = path.with_name("official_full_engine_prototypes_v1.json")
        engine_payload = json.loads(Path(full_engine_path).read_text(encoding="utf-8"))
        if engine_payload.get("schema_version") != "0025_official_full_engine_prototypes_v1":
            raise ValueError("unsupported full-engine prototype schema")
        engine_cards = {int(item["card_id"]): item for item in engine_payload["cards"]}
        skills = {int(item["skill_id"]): item for item in engine_payload["skills"]}
        engine_attacks = {int(item["attack_id"]): item for item in engine_payload["attacks"]}
        if set(engine_cards) != set(cards) or set(engine_attacks) != set(attacks):
            raise ValueError("public and full-engine prototype identity sets disagree")
        effects: dict[int, Mapping[str, Any]] = {}
        skill_effect_refs: dict[int, tuple[int, ...]] = {}
        attack_effect_refs: dict[int, tuple[int, ...]] = {}
        effect_ref = 1
        for skill_id, skill in sorted(skills.items()):
            refs: list[int] = []
            for ordinal, effect in enumerate(skill["effects"]):
                effects[effect_ref] = {"parent_kind": 1, "parent_id": skill_id, "phase": 0, "ordinal": ordinal, **effect}
                refs.append(effect_ref)
                effect_ref += 1
            skill_effect_refs[skill_id] = tuple(refs)
        for attack_id, attack in sorted(engine_attacks.items()):
            refs = []
            for phase, name in ((1, "pre_effects"), (2, "post_effects")):
                for ordinal, effect in enumerate(attack[name]):
                    effects[effect_ref] = {"parent_kind": 2, "parent_id": attack_id, "phase": phase, "ordinal": ordinal, **effect}
                    refs.append(effect_ref)
                    effect_ref += 1
            attack_effect_refs[attack_id] = tuple(refs)
        return cls(
            payload, cards, attacks, engine_payload, engine_cards, engine_attacks,
            skills, effects, skill_effect_refs, attack_effect_refs,
        )


__all__ = ["FieldState", "PrototypeIndex", "SCHEMA_VERSION", "export_public_runtime_prototypes"]

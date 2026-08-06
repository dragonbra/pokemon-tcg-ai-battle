"""Validated Card, Attack, and Skill prototype index for the audited actor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


PUBLIC_SCHEMA_VERSION = "0025_official_public_prototypes_v1"
ENGINE_SCHEMA_VERSION = "0031_official_full_engine_prototypes_v2"


class FieldState(IntEnum):
    """Numeric state; present zero is distinct from missing and padding."""

    PAD = 0
    PRESENT = 1
    UNKNOWN = 2
    NOT_APPLICABLE = 3


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class PrototypeIndex:
    payload: Mapping[str, Any]
    cards: Mapping[int, Mapping[str, Any]]
    attacks: Mapping[int, Mapping[str, Any]]
    engine_payload: Mapping[str, Any]
    engine_cards: Mapping[int, Mapping[str, Any]]
    engine_attacks: Mapping[int, Mapping[str, Any]]
    skills: Mapping[int, Mapping[str, Any]]

    @classmethod
    def empty(cls) -> "PrototypeIndex":
        empty: Mapping = MappingProxyType({})
        return cls(empty, empty, empty, empty, empty, empty, empty)

    @classmethod
    def load(
        cls,
        public_path: Path | str,
        full_engine_path: Path | str | None = None,
    ) -> "PrototypeIndex":
        public_path = Path(public_path)
        payload = json.loads(public_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PUBLIC_SCHEMA_VERSION:
            raise ValueError("unsupported public prototype schema")
        claimed = payload.get("content_sha256")
        unhashed = dict(payload)
        unhashed.pop("content_sha256", None)
        if claimed != _sha256(canonical_json(unhashed)):
            raise ValueError("public prototype content commitment mismatch")

        cards = {int(item["card_id"]): item for item in payload["cards"]}
        attacks = {int(item["attack_id"]): item for item in payload["attacks"]}
        if len(cards) != len(payload["cards"]) or len(attacks) != len(payload["attacks"]):
            raise ValueError("duplicate public prototype identity")

        engine_path = (
            Path(full_engine_path)
            if full_engine_path is not None
            else public_path.with_name("official_full_engine_prototypes_v2.json")
        )
        engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
        if engine_payload.get("schema_version") != ENGINE_SCHEMA_VERSION:
            raise ValueError("unsupported full-engine prototype schema")
        engine_cards = {int(item["card_id"]): item for item in engine_payload["cards"]}
        engine_attacks = {
            int(item["attack_id"]): item for item in engine_payload["attacks"]
        }
        skills = {int(item["skill_id"]): item for item in engine_payload["skills"]}
        if set(engine_cards) != set(cards) or set(engine_attacks) != set(attacks):
            raise ValueError("public and full-engine prototype identity sets disagree")
        if len(skills) != len(engine_payload["skills"]):
            raise ValueError("duplicate full-engine skill identity")
        return cls(
            payload=payload,
            cards=cards,
            attacks=attacks,
            engine_payload=engine_payload,
            engine_cards=engine_cards,
            engine_attacks=engine_attacks,
            skills=skills,
        )


__all__ = [
    "ENGINE_SCHEMA_VERSION",
    "FieldState",
    "PrototypeIndex",
    "PUBLIC_SCHEMA_VERSION",
    "canonical_json",
]

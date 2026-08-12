from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .official_ir import OfficialIRSummary, validate_official_ir


MAGIC = b"PTCGRUL1"
ABI_VERSION = 1
HEADER_SIZE = 256
SECTION_NAMES = (
    "cards",
    "skills",
    "attacks",
    "effects",
    "targets",
    "conditions",
    "triggers",
    "card_attack_ids",
    "name_sets",
    "name_card_ids",
    "continuations",
    "reserved",
)
HEADER = struct.Struct("<8sIIQ12I12Q32s56x")
CARD = struct.Struct("<Q19i12x")
SKILL = struct.Struct("<Q16i8x")
ATTACK = struct.Struct("<Q17i20x")
EFFECT = struct.Struct("<Q15i12x")
TARGET = struct.Struct("<10i8x")
CONDITION = struct.Struct("<5i12x")
TRIGGER = struct.Struct("<2i8x")
NAME_SET = struct.Struct("<9i12x")
CONTINUATION = struct.Struct("<I12x")
UINT32 = struct.Struct("<I")

ROW_SIZES = {
    "cards": CARD.size,
    "skills": SKILL.size,
    "attacks": ATTACK.size,
    "effects": EFFECT.size,
    "targets": TARGET.size,
    "conditions": CONDITION.size,
    "triggers": TRIGGER.size,
    "card_attack_ids": UINT32.size,
    "name_sets": NAME_SET.size,
    "name_card_ids": UINT32.size,
    "continuations": CONTINUATION.size,
    "reserved": 1,
}


@dataclass(frozen=True)
class OfficialRulePackSummary:
    total_bytes: int
    payload_sha256: str
    counts: dict[str, int]
    offsets: dict[str, int]
    source_ir_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "payload_sha256": self.payload_sha256,
            "counts": self.counts,
            "offsets": self.offsets,
            "source_ir_sha256": self.source_ir_sha256,
        }


def _align16(buffer: bytearray) -> None:
    buffer.extend(b"\0" * ((-len(buffer)) % 16))


def _pack_rows(format_: struct.Struct, rows: Iterable[tuple[Any, ...]]) -> bytes:
    output = bytearray()
    for row in rows:
        output.extend(format_.pack(*row))
    return bytes(output)


def compile_official_rule_pack(
    payload: dict[str, Any],
) -> tuple[bytes, OfficialRulePackSummary, OfficialIRSummary]:
    ir_summary = validate_official_ir(payload)
    effects: list[tuple[Any, ...]] = []
    targets: list[tuple[Any, ...]] = []
    conditions: list[tuple[Any, ...]] = []
    triggers: list[tuple[Any, ...]] = []
    card_attack_ids: list[int] = []
    name_card_ids: list[int] = []

    def append_target(target: dict[str, Any]) -> int:
        condition_offset = len(conditions)
        for condition in target["conditions"]:
            conditions.append(
                (
                    int(condition["type"]),
                    int(condition["comparator"]),
                    int(condition["value"]),
                    int(condition["value2"]),
                    int(condition["name_id"]),
                )
            )
        areas = [int(value) for value in target["areas"]]
        if len(areas) > 4:
            raise ValueError("target area capacity exceeded")
        target_index = len(targets)
        targets.append(
            (
                int(target["player"]),
                int(target["not_me"]),
                int(target["skip_enemy_target"]),
                *(areas + [0] * (4 - len(areas))),
                len(areas),
                condition_offset,
                len(target["conditions"]),
            )
        )
        return target_index

    def append_effect(effect: dict[str, Any]) -> None:
        target_index = append_target(effect["target"])
        effects.append(
            (
                int(effect["flags"]),
                int(effect["type"]),
                int(effect["select_type"]),
                int(effect["select_count"]),
                int(effect["select_context"]),
                int(effect["loop_count"]),
                int(effect["priority"]),
                int(effect["values"][0]),
                int(effect["values"][1]),
                int(effect["condition_type"]),
                int(effect["comparator"]),
                int(effect["fail_skip"]),
                int(effect["skill_id"]),
                int(effect["linked_skill_id"]),
                int(effect["linked_attack_id"]),
                target_index,
            )
        )

    card_rows: list[tuple[Any, ...]] = []
    for card in payload["cards"]:
        attack_offset = len(card_attack_ids)
        card_attack_ids.extend(int(value) for value in card["attack_ids"])
        card_rows.append(
            (
                int(card["flags"]),
                int(card["id"]),
                int(card["card_type"]),
                int(card["pokemon_type"]),
                int(card["evolution_type"]),
                int(card["retreat_cost"]),
                int(card["hp"]),
                int(card["weakness"]),
                int(card["resistance"]),
                int(card["energy_type"]),
                int(card["energy_count"]),
                int(card["number"]),
                int(card["name_id"]),
                int(card["evolves_from_name_id"]),
                int(card["evolves_from2_name_id"]),
                int(card["ability_id"]),
                int(card["play_id"]),
                int(card["delay_id"]),
                attack_offset,
                len(card["attack_ids"]),
            )
        )

    skill_rows: list[tuple[Any, ...]] = []
    for skill in payload["skills"]:
        trigger_offset = len(triggers)
        for trigger in skill["triggers"]:
            triggers.append((int(trigger["type"]), append_target(trigger["subject"])))
        effect_offset = len(effects)
        for effect in skill["effects"]:
            append_effect(effect)
        areas = [int(value) for value in skill["areas"]]
        if len(areas) > 2:
            raise ValueError("skill area capacity exceeded")
        skill_rows.append(
            (
                int(skill["flags"]),
                int(skill["id"]),
                int(skill["card_id"]),
                int(skill["skill_type"]),
                int(skill["priority"]),
                int(skill["first_condition_count"]),
                int(skill["second_effect_start"]),
                int(skill["second_effect_start_enemy"]),
                int(skill["trigger_start"]),
                int(skill["name_id"]),
                *(areas + [0] * (2 - len(areas))),
                len(areas),
                trigger_offset,
                len(skill["triggers"]),
                effect_offset,
                len(skill["effects"]),
            )
        )

    attack_rows: list[tuple[Any, ...]] = []
    for attack in payload["attacks"]:
        energies = [int(value) for value in attack["energies"]]
        if len(energies) > 7:
            raise ValueError("attack energy capacity exceeded")
        pre_offset = len(effects)
        for effect in attack["pre_effects"]:
            append_effect(effect)
        post_offset = len(effects)
        for effect in attack["post_effects"]:
            append_effect(effect)
        attack_rows.append(
            (
                int(attack["flags"]),
                int(attack["id"]),
                int(attack["card_id"]),
                int(attack["damage"]),
                int(attack["last_cancel_fail_attack"]),
                int(attack["name_id"]),
                *(energies + [0] * (7 - len(energies))),
                len(energies),
                pre_offset,
                len(attack["pre_effects"]),
                post_offset,
                len(attack["post_effects"]),
            )
        )

    name_set_rows: list[tuple[Any, ...]] = []
    for name_set in payload["name_sets"]:
        spans: list[int] = []
        for family in ("equal_cards", "contains_cards", "ability_cards", "attack_cards"):
            values = [int(value) for value in name_set[family]]
            spans.extend((len(name_card_ids), len(values)))
            name_card_ids.extend(values)
        name_set_rows.append((int(name_set["id"]), *spans))

    sections = {
        "cards": _pack_rows(CARD, card_rows),
        "skills": _pack_rows(SKILL, skill_rows),
        "attacks": _pack_rows(ATTACK, attack_rows),
        "effects": _pack_rows(EFFECT, effects),
        "targets": _pack_rows(TARGET, targets),
        "conditions": _pack_rows(CONDITION, conditions),
        "triggers": _pack_rows(TRIGGER, triggers),
        "card_attack_ids": _pack_rows(UINT32, ((value,) for value in card_attack_ids)),
        "name_sets": _pack_rows(NAME_SET, name_set_rows),
        "name_card_ids": _pack_rows(UINT32, ((value,) for value in name_card_ids)),
        "continuations": _pack_rows(
            CONTINUATION, ((int(row["id"]),) for row in payload["continuations"])
        ),
        "reserved": b"",
    }
    counts = {
        "cards": len(card_rows),
        "skills": len(skill_rows),
        "attacks": len(attack_rows),
        "effects": len(effects),
        "targets": len(targets),
        "conditions": len(conditions),
        "triggers": len(triggers),
        "card_attack_ids": len(card_attack_ids),
        "name_sets": len(name_set_rows),
        "name_card_ids": len(name_card_ids),
        "continuations": len(payload["continuations"]),
        "reserved": 0,
    }

    output = bytearray(b"\0" * HEADER_SIZE)
    offsets: dict[str, int] = {}
    for name in SECTION_NAMES:
        _align16(output)
        offsets[name] = len(output)
        output.extend(sections[name])
    payload_hash = hashlib.sha256(output[HEADER_SIZE:]).digest()
    header = HEADER.pack(
        MAGIC,
        int(payload["schema_version"]),
        ABI_VERSION,
        len(output),
        *(counts[name] for name in SECTION_NAMES),
        *(offsets[name] for name in SECTION_NAMES),
        payload_hash,
    )
    output[:HEADER_SIZE] = header
    source_ir_hash = ir_summary.canonical_sha256
    summary = OfficialRulePackSummary(
        total_bytes=len(output),
        payload_sha256=payload_hash.hex(),
        counts=counts,
        offsets=offsets,
        source_ir_sha256=source_ir_hash,
    )
    validate_official_rule_pack(bytes(output))
    return bytes(output), summary, ir_summary


def validate_official_rule_pack(data: bytes) -> OfficialRulePackSummary:
    if len(data) < HEADER_SIZE:
        raise ValueError("official rule pack is truncated")
    unpacked = HEADER.unpack_from(data)
    magic, schema_version, abi_version, total_bytes = unpacked[:4]
    if magic != MAGIC:
        raise ValueError("official rule pack magic mismatch")
    if schema_version != 1 or abi_version != ABI_VERSION:
        raise ValueError("unsupported official rule pack version")
    if total_bytes != len(data):
        raise ValueError("official rule pack byte count mismatch")
    counts_raw = unpacked[4:16]
    offsets_raw = unpacked[16:28]
    expected_hash = unpacked[28]
    actual_hash = hashlib.sha256(data[HEADER_SIZE:]).digest()
    if expected_hash != actual_hash:
        raise ValueError("official rule pack payload hash mismatch")
    counts = dict(zip(SECTION_NAMES, counts_raw))
    offsets = dict(zip(SECTION_NAMES, offsets_raw))
    previous = HEADER_SIZE
    for index, name in enumerate(SECTION_NAMES):
        offset = offsets[name]
        end = offsets[SECTION_NAMES[index + 1]] if index + 1 < len(SECTION_NAMES) else len(data)
        if offset < previous or offset % 16 != 0 or end < offset or end > len(data):
            raise ValueError(f"invalid section bounds for {name}")
        required = counts[name] * ROW_SIZES[name]
        if required > end - offset:
            raise ValueError(f"section {name} is truncated")
        previous = offset
    return OfficialRulePackSummary(
        total_bytes=len(data),
        payload_sha256=actual_hash.hex(),
        counts=counts,
        offsets=offsets,
        source_ir_sha256="",
    )


def write_official_rule_pack(path: str | Path, payload: dict[str, Any]) -> OfficialRulePackSummary:
    output = Path(path)
    data, summary, _ = compile_official_rule_pack(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return summary

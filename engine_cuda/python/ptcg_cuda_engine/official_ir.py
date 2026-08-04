from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


OFFICIAL_IR_SCHEMA_VERSION = 1

# The official engine hard-codes these TargetType families by card name.  The
# extractor normalizes them into numeric card sets so the private IR carries no
# source text.  Treat their shape as part of the frozen schema; accepting a
# missing or malformed set would silently turn a reachable target into no-op.
SPECIAL_TARGET_NAME_SETS = {
    51: ("equal_cards", 2),
    52: ("contains_cards", 2),
    53: ("equal_cards", 3),
}


@dataclass(frozen=True)
class OfficialIRSummary:
    cards: int
    skills: int
    attacks: int
    continuations: int
    name_sets: int
    effects: int
    triggers: int
    target_conditions: int
    effect_types_used: tuple[int, ...]
    target_types_used: tuple[int, ...]
    select_contexts_used: tuple[int, ...]
    unique_effect_signatures: int
    canonical_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cards": self.cards,
            "skills": self.skills,
            "attacks": self.attacks,
            "continuations": self.continuations,
            "name_sets": self.name_sets,
            "effects": self.effects,
            "triggers": self.triggers,
            "target_conditions": self.target_conditions,
            "effect_types_used": list(self.effect_types_used),
            "target_types_used": list(self.target_types_used),
            "select_contexts_used": list(self.select_contexts_used),
            "unique_effect_signatures": self.unique_effect_signatures,
            "canonical_sha256": self.canonical_sha256,
        }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{path} schema mismatch: missing={missing}, unexpected={unexpected}"
        )


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _index_unique(rows: Iterable[Mapping[str, Any]], path: str) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        row_id = _require_int(row.get("id"), f"{path}[{index}].id")
        if row_id <= 0:
            raise ValueError(f"{path}[{index}].id must be positive")
        if row_id in result:
            raise ValueError(f"duplicate {path} ID: {row_id}")
        result[row_id] = row
    if list(result) != sorted(result):
        raise ValueError(f"{path} must be sorted by ID")
    return result


def _validate_target(
    target: Any,
    *,
    path: str,
    name_ids: set[int],
    name_sets: Mapping[int, Mapping[str, Any]],
    target_types: set[int],
) -> int:
    raw = _require_mapping(target, path)
    _require_exact_keys(
        raw,
        {"player", "not_me", "skip_enemy_target", "areas", "conditions"},
        path,
    )
    for field in ("player", "not_me", "skip_enemy_target"):
        _require_int(raw.get(field), f"{path}.{field}")
    areas = _require_list(raw.get("areas"), f"{path}.areas")
    for index, area in enumerate(areas):
        _require_int(area, f"{path}.areas[{index}]")
    conditions = _require_list(raw.get("conditions"), f"{path}.conditions")
    for index, condition_value in enumerate(conditions):
        condition_path = f"{path}.conditions[{index}]"
        condition = _require_mapping(condition_value, condition_path)
        _require_exact_keys(
            condition,
            {"type", "comparator", "value", "value2", "name_id"},
            condition_path,
        )
        condition_type = _require_int(condition.get("type"), f"{condition_path}.type")
        target_types.add(condition_type)
        for field in ("comparator", "value", "value2"):
            _require_int(condition.get(field), f"{condition_path}.{field}")
        condition_name_id = _require_int(
            condition.get("name_id"), f"{condition_path}.name_id"
        )
        if condition_name_id not in name_ids | {0}:
            raise ValueError(f"{condition_path}.name_id references {condition_name_id}")
        special_set = SPECIAL_TARGET_NAME_SETS.get(condition_type)
        if special_set is not None:
            family, expected_count = special_set
            if condition_name_id == 0:
                raise ValueError(
                    f"{condition_path}.name_id must reference the TargetType "
                    f"{condition_type} numeric name set"
                )
            referenced = name_sets[condition_name_id]
            actual_count = len(referenced[family])
            if actual_count != expected_count:
                raise ValueError(
                    f"{condition_path}.name_id TargetType {condition_type} requires "
                    f"{expected_count} {family}, actual={actual_count}"
                )
    return len(conditions)


def _effect_signature(effect: Mapping[str, Any]) -> str:
    semantic = {
        key: effect[key]
        for key in (
            "type",
            "select_type",
            "select_count",
            "select_context",
            "flags",
            "loop_count",
            "priority",
            "values",
            "condition_type",
            "comparator",
            "fail_skip",
            "skill_id",
            "linked_skill_id",
            "linked_attack_id",
            "target",
        )
    }
    return canonical_sha256(semantic)


def validate_official_ir(payload: Any) -> OfficialIRSummary:
    root = _require_mapping(payload, "root")
    _require_exact_keys(
        root,
        {
            "schema_version",
            "counts",
            "flag_schema",
            "name_sets",
            "cards",
            "skills",
            "attacks",
            "continuations",
        },
        "root",
    )
    schema_version = _require_int(root.get("schema_version"), "schema_version")
    if schema_version != OFFICIAL_IR_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported official IR schema {schema_version}; "
            f"expected {OFFICIAL_IR_SCHEMA_VERSION}"
        )

    counts = _require_mapping(root.get("counts"), "counts")
    _require_exact_keys(
        counts,
        {"cards", "skills", "attacks", "continuations", "name_sets"},
        "counts",
    )
    flag_schema = _require_mapping(root.get("flag_schema"), "flag_schema")
    _require_exact_keys(flag_schema, {"card", "skill", "effect"}, "flag_schema")
    for family in ("card", "skill", "effect"):
        names = _require_list(flag_schema.get(family), f"flag_schema.{family}")
        if not names or len(names) != len(set(names)):
            raise ValueError(f"flag_schema.{family} must contain unique names")

    name_rows = [
        _require_mapping(value, f"name_sets[{index}]")
        for index, value in enumerate(_require_list(root.get("name_sets"), "name_sets"))
    ]
    name_index = _index_unique(name_rows, "name_sets")
    if set(name_index) != set(range(1, len(name_index) + 1)):
        raise ValueError("name set IDs must be dense and one-based")
    for name_set_id, row in name_index.items():
        _require_exact_keys(
            row,
            {"id", "equal_cards", "contains_cards", "ability_cards", "attack_cards"},
            f"name_sets[{name_set_id}]",
        )
        for family in ("equal_cards", "contains_cards", "ability_cards", "attack_cards"):
            values = _require_list(row.get(family), f"name_sets[{name_set_id}].{family}")
            parsed = [
                _require_int(value, f"name_sets[{name_set_id}].{family}[{index}]")
                for index, value in enumerate(values)
            ]
            if parsed != sorted(set(parsed)):
                raise ValueError(f"name_sets[{name_set_id}].{family} must be sorted and unique")

    card_rows = [
        _require_mapping(value, f"cards[{index}]")
        for index, value in enumerate(_require_list(root.get("cards"), "cards"))
    ]
    skill_rows = [
        _require_mapping(value, f"skills[{index}]")
        for index, value in enumerate(_require_list(root.get("skills"), "skills"))
    ]
    attack_rows = [
        _require_mapping(value, f"attacks[{index}]")
        for index, value in enumerate(_require_list(root.get("attacks"), "attacks"))
    ]
    cards = _index_unique(card_rows, "cards")
    skills = _index_unique(skill_rows, "skills")
    attacks = _index_unique(attack_rows, "attacks")
    if set(cards) != set(range(1, len(cards) + 1)):
        raise ValueError("card IDs must be dense and one-based")
    if set(attacks) != set(range(1, len(attacks) + 1)):
        raise ValueError("attack IDs must be dense and one-based")
    if skills and set(skills) != set(range(min(skills), max(skills) + 1)):
        raise ValueError("skill IDs must be dense")
    name_ids = set(name_index)

    for name_set_id, row in name_index.items():
        for family in ("equal_cards", "contains_cards", "ability_cards", "attack_cards"):
            for card_id in row[family]:
                if card_id not in cards:
                    raise ValueError(
                        f"name_sets[{name_set_id}].{family} references card {card_id}"
                    )

    for count_name, actual in (
        ("cards", len(cards)),
        ("skills", len(skills)),
        ("attacks", len(attacks)),
        ("name_sets", len(name_index)),
    ):
        recorded = _require_int(counts.get(count_name), f"counts.{count_name}")
        if recorded != actual:
            raise ValueError(f"counts.{count_name}={recorded}, actual={actual}")

    for card_id, card in cards.items():
        _require_exact_keys(
            card,
            {
                "id", "card_type", "pokemon_type", "evolution_type", "retreat_cost",
                "hp", "weakness", "resistance", "energy_type", "energy_count", "flags",
                "number", "name_id", "evolves_from_name_id", "evolves_from2_name_id",
                "ability_id", "play_id", "delay_id", "attack_ids",
            },
            f"cards[{card_id}]",
        )
        for field in (
            "card_type",
            "pokemon_type",
            "evolution_type",
            "retreat_cost",
            "hp",
            "weakness",
            "resistance",
            "energy_type",
            "energy_count",
            "flags",
            "number",
        ):
            _require_int(card.get(field), f"cards[{card_id}].{field}")
        for field in ("name_id", "evolves_from_name_id", "evolves_from2_name_id"):
            ref = _require_int(card.get(field), f"cards[{card_id}].{field}")
            if ref not in name_ids | {0}:
                raise ValueError(f"cards[{card_id}].{field} references {ref}")
        for field in ("ability_id", "play_id", "delay_id"):
            ref = _require_int(card.get(field), f"cards[{card_id}].{field}")
            if ref not in set(skills) | {0}:
                raise ValueError(f"cards[{card_id}].{field} references skill {ref}")
        attack_ids = _require_list(card.get("attack_ids"), f"cards[{card_id}].attack_ids")
        for index, attack_id_value in enumerate(attack_ids):
            attack_id = _require_int(
                attack_id_value, f"cards[{card_id}].attack_ids[{index}]"
            )
            if attack_id not in attacks:
                raise ValueError(f"cards[{card_id}] references attack {attack_id}")

    effect_types: set[int] = set()
    target_types: set[int] = set()
    select_contexts: set[int] = set()
    effect_signatures: set[str] = set()
    effect_count = 0
    trigger_count = 0
    condition_count = 0

    def validate_effects(values: Any, path: str) -> None:
        nonlocal effect_count, condition_count
        effects = _require_list(values, path)
        for index, effect_value in enumerate(effects):
            effect_path = f"{path}[{index}]"
            effect = _require_mapping(effect_value, effect_path)
            _require_exact_keys(
                effect,
                {
                    "type", "select_type", "select_count", "select_context", "flags",
                    "loop_count", "priority", "values", "condition_type", "comparator",
                    "fail_skip", "skill_id", "linked_skill_id", "linked_attack_id", "target",
                },
                effect_path,
            )
            effect_type = _require_int(effect.get("type"), f"{effect_path}.type")
            select_context = _require_int(
                effect.get("select_context"), f"{effect_path}.select_context"
            )
            effect_types.add(effect_type)
            select_contexts.add(select_context)
            for field in (
                "select_type",
                "select_count",
                "flags",
                "loop_count",
                "priority",
                "condition_type",
                "comparator",
                "fail_skip",
                "skill_id",
            ):
                _require_int(effect.get(field), f"{effect_path}.{field}")
            values_field = _require_list(effect.get("values"), f"{effect_path}.values")
            if len(values_field) != 2:
                raise ValueError(f"{effect_path}.values must contain two integers")
            for value_index, value in enumerate(values_field):
                _require_int(value, f"{effect_path}.values[{value_index}]")
            linked_skill = _require_int(
                effect.get("linked_skill_id"), f"{effect_path}.linked_skill_id"
            )
            linked_attack = _require_int(
                effect.get("linked_attack_id"), f"{effect_path}.linked_attack_id"
            )
            if linked_skill not in set(skills) | {0}:
                raise ValueError(f"{effect_path} references skill {linked_skill}")
            if linked_attack not in set(attacks) | {0}:
                raise ValueError(f"{effect_path} references attack {linked_attack}")
            condition_count += _validate_target(
                effect.get("target"),
                path=f"{effect_path}.target",
                name_ids=name_ids,
                name_sets=name_index,
                target_types=target_types,
            )
            effect_signatures.add(_effect_signature(effect))
            effect_count += 1

    for skill_id, skill in skills.items():
        _require_exact_keys(
            skill,
            {
                "id", "card_id", "skill_type", "flags", "priority",
                "first_condition_count", "second_effect_start", "second_effect_start_enemy",
                "trigger_start", "name_id", "areas", "triggers", "effects",
            },
            f"skills[{skill_id}]",
        )
        card_id = _require_int(skill.get("card_id"), f"skills[{skill_id}].card_id")
        if card_id not in cards:
            raise ValueError(f"skills[{skill_id}] references card {card_id}")
        for field in (
            "skill_type",
            "flags",
            "priority",
            "first_condition_count",
            "second_effect_start",
            "second_effect_start_enemy",
            "trigger_start",
        ):
            _require_int(skill.get(field), f"skills[{skill_id}].{field}")
        skill_name_id = _require_int(skill.get("name_id"), f"skills[{skill_id}].name_id")
        if skill_name_id not in name_ids | {0}:
            raise ValueError(f"skills[{skill_id}].name_id references {skill_name_id}")
        for index, area in enumerate(_require_list(skill.get("areas"), f"skills[{skill_id}].areas")):
            _require_int(area, f"skills[{skill_id}].areas[{index}]")
        triggers = _require_list(skill.get("triggers"), f"skills[{skill_id}].triggers")
        for index, trigger_value in enumerate(triggers):
            trigger_path = f"skills[{skill_id}].triggers[{index}]"
            trigger = _require_mapping(trigger_value, trigger_path)
            _require_exact_keys(trigger, {"type", "subject"}, trigger_path)
            _require_int(trigger.get("type"), f"{trigger_path}.type")
            condition_count += _validate_target(
                trigger.get("subject"),
                path=f"{trigger_path}.subject",
                name_ids=name_ids,
                name_sets=name_index,
                target_types=target_types,
            )
            trigger_count += 1
        validate_effects(skill.get("effects"), f"skills[{skill_id}].effects")

    for attack_id, attack in attacks.items():
        _require_exact_keys(
            attack,
            {
                "id", "card_id", "damage", "flags", "last_cancel_fail_attack",
                "name_id", "energies", "pre_effects", "post_effects",
            },
            f"attacks[{attack_id}]",
        )
        card_id = _require_int(attack.get("card_id"), f"attacks[{attack_id}].card_id")
        if card_id not in cards:
            raise ValueError(f"attacks[{attack_id}] references card {card_id}")
        for field in ("damage", "flags", "last_cancel_fail_attack"):
            _require_int(attack.get(field), f"attacks[{attack_id}].{field}")
        attack_name_id = _require_int(attack.get("name_id"), f"attacks[{attack_id}].name_id")
        if attack_name_id not in name_ids | {0}:
            raise ValueError(f"attacks[{attack_id}].name_id references {attack_name_id}")
        for index, energy in enumerate(
            _require_list(attack.get("energies"), f"attacks[{attack_id}].energies")
        ):
            _require_int(energy, f"attacks[{attack_id}].energies[{index}]")
        validate_effects(attack.get("pre_effects"), f"attacks[{attack_id}].pre_effects")
        validate_effects(attack.get("post_effects"), f"attacks[{attack_id}].post_effects")

    continuations = _require_list(root.get("continuations"), "continuations")
    parsed_continuations: list[int] = []
    continuation_symbols: list[str] = []
    for index, value in enumerate(continuations):
        row = _require_mapping(value, f"continuations[{index}]")
        _require_exact_keys(row, {"id", "symbol"}, f"continuations[{index}]")
        parsed_continuations.append(
            _require_int(row.get("id"), f"continuations[{index}].id")
        )
        continuation_symbols.append(
            _require_str(row.get("symbol"), f"continuations[{index}].symbol")
        )
    expected_continuations = list(range(len(continuations)))
    if parsed_continuations != expected_continuations:
        raise ValueError("continuation IDs must be dense and zero-based")
    if len(set(continuation_symbols)) != len(continuation_symbols):
        raise ValueError("continuation symbols must be unique")
    recorded_continuations = _require_int(counts.get("continuations"), "counts.continuations")
    if recorded_continuations != len(continuations):
        raise ValueError(
            f"counts.continuations={recorded_continuations}, actual={len(continuations)}"
        )

    return OfficialIRSummary(
        cards=len(cards),
        skills=len(skills),
        attacks=len(attacks),
        continuations=len(continuations),
        name_sets=len(name_index),
        effects=effect_count,
        triggers=trigger_count,
        target_conditions=condition_count,
        effect_types_used=tuple(sorted(effect_types)),
        target_types_used=tuple(sorted(target_types)),
        select_contexts_used=tuple(sorted(select_contexts)),
        unique_effect_signatures=len(effect_signatures),
        canonical_sha256=canonical_sha256(root),
    )


def load_and_validate_official_ir(path: str | Path) -> tuple[dict[str, Any], OfficialIRSummary]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload, validate_official_ir(payload)

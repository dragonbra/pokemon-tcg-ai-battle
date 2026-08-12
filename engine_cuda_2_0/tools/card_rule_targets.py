from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pure_policy_codec_v1 import (
    OWNER_OPP,
    ZONE_OPP_ACTIVE,
    ZONE_OPP_BENCH,
    ZONE_OPP_ENERGY,
    ZONE_OPP_EVOLUTION,
    ZONE_OPP_TOOL,
    ZONE_OWN_ACTIVE,
    ZONE_OWN_BENCH,
    ZONE_OWN_ENERGY,
    ZONE_OWN_EVOLUTION,
    ZONE_OWN_TOOL,
    ZONE_STADIUM,
)


DEFAULT_RULE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "card_rule_semantics_v1.json"
BLOCK_REASON_NAMES = (
    "attack_effect_shield",
    "normal_damage_shield",
    "damage_counter_shield",
    "damage_reduction",
    "attached_modifier",
    "boardwide_ability",
)
DAMAGE_BUCKET_EDGES = (0, 10, 20, 30, 40, 60, 80, 100, 130, 160, 200, 250, 300)
IN_PLAY_ZONES = {
    ZONE_OWN_ACTIVE,
    ZONE_OWN_BENCH,
    ZONE_OPP_ACTIVE,
    ZONE_OPP_BENCH,
    ZONE_OWN_ENERGY,
    ZONE_OPP_ENERGY,
    ZONE_OWN_TOOL,
    ZONE_OPP_TOOL,
    ZONE_OWN_EVOLUTION,
    ZONE_OPP_EVOLUTION,
    ZONE_STADIUM,
}


@dataclass(frozen=True)
class AttackRelationTarget:
    valid: float
    effective: float
    blocked_reasons: tuple[float, ...]
    damage_bucket: int
    damage_bucket_weight: float
    unblocked_channels: tuple[str, ...]
    blocked_channels: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "effective": self.effective,
            "blocked_reasons": list(self.blocked_reasons),
            "damage_bucket": self.damage_bucket,
            "damage_bucket_weight": self.damage_bucket_weight,
            "unblocked_channels": list(self.unblocked_channels),
            "blocked_channels": list(self.blocked_channels),
        }


class CardRuleTargetIndex:
    """Deterministic labels from visible structured rules, never policy actions."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.card_atoms = {
            int(card_id): frozenset(str(atom) for atom in atoms)
            for card_id, atoms in dict(payload.get("cards") or {}).items()
        }
        self.attack_atoms = {
            int(attack_id): frozenset(str(atom) for atom in atoms)
            for attack_id, atoms in dict(payload.get("attacks") or {}).items()
        }
        self.attack_profiles = {
            int(attack_id): dict(profile)
            for attack_id, profile in dict(payload.get("attack_profiles") or {}).items()
        }
        self.protection_rules = {
            int(card_id): tuple(dict(rule) for rule in rules)
            for card_id, rules in dict(payload.get("protection_rules") or {}).items()
        }
        self.damage_modifier_rules = {
            int(card_id): tuple(dict(rule) for rule in rules)
            for card_id, rules in dict(payload.get("damage_modifier_rules") or {}).items()
        }
        self.vocabulary = tuple(str(value) for value in payload.get("vocabulary") or ())
        self.vocabulary_index = {value: index for index, value in enumerate(self.vocabulary)}

    @classmethod
    def from_path(cls, path: str | Path = DEFAULT_RULE_CONFIG) -> "CardRuleTargetIndex":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def option_atom_target(self, option_cat: list[int]) -> tuple[list[float], float]:
        source_card = int(option_cat[4])
        target_card = int(option_cat[5])
        attack_id = int(option_cat[6])
        atoms = set(self.card_atoms.get(source_card, ()))
        atoms.update(self.card_atoms.get(target_card, ()))
        atoms.update(self.attack_atoms.get(attack_id, ()))
        target = [0.0] * len(self.vocabulary)
        for atom in atoms:
            index = self.vocabulary_index.get(atom)
            if index is not None:
                target[index] = 1.0
        return target, float(bool(atoms))

    def _entity_atoms(
        self,
        entity_index: int,
        entity_cat: list[list[int]],
        entity_parent: list[int],
    ) -> set[str]:
        if not 0 <= entity_index < len(entity_cat):
            return set()
        atoms = set(self.card_atoms.get(int(entity_cat[entity_index][0]), ()))
        for child_index, parent in enumerate(entity_parent):
            if int(parent) == entity_index and 0 <= child_index < len(entity_cat):
                child_atoms = self.card_atoms.get(int(entity_cat[child_index][0]), ())
                atoms.update(f"attached:{atom}" for atom in child_atoms)
        return atoms

    @staticmethod
    def _predicate_matches(predicate: str, atoms: set[str], *, attached: bool = False) -> bool:
        prefix = "attached:" if attached else ""
        aliases = {
            "has_ability": "trait:has_ability",
            "has_special_energy": "attached:card_type:special_energy",
        }
        expected = aliases.get(predicate, predicate)
        if predicate.startswith("type:"):
            expected = f"pokemon_type:{predicate.split(':', 1)[1]}"
        elif predicate.startswith("type_any:"):
            return any(
                f"pokemon_type:{name}" in atoms
                for name in predicate.split(":", 1)[1].split(",")
            )
        elif predicate.startswith("attached_energy:"):
            expected = f"attached:energy_type:{predicate.split(':', 1)[1]}"
        return f"{prefix}{expected}" in atoms if attached else expected in atoms

    def _damage_modifier(
        self,
        attacker: int,
        target: int,
        entity_cat: list[list[int]],
        entity_parent: list[int],
    ) -> tuple[int, bool]:
        attacker_atoms = self._entity_atoms(attacker, entity_cat, entity_parent)
        target_atoms = self._entity_atoms(target, entity_cat, entity_parent)
        bonus = 0
        ambiguous = False
        for source_index, row in enumerate(entity_cat):
            for rule in self.damage_modifier_rules.get(int(row[0]), ()):
                scope = str(rule.get("scope") or "source")
                if scope == "attached_pokemon" and int(entity_parent[source_index]) != attacker:
                    continue
                if scope == "source" and source_index != attacker:
                    continue
                if bool(rule.get("dynamic")):
                    ambiguous = True
                    continue
                if not all(
                    self._predicate_matches(str(value), attacker_atoms)
                    for value in rule.get("source_predicates") or ()
                ):
                    continue
                if not all(
                    self._predicate_matches(str(value), target_atoms)
                    for value in rule.get("target_predicates") or ()
                ):
                    continue
                bonus += int(rule.get("amount") or 0)
        return bonus, ambiguous

    def _rule_applies(
        self,
        rule: dict[str, Any],
        protection_source: int,
        attacker: int,
        target: int,
        entity_cat: list[list[int]],
        entity_parent: list[int],
    ) -> tuple[bool, bool]:
        source_row = entity_cat[protection_source]
        target_row = entity_cat[target]
        scope = str(rule.get("scope") or "self")
        ambiguous = False
        if scope == "attached_pokemon":
            applies = int(entity_parent[protection_source]) == target
        elif scope in {"own_basic_team_rocket", "own_all"}:
            applies = int(source_row[1]) == int(target_row[1])
        elif scope == "both_all":
            applies = True
        elif scope == "bench":
            applies = int(target_row[2]) in {ZONE_OWN_BENCH, ZONE_OPP_BENCH}
        elif scope == "self":
            applies = protection_source == target
        elif scope == "chosen_own_pokemon":
            applies = False
            ambiguous = int(source_row[1]) == int(target_row[1])
        else:
            applies = False
            ambiguous = True
        if not applies:
            return False, ambiguous
        attacker_atoms = self._entity_atoms(attacker, entity_cat, entity_parent)
        target_atoms = self._entity_atoms(target, entity_cat, entity_parent)
        for predicate in rule.get("source_predicates") or ():
            if not self._predicate_matches(str(predicate), attacker_atoms):
                return False, ambiguous
        for predicate in rule.get("target_predicates") or ():
            if not self._predicate_matches(str(predicate), target_atoms):
                return False, ambiguous
        if bool(rule.get("coin_flip")) or int(rule.get("threshold") or 0) > 0:
            ambiguous = True
        return True, ambiguous

    @staticmethod
    def _fallback_entity(
        entity_cat: list[list[int]], owner: int, zone: int
    ) -> int:
        return next(
            (
                index
                for index, row in enumerate(entity_cat)
                if int(row[1]) == owner and int(row[2]) == zone
            ),
            -1,
        )

    @staticmethod
    def _damage_bucket(value: int) -> int:
        for index, edge in enumerate(DAMAGE_BUCKET_EDGES):
            if value <= edge:
                return index
        return len(DAMAGE_BUCKET_EDGES)

    def classify_attack_option(
        self,
        entity_cat: list[list[int]],
        entity_parent: list[int],
        option_cat: list[int],
    ) -> AttackRelationTarget:
        attack_id = int(option_cat[6])
        profile = self.attack_profiles.get(attack_id)
        if attack_id <= 0 or profile is None:
            return AttackRelationTarget(0.0, 0.0, (0.0,) * len(BLOCK_REASON_NAMES), 0, 0.0, (), ())
        source = int(option_cat[8]) - 1
        target = int(option_cat[9]) - 1
        if source < 0:
            source = self._fallback_entity(entity_cat, 1, ZONE_OWN_ACTIVE)
        if target < 0:
            target = self._fallback_entity(entity_cat, OWNER_OPP, ZONE_OPP_ACTIVE)
        if not (0 <= source < len(entity_cat) and 0 <= target < len(entity_cat)):
            return AttackRelationTarget(0.0, 0.0, (0.0,) * len(BLOCK_REASON_NAMES), 0, 0.0, (), ())

        delivered = set(str(value) for value in profile.get("delivery_channels") or ())
        meaningful = delivered & {"normal_damage", "attack_effect", "damage_counter", "non_damage_effect"}
        blocked: set[str] = set()
        reduction = 0
        reasons = [0.0] * len(BLOCK_REASON_NAMES)
        ambiguous = False
        for protection_source, row in enumerate(entity_cat):
            if int(row[2]) not in IN_PLAY_ZONES:
                continue
            for rule in self.protection_rules.get(int(row[0]), ()):
                applies, rule_ambiguous = self._rule_applies(
                    rule,
                    protection_source,
                    source,
                    target,
                    entity_cat,
                    entity_parent,
                )
                ambiguous |= rule_ambiguous
                if not applies:
                    continue
                channels = set(str(value) for value in rule.get("channels") or ())
                affected = meaningful & channels
                if not affected:
                    continue
                if str(rule.get("mode")) == "reduce":
                    reduction += int(rule.get("amount") or 0)
                    reasons[BLOCK_REASON_NAMES.index("damage_reduction")] = 1.0
                else:
                    blocked.update(affected)
                    if "attack_effect" in affected and "damage_counter" in meaningful:
                        blocked.add("damage_counter")
                for channel, reason in (
                    ("attack_effect", "attack_effect_shield"),
                    ("normal_damage", "normal_damage_shield"),
                    ("damage_counter", "damage_counter_shield"),
                ):
                    if channel in affected:
                        reasons[BLOCK_REASON_NAMES.index(reason)] = 1.0
                if str(rule.get("scope")) == "attached_pokemon":
                    reasons[BLOCK_REASON_NAMES.index("attached_modifier")] = 1.0
                elif str(rule.get("scope")) in {"own_basic_team_rocket", "own_all", "both_all"}:
                    reasons[BLOCK_REASON_NAMES.index("boardwide_ability")] = 1.0

        damage_bonus, modifier_ambiguous = self._damage_modifier(
            source, target, entity_cat, entity_parent
        )
        ambiguous |= modifier_ambiguous
        normal_damage = max(0, int(profile.get("base_damage") or 0) + damage_bonus)
        attacker_atoms = self._entity_atoms(source, entity_cat, entity_parent)
        target_atoms = self._entity_atoms(target, entity_cat, entity_parent)
        attacker_types = {
            atom.split(":", 1)[1]
            for atom in attacker_atoms
            if atom.startswith("pokemon_type:")
        }
        if any(f"weakness:type:{name}" in target_atoms for name in attacker_types):
            normal_damage *= 2
        if any(f"resistance:type:{name}" in target_atoms for name in attacker_types):
            normal_damage = max(0, normal_damage - 30)
        normal_damage = max(0, normal_damage - reduction)
        if "normal_damage" in blocked:
            normal_damage = 0
        counter_damage = int(profile.get("base_damage_counters") or 0) * 10
        if "damage_counter" in blocked or "attack_effect" in blocked:
            counter_damage = 0
        unblocked = meaningful - blocked
        if "normal_damage" in unblocked and normal_damage <= 0:
            unblocked.remove("normal_damage")
        if "damage_counter" in unblocked and counter_damage <= 0:
            unblocked.remove("damage_counter")
        effective = bool(unblocked)
        scales = bool(profile.get("scales_with"))
        damage_weight = float(not ambiguous and not scales and effective)
        return AttackRelationTarget(
            valid=float(not ambiguous),
            effective=float(effective),
            blocked_reasons=tuple(reasons),
            damage_bucket=self._damage_bucket(normal_damage + counter_damage),
            damage_bucket_weight=damage_weight,
            unblocked_channels=tuple(sorted(unblocked)),
            blocked_channels=tuple(sorted(blocked)),
        )

    def classify_options(
        self,
        entity_cat: list[list[int]],
        entity_parent: list[int],
        option_cat: Iterable[list[int]],
    ) -> list[dict[str, Any]]:
        return [
            self.classify_attack_option(entity_cat, entity_parent, option).as_dict()
            for option in option_cat
        ]

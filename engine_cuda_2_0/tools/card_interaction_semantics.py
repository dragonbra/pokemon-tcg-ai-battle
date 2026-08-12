from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from card_rule_targets import CardRuleTargetIndex
from pure_policy_codec_v1 import ENTITY_CAT_DIM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERACTION_CONFIG = ROOT / "configs" / "card_rule_semantics_v1.json"
LEGACY_INTERACTION_CONFIG = ROOT / "configs" / "card_interaction_semantics_v1.json"

ATTACK_CHANNEL_NAMES = (
    "normal_damage",
    "attack_effect",
    "damage_counter",
    "non_damage_effect",
)
ATTACK_CHANNEL_BITS = {
    name: 1 << index for index, name in enumerate(ATTACK_CHANNEL_NAMES)
}

PROTECTION_SCOPE_NAMES = (
    "none",
    "attached_pokemon",
    "bench",
    "both_all",
    "chosen_own_pokemon",
    "own_all",
    "own_basic_team_rocket",
    "self",
)
PROTECTION_SCOPE_IDS = {
    name: index for index, name in enumerate(PROTECTION_SCOPE_NAMES)
}
PROTECTION_MODE_NONE = 0
PROTECTION_MODE_BLOCK = 1
PROTECTION_MODE_REDUCE = 2

# The full generated rule file currently uses these predicates for persistent
# protection. Keeping a stable vocabulary makes the compiled buffers small and
# checkpoint-compatible while preserving the exact structured predicates.
INTERACTION_PREDICATE_NAMES = (
    "attached_energy:metal",
    "attached_energy:water",
    "has_ability",
    "has_special_energy",
    "rulebox:none",
    "rulebox:tera",
    "stage:basic",
    "trait:steven",
    "trait:team_rocket",
    "type:colorless",
    "type:dragon",
    "type:fighting",
    "type:metal",
    "type:psychic",
    "type_any:fire,grass,lightning,water",
    "type_any:fire,water",
    "zone:bench",
)
INTERACTION_PREDICATE_BITS = {
    name: 1 << index for index, name in enumerate(INTERACTION_PREDICATE_NAMES)
}
ATTACHED_PREDICATE_NAMES = (
    "attached_energy:metal",
    "attached_energy:water",
    "has_special_energy",
)
ATTACHED_PREDICATE_MASK = sum(
    INTERACTION_PREDICATE_BITS[name] for name in ATTACHED_PREDICATE_NAMES
)
ZONE_BENCH_PREDICATE_BIT = INTERACTION_PREDICATE_BITS["zone:bench"]

RELATION_FEATURE_NAMES = (
    "attack_normal_damage",
    "attack_effect",
    "attack_damage_counter",
    "attack_non_damage_effect",
    "protection_source_visible",
    "protection_scope_matches",
    "protection_source_predicates_match",
    "protection_target_predicates_match",
    "normal_damage_blocked",
    "attack_effect_blocked",
    "damage_counter_blocked",
    "normal_damage_reduced",
    "attached_protection_applies",
    "boardwide_protection_applies",
    "self_protection_applies",
    "ambiguous_protection_applies",
    "any_effective_channel",
    "all_meaningful_channels_blocked",
    "normal_damage_passes_effect_shield",
    "grants_future_block",
    "grants_future_reduction",
)

INTERACTION_GATE_GROUP_FEATURES = {
    "effectiveness": (
        "attack_normal_damage",
        "attack_effect",
        "attack_damage_counter",
        "attack_non_damage_effect",
        "any_effective_channel",
    ),
    "block_immunity": (
        "protection_source_visible",
        "protection_scope_matches",
        "protection_source_predicates_match",
        "protection_target_predicates_match",
        "normal_damage_blocked",
        "attack_effect_blocked",
        "damage_counter_blocked",
        "normal_damage_reduced",
        "attached_protection_applies",
        "boardwide_protection_applies",
        "self_protection_applies",
        "ambiguous_protection_applies",
        "all_meaningful_channels_blocked",
        "normal_damage_passes_effect_shield",
    ),
    "future_protection": (
        "grants_future_block",
        "grants_future_reduction",
    ),
}


@dataclass(frozen=True)
class CompiledInteractionIndex:
    card_predicates: torch.Tensor
    card_protection_channels: torch.Tensor
    card_protection_scope: torch.Tensor
    card_protection_mode: torch.Tensor
    card_protection_amount: torch.Tensor
    card_protection_source_predicates: torch.Tensor
    card_protection_target_predicates: torch.Tensor
    card_protection_ambiguous: torch.Tensor
    attack_channels: torch.Tensor
    attack_future_mode: torch.Tensor
    metadata: dict[str, Any]

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in (
                "card_predicates",
                "card_protection_channels",
                "card_protection_scope",
                "card_protection_mode",
                "card_protection_amount",
                "card_protection_source_predicates",
                "card_protection_target_predicates",
                "card_protection_ambiguous",
                "attack_channels",
                "attack_future_mode",
            )
        }


@lru_cache(maxsize=8)
def load_interaction_payload(
    path: str | Path = DEFAULT_INTERACTION_CONFIG,
) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _predicate_mask(predicates: list[str] | tuple[str, ...]) -> int:
    unknown = sorted(
        str(value)
        for value in predicates
        if str(value) not in INTERACTION_PREDICATE_BITS
    )
    if unknown:
        raise ValueError(f"unsupported interaction predicates: {unknown}")
    return sum(INTERACTION_PREDICATE_BITS[str(value)] for value in predicates)


def _card_predicate_mask(atoms: list[str] | tuple[str, ...]) -> int:
    values = {str(value) for value in atoms}
    mask = 0
    for predicate, bit in INTERACTION_PREDICATE_BITS.items():
        matched = False
        if predicate.startswith("type:"):
            matched = f"pokemon_type:{predicate.split(':', 1)[1]}" in values
        elif predicate.startswith("type_any:"):
            matched = any(
                f"pokemon_type:{name}" in values
                for name in predicate.split(":", 1)[1].split(",")
            )
        elif predicate == "has_ability":
            matched = "trait:has_ability" in values
        elif predicate == "has_special_energy":
            matched = "card_type:special_energy" in values
        elif predicate.startswith("attached_energy:"):
            matched = f"energy_type:{predicate.split(':', 1)[1]}" in values
        elif predicate != "zone:bench":
            matched = predicate in values
        if matched:
            mask |= bit
    return mask


def _legacy_to_structured(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the original seven-card fixture into the structured schema."""
    cards = dict(payload.get("cards") or {})
    attacks = dict(payload.get("attacks") or {})
    structured_cards: dict[str, list[str]] = {}
    protection_rules: dict[str, list[dict[str, Any]]] = {}
    for card_id, raw_atoms in cards.items():
        atoms = {str(value) for value in raw_atoms}
        structured = [
            atom
            for atom in atoms
            if atom.startswith("stage:") or atom.startswith("trait:")
        ]
        structured_cards[str(card_id)] = structured
        scope = None
        target_predicates: list[str] = []
        if "protects:self_from_opponent_attack_effect" in atoms:
            scope = "self"
        elif "protects:own_basic_team_rocket_from_opponent_attack_effect" in atoms:
            scope = "own_basic_team_rocket"
            target_predicates = ["stage:basic", "trait:team_rocket"]
        if scope is not None:
            protection_rules[str(card_id)] = [
                {
                    "channels": ["attack_effect"],
                    "scope": scope,
                    "source_predicates": [],
                    "target_predicates": target_predicates,
                    "mode": "block",
                    "amount": 0,
                    "threshold": 0,
                    "coin_flip": False,
                    "duration": "continuous",
                }
            ]
    attack_profiles: dict[str, dict[str, Any]] = {}
    for attack_id, raw_atoms in attacks.items():
        atoms = {str(value) for value in raw_atoms}
        channels = []
        if "delivery:normal_damage" in atoms:
            channels.append("normal_damage")
        if "source:attack_effect" in atoms:
            channels.append("attack_effect")
        if "delivery:damage_counter" in atoms:
            channels.append("damage_counter")
        attack_profiles[str(attack_id)] = {
            "base_damage": 0,
            "base_damage_counters": 0,
            "delivery_channels": channels,
            "scales_with": [],
            "targeting": [],
        }
    return {
        "version": str(payload.get("version", "")),
        "cards": structured_cards,
        "attacks": attacks,
        "attack_profiles": attack_profiles,
        "protection_rules": protection_rules,
        "attack_protection_rules": {},
    }


@lru_cache(maxsize=16)
def load_compiled_interaction_index(
    path: str | Path,
    *,
    max_card_id: int,
    max_attack_id: int,
) -> CompiledInteractionIndex:
    resolved = Path(path).resolve()
    source_payload = load_interaction_payload(resolved)
    payload = (
        source_payload
        if "attack_profiles" in source_payload
        else _legacy_to_structured(source_payload)
    )
    card_predicates = torch.zeros(max_card_id + 1, dtype=torch.long)
    card_channels = torch.zeros(max_card_id + 1, dtype=torch.long)
    card_scope = torch.zeros(max_card_id + 1, dtype=torch.long)
    card_mode = torch.zeros(max_card_id + 1, dtype=torch.long)
    card_amount = torch.zeros(max_card_id + 1, dtype=torch.float32)
    card_source_predicates = torch.zeros(max_card_id + 1, dtype=torch.long)
    card_target_predicates = torch.zeros(max_card_id + 1, dtype=torch.long)
    card_ambiguous = torch.zeros(max_card_id + 1, dtype=torch.bool)
    attack_channels = torch.zeros(max_attack_id + 1, dtype=torch.long)
    attack_future_mode = torch.zeros(max_attack_id + 1, dtype=torch.long)

    cards = dict(payload.get("cards") or {})
    for raw_id, atoms in cards.items():
        card_id = int(raw_id)
        if 0 <= card_id <= max_card_id:
            card_predicates[card_id] = _card_predicate_mask(list(atoms or ()))

    protection_rules = dict(payload.get("protection_rules") or {})
    for raw_id, rules in protection_rules.items():
        card_id = int(raw_id)
        if not 0 <= card_id <= max_card_id or not rules:
            continue
        if len(rules) != 1:
            raise ValueError(f"card {card_id} has multiple protection rules")
        rule = dict(rules[0])
        card_channels[card_id] = sum(
            ATTACK_CHANNEL_BITS[str(channel)]
            for channel in rule.get("channels") or ()
            if str(channel) in ATTACK_CHANNEL_BITS
        )
        scope = str(rule.get("scope") or "none")
        if scope not in PROTECTION_SCOPE_IDS:
            raise ValueError(f"unsupported protection scope: {scope}")
        card_scope[card_id] = PROTECTION_SCOPE_IDS[scope]
        mode = str(rule.get("mode") or "")
        card_mode[card_id] = (
            PROTECTION_MODE_BLOCK
            if mode == "block"
            else PROTECTION_MODE_REDUCE
            if mode == "reduce"
            else PROTECTION_MODE_NONE
        )
        card_amount[card_id] = float(rule.get("amount") or 0)
        card_source_predicates[card_id] = _predicate_mask(
            list(rule.get("source_predicates") or ())
        )
        card_target_predicates[card_id] = _predicate_mask(
            list(rule.get("target_predicates") or ())
        )
        card_ambiguous[card_id] = bool(
            rule.get("coin_flip")
            or int(rule.get("threshold") or 0) > 0
            or scope == "chosen_own_pokemon"
        )

    profiles = dict(payload.get("attack_profiles") or {})
    for raw_id, profile in profiles.items():
        attack_id = int(raw_id)
        if 0 <= attack_id <= max_attack_id:
            attack_channels[attack_id] = sum(
                ATTACK_CHANNEL_BITS[str(channel)]
                for channel in profile.get("delivery_channels") or ()
                if str(channel) in ATTACK_CHANNEL_BITS
            )

    attack_protection = dict(payload.get("attack_protection_rules") or {})
    for raw_id, rules in attack_protection.items():
        attack_id = int(raw_id)
        if not 0 <= attack_id <= max_attack_id or not rules:
            continue
        modes = {str(rule.get("mode") or "") for rule in rules}
        attack_future_mode[attack_id] = (
            PROTECTION_MODE_BLOCK
            if "block" in modes
            else PROTECTION_MODE_REDUCE
            if "reduce" in modes
            else PROTECTION_MODE_NONE
        )

    metadata = {
        "version": str(payload.get("version", "")),
        "source_path": str(resolved),
        "runtime_version": 2,
        "card_count": len(cards),
        "attack_count": len(profiles),
        "protection_rule_count": sum(len(rules) for rules in protection_rules.values()),
        "attack_protection_rule_count": sum(
            len(rules) for rules in attack_protection.values()
        ),
        "predicate_names": list(INTERACTION_PREDICATE_NAMES),
        "relation_feature_names": list(RELATION_FEATURE_NAMES),
        "relation_feature_count": len(RELATION_FEATURE_NAMES),
    }
    return CompiledInteractionIndex(
        card_predicates=card_predicates,
        card_protection_channels=card_channels,
        card_protection_scope=card_scope,
        card_protection_mode=card_mode,
        card_protection_amount=card_amount,
        card_protection_source_predicates=card_source_predicates,
        card_protection_target_predicates=card_target_predicates,
        card_protection_ambiguous=card_ambiguous,
        attack_channels=attack_channels,
        attack_future_mode=attack_future_mode,
        metadata=metadata,
    )


def _entity_rows(value: Any) -> list[list[int]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        return [list(row) for row in value]
    return [
        list(value[offset : offset + ENTITY_CAT_DIM])
        for offset in range(0, len(value), ENTITY_CAT_DIM)
        if len(value[offset : offset + ENTITY_CAT_DIM]) == ENTITY_CAT_DIM
    ]


@lru_cache(maxsize=8)
def _target_index(path: str) -> CardRuleTargetIndex:
    return CardRuleTargetIndex.from_path(path)


def classify_attack_interaction(
    entity_cat: Any,
    option_cat: list[int],
    *,
    entity_parent: Any = None,
    config_path: str | Path = DEFAULT_INTERACTION_CONFIG,
) -> dict[str, float]:
    """Return a rule-derived effectiveness label for one encoded attack.

    This is auxiliary supervision, never reward. Ambiguous protections and
    unknown attacks receive zero weight instead of a guessed target.
    """
    if len(option_cat) < 12:
        return {"target": 0.0, "weight": 0.0, "blocked": 0.0}
    rows = _entity_rows(entity_cat)
    if hasattr(entity_parent, "tolist"):
        entity_parent = entity_parent.tolist()
    parents = [int(value) for value in (entity_parent or [-1] * len(rows))]
    if len(parents) != len(rows):
        parents = [-1] * len(rows)
    resolved = str(Path(config_path).resolve())
    payload = load_interaction_payload(resolved)
    if "attack_profiles" not in payload:
        # Legacy fixtures cannot express attachment or generic predicates.
        payload = _legacy_to_structured(payload)
        index = CardRuleTargetIndex(payload)
    else:
        index = _target_index(resolved)
    target = index.classify_attack_option(rows, parents, option_cat)
    return {
        "target": float(target.effective),
        "weight": float(target.valid),
        "blocked": float(target.valid > 0.0 and target.effective <= 0.0),
    }

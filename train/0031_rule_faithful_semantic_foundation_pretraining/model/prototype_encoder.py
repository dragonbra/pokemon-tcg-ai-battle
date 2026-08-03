"""Rule-faithful neural representation of official finite prototype facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..domain.prototypes import PrototypeIndex
from .config import ModelConfig
from .typed_fields import CategoricalFields, NumericFields


CARD_BOOLEANS = (
    "can_play_first_turn", "can_trash", "transform_only", "trash_my_turn_end",
    "cannot_to_hand_or_deck_in_trash", "tera", "to_bench", "to_battle_field_only_setup",
    "to_active_only_setup", "no_prize", "only_team_rocket", "ancient", "future", "hop",
    "lillie", "iono", "n", "ethan", "cynthia", "misty", "arven", "steven", "marnie",
    "erika", "larry", "team_rocket", "ace_spec", "can_use",
)
SKILL_BOOLEANS = (
    "main_ability", "once_turn", "select_activation", "not_stack", "activate_in_discard",
    "attach_bench", "ko_self", "lucky_bonus",
)
EFFECT_BOOLEANS = (
    "is_condition", "enemy_select", "random_select", "each_selected", "each_list",
    "add_check_list", "keep_selected_list", "exclude_previous_target", "keep_target_list",
    "multiply_previous_target_count", "multiply_coin_heads", "can_select_zero",
    "can_select_zero_if_previous", "cannot_select_zero", "energy_max_select",
    "select_target_count", "select_coin_head_count", "select_coin_head_count_x2",
    "select_enemy_energy_count", "skip_no_target", "open", "switch_bench_as_target",
    "active_effect_target", "bench_effect_target", "remove_if_no_effect", "seeing_deck",
    "separator", "fail_skip",
)


@dataclass(frozen=True, slots=True)
class PrototypeEmbeddings:
    cards: Tensor
    attacks: Tensor
    skills: Tensor
    effects: Tensor

    def card(self, identity: Tensor) -> Tensor:
        return self.cards[identity]

    def attack(self, identity: Tensor) -> Tensor:
        return self.attacks[identity]

    def skill(self, identity: Tensor) -> Tensor:
        return self.skills[identity]

    def effect(self, identity: Tensor) -> Tensor:
        return self.effects[identity]


def _symbols(prototypes: PrototypeIndex) -> dict[str, int]:
    values = {""}
    for card in prototypes.engine_cards.values():
        values.update(str(card.get(name, "")) for name in ("code", "evolves_from", "evolves_from_2", "name", "name_en"))
    for collection in (prototypes.engine_attacks.values(), prototypes.skills.values()):
        for item in collection:
            values.update(str(item.get(name, "")) for name in ("name", "name_en"))
    for effect in prototypes.effects.values():
        for condition in effect.get("target", {}).get("conditions", ()):
            values.add(str(condition.get("name", "")))
    return {value: index + 1 for index, value in enumerate(sorted(values)) if value}


def _target_cat(target: Mapping[str, Any], symbols: Mapping[str, int]) -> list[int]:
    result = [
        int(target.get("target_player", 0)) + 1,
        int(bool(target.get("not_me"))) + 1,
        int(bool(target.get("skip_enemy_target"))) + 1,
    ]
    areas = list(target.get("areas", ()))
    result.extend([int(areas[i]) + 1 if i < len(areas) else 0 for i in range(3)])
    conditions = list(target.get("conditions", ()))
    for index in range(3):
        condition = conditions[index] if index < len(conditions) else None
        result.extend([
            int(condition.get("target_type", 0)) + 1 if condition else 0,
            int(condition.get("comparator_type", 0)) + 1 if condition else 0,
            symbols.get(str(condition.get("name", "")), 0) if condition else 0,
        ])
    return result


def _target_num(target: Mapping[str, Any]) -> list[float]:
    conditions = list(target.get("conditions", ()))
    result: list[float] = []
    for index in range(3):
        condition = conditions[index] if index < len(conditions) else None
        result.extend([
            float(condition.get("value", 0)) if condition else 0.0,
            float(condition.get("value2", 0)) if condition else 0.0,
        ])
    return result


class OfficialPrototypeEncoder(nn.Module):
    """Encode exact engine fields; no public compressed semantic field is consulted."""

    def __init__(self, config: ModelConfig, prototypes: PrototypeIndex):
        super().__init__()
        self.config = config
        maxima = {
            "card": max(prototypes.engine_cards, default=0), "attack": max(prototypes.engine_attacks, default=0),
            "skill": max(prototypes.skills, default=0), "effect": max(prototypes.effects, default=0),
        }
        limits = {"card": config.max_card_id, "attack": config.max_attack_id,
                  "skill": config.max_skill_id, "effect": config.max_effect_id}
        exceeded = {name: (value, limits[name]) for name, value in maxima.items() if value > limits[name]}
        if exceeded:
            raise ValueError(f"prototype identities exceed model limits: {exceeded}")

        symbols = _symbols(prototypes)
        symbol_vocab = max(symbols.values(), default=0) + 1
        self.card_cat_vocabs = (8, 6, 5, 1025, 1025, 1025, symbol_vocab, symbol_vocab,
                                symbol_vocab, symbol_vocab, symbol_vocab,
                                *([3] * (len(CARD_BOOLEANS) + 27)))
        self.attack_cat_vocabs = (config.max_card_id + 1, *([3] * 20), *([513] * 5), 3)
        # card, type, two exact area slots, then eight booleans and two trigger structures.
        trigger_vocabs = (32, 5, 3, 3, 26, 26, 103, 7, symbol_vocab, 103, 7, symbol_vocab)
        self.skill_cat_vocabs = (
            config.max_card_id + 1, 18, 26, 26, *([3] * len(SKILL_BOOLEANS)),
            *trigger_vocabs, *trigger_vocabs,
        )
        target_vocabs = (5, 3, 3, 26, 26, 26, 103, 7, symbol_vocab, 103, 7, symbol_vocab, 103, 7, symbol_vocab)
        self.effect_cat_vocabs = (
            4, 4, 246, 34, 66, 130, 18, 130, config.max_skill_id + 1,
            *([3] * len(EFFECT_BOOLEANS)), *target_vocabs,
        )

        d = config.d_model
        self.card_identity = nn.Embedding(config.max_card_id + 1, d, padding_idx=0)
        self.attack_identity = nn.Embedding(config.max_attack_id + 1, d, padding_idx=0)
        self.skill_identity = nn.Embedding(config.max_skill_id + 1, d, padding_idx=0)
        self.effect_identity = nn.Embedding(config.max_effect_id + 1, d, padding_idx=0)
        self.card_cat = CategoricalFields(self.card_cat_vocabs, d)
        self.card_num = NumericFields(4, d)
        self.attack_cat = CategoricalFields(self.attack_cat_vocabs, d)
        self.attack_num = NumericFields(2, d)
        self.skill_cat = CategoricalFields(self.skill_cat_vocabs, d)
        self.skill_num = NumericFields(7 + 2 * 4, d)
        self.effect_cat = CategoricalFields(self.effect_cat_vocabs, d)
        self.effect_num = NumericFields(7 + 6, d)
        self.norm = nn.LayerNorm(d)
        for name, table in self._build_tables(config, prototypes, symbols).items():
            self.register_buffer(name, table, persistent=True)
        self._validate_categorical_tables()

    @staticmethod
    def _build_tables(config: ModelConfig, prototypes: PrototypeIndex,
                      symbols: Mapping[str, int]) -> dict[str, Tensor]:
        card_cat_rows: dict[int, list[int]] = {}
        card_num = torch.zeros(config.max_card_id + 1, 4)
        card_skills = torch.zeros(config.max_card_id + 1, 3, dtype=torch.long)
        card_attacks = torch.zeros(config.max_card_id + 1, 4, dtype=torch.long)
        for identity, card in prototypes.engine_cards.items():
            card_cat_rows[identity] = [
                int(card["card_type"]) + 1, int(card["pokemon_type"]) + 1,
                int(card["evolution_type"]) + 1, int(card["energy_type_mask"]) + 1,
                int(card["weakness"]) + 1, int(card["resistance"]) + 1,
                *[symbols.get(str(card.get(name, "")), 0) for name in
                  ("code", "evolves_from", "evolves_from_2", "name", "name_en")],
                *[int(bool(card[name])) + 1 for name in CARD_BOOLEANS],
                *[int(bool(int(card[mask_name]) & (1 << bit))) + 1
                  for mask_name in ("energy_type_mask", "weakness", "resistance") for bit in range(9)],
            ]
            card_num[identity] = torch.tensor([
                float(card["hp"]), float(card["retreat_cost"]),
                float(card["energy_count"]), float(card["number"]),
            ])
            card_skills[identity] = torch.tensor([
                int(card["ability_skill_id"]), int(card["play_skill_id"]), int(card["delay_skill_id"]),
            ])
            attacks = list(card["attack_ids"])
            card_attacks[identity, : min(4, len(attacks))] = torch.tensor(attacks[:4])

        attack_cat_rows: dict[int, list[int]] = {}
        attack_num = torch.zeros(config.max_attack_id + 1, 2)
        attack_effects = torch.zeros(config.max_attack_id + 1, 7, dtype=torch.long)
        for identity, attack in prototypes.engine_attacks.items():
            energies = list(attack["energies"])
            attack_cat_rows[identity] = [
                int(attack["card_id"]),
                *[int(bool(int(attack["attack_flags"]) & (1 << bit))) + 1 for bit in range(20)],
                *[int(energies[i]) + 1 if i < len(energies) else 0 for i in range(5)],
                int(bool(attack.get("last_cancel_fail_attack"))) + 1,
            ]
            attack_num[identity] = torch.tensor([float(attack["damage"]), float(attack["attack_flags"])])
            refs = list(prototypes.attack_effect_refs.get(identity, ()))
            attack_effects[identity, : min(7, len(refs))] = torch.tensor(refs[:7])

        def trigger_cat(trigger: Mapping[str, Any] | None) -> list[int]:
            if trigger is None:
                return [0] * 12
            subject = trigger.get("subject", {})
            areas = list(subject.get("areas", ()))
            conditions = list(subject.get("conditions", ()))
            result = [
                int(trigger.get("trigger_type", 0)) + 1, int(subject.get("target_player", 0)) + 1,
                int(bool(subject.get("not_me"))) + 1, int(bool(subject.get("skip_enemy_target"))) + 1,
                int(areas[0]) + 1 if areas else 0, int(areas[1]) + 1 if len(areas) > 1 else 0,
            ]
            for i in range(2):
                condition = conditions[i] if i < len(conditions) else None
                result.extend([
                    int(condition.get("target_type", 0)) + 1 if condition else 0,
                    int(condition.get("comparator_type", 0)) + 1 if condition else 0,
                    symbols.get(str(condition.get("name", "")), 0) if condition else 0,
                ])
            return result

        def trigger_num(trigger: Mapping[str, Any] | None) -> list[float]:
            conditions = list(trigger.get("subject", {}).get("conditions", ())) if trigger else []
            result: list[float] = []
            for i in range(2):
                condition = conditions[i] if i < len(conditions) else None
                result.extend([float(condition.get("value", 0)) if condition else 0.0,
                               float(condition.get("value2", 0)) if condition else 0.0])
            return result

        skill_cat_rows: dict[int, list[int]] = {}
        skill_num = torch.zeros(config.max_skill_id + 1, 15)
        skill_effects = torch.zeros(config.max_skill_id + 1, 11, dtype=torch.long)
        for identity, skill in prototypes.skills.items():
            areas, triggers = list(skill["areas"]), list(skill["triggers"])
            skill_cat_rows[identity] = [
                int(skill["card_id"]), int(skill["skill_type"]) + 1,
                int(areas[0]) + 1 if areas else 0, int(areas[1]) + 1 if len(areas) > 1 else 0,
                *[int(bool(skill[name])) + 1 for name in SKILL_BOOLEANS],
                *trigger_cat(triggers[0] if triggers else None),
                *trigger_cat(triggers[1] if len(triggers) > 1 else None),
            ]
            skill_num[identity] = torch.tensor([
                float(skill["priority"]), float(skill["first_condition_count"]),
                float(skill["second_effect_start"]), float(skill["second_effect_start_enemy"]),
                float(skill["trigger_start"]), float(len(skill["effects"])), float(len(triggers)),
                *trigger_num(triggers[0] if triggers else None),
                *trigger_num(triggers[1] if len(triggers) > 1 else None),
            ])
            refs = list(prototypes.skill_effect_refs.get(identity, ()))
            skill_effects[identity, : min(11, len(refs))] = torch.tensor(refs[:11])

        effect_cat_rows: dict[int, list[int]] = {}
        effect_num = torch.zeros(config.max_effect_id + 1, 13)
        for identity, effect in prototypes.effects.items():
            target = effect["target"]
            effect_cat_rows[identity] = [
                int(effect["parent_kind"]) + 1, int(effect["phase"]) + 1,
                int(effect["effect_type"]) + 1, int(effect["select_type"]) + 1,
                int(effect["select_context"]) + 1, int(effect["condition_type"]) + 1,
                int(effect["comparator_type"]) + 1, int(effect.get("condition_type", 0)) + 1,
                int(effect.get("linked_skill_id", 0)),
                *[int(bool(effect[name])) + 1 for name in EFFECT_BOOLEANS],
                *_target_cat(target, symbols),
            ]
            effect_num[identity] = torch.tensor([
                float(effect["ordinal"]), float(effect["parent_id"]), float(effect["select_count"]),
                float(effect["values"][0]), float(effect["values"][1]),
                float(effect["loop_count"]), float(effect["priority"]), *_target_num(target),
            ])

        def table(rows: Mapping[int, list[int]], count: int, width: int) -> Tensor:
            output = torch.zeros(count + 1, width, dtype=torch.long)
            for identity, values in rows.items():
                output[identity] = torch.tensor(values)
            return output

        return {
            "card_cat_table": table(card_cat_rows, config.max_card_id, 11 + len(CARD_BOOLEANS) + 27),
            "card_num_table": card_num, "card_skill_table": card_skills,
            "card_attack_table": card_attacks,
            "attack_cat_table": table(attack_cat_rows, config.max_attack_id, 27),
            "attack_num_table": attack_num,
            "attack_effect_table": attack_effects,
            "skill_cat_table": table(skill_cat_rows, config.max_skill_id, 4 + len(SKILL_BOOLEANS) + 24),
            "skill_num_table": skill_num,
            "skill_effect_table": skill_effects,
            "effect_cat_table": table(effect_cat_rows, config.max_effect_id, 9 + len(EFFECT_BOOLEANS) + 15),
            "effect_num_table": effect_num,
        }

    def _validate_categorical_tables(self) -> None:
        for name, table, vocabularies in (
            ("card", self.card_cat_table, self.card_cat_vocabs),
            ("attack", self.attack_cat_table, self.attack_cat_vocabs),
            ("skill", self.skill_cat_table, self.skill_cat_vocabs),
            ("effect", self.effect_cat_table, self.effect_cat_vocabs),
        ):
            if table.shape[1] != len(vocabularies):
                raise ValueError(f"{name} prototype width mismatch")
            for column, vocabulary in enumerate(vocabularies):
                minimum, maximum = int(table[:, column].min()), int(table[:, column].max())
                if minimum < 0 or maximum >= vocabulary:
                    raise ValueError(f"{name} field {column} range [{minimum}, {maximum}] exceeds {vocabulary}")

    @staticmethod
    def _validate_identity(identity: Tensor, maximum: int, kind: str) -> None:
        if identity.device.type == "cpu" and (torch.any(identity < 0) or torch.any(identity > maximum)):
            raise ValueError(f"{kind} identity outside configured range")

    def card(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_card_id, "card")
        skills = self.card_skill_table[identity]
        attacks = self.card_attack_table[identity]
        encoded = self.norm(self.card_identity(identity) + self.card_cat(self.card_cat_table[identity])
                            + self.card_num(self.card_num_table[identity]) + self.skill(skills).sum(dim=-2)
                            + self.attack(attacks).sum(dim=-2))
        return encoded * identity.ne(0).unsqueeze(-1)

    def attack(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_attack_id, "attack")
        effects = self.attack_effect_table[identity]
        encoded = self.norm(self.attack_identity(identity) + self.attack_cat(self.attack_cat_table[identity])
                            + self.attack_num(self.attack_num_table[identity])
                            + self.effect(effects).sum(dim=-2))
        return encoded * identity.ne(0).unsqueeze(-1)

    def skill(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_skill_id, "skill")
        effects = self.skill_effect_table[identity]
        encoded = self.norm(self.skill_identity(identity) + self.skill_cat(self.skill_cat_table[identity])
                            + self.skill_num(self.skill_num_table[identity])
                            + self.effect(effects).sum(dim=-2))
        return encoded * identity.ne(0).unsqueeze(-1)

    def effect(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_effect_id, "effect")
        encoded = self.norm(self.effect_identity(identity) + self.effect_cat(self.effect_cat_table[identity])
                            + self.effect_num(self.effect_num_table[identity]))
        return encoded * identity.ne(0).unsqueeze(-1)

    def encode_all(self) -> PrototypeEmbeddings:
        device = self.card_identity.weight.device
        return PrototypeEmbeddings(
            self.card(torch.arange(self.config.max_card_id + 1, device=device)),
            self.attack(torch.arange(self.config.max_attack_id + 1, device=device)),
            self.skill(torch.arange(self.config.max_skill_id + 1, device=device)),
            self.effect(torch.arange(self.config.max_effect_id + 1, device=device)),
        )


__all__ = ["OfficialPrototypeEncoder", "PrototypeEmbeddings"]

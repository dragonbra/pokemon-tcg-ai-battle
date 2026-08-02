"""Typed, relation-preserving encoders for official Card/Attack/Skill/Effect prototypes."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ...features.prototypes import FieldState, PrototypeIndex
from .config import CanonicalModelConfig
from .typed import CategoricalFields, NumericFields


CARD_CAT_VOCABS = (8, 16, 8, 14, 14, 14)
ATTACK_CAT_VOCABS = (2049,)
SKILL_CAT_VOCABS = (2049, 18, 4, 4, 4, 4)
EFFECT_CAT_VOCABS = (4, 4, 246, 34, 66, 130, 18, 6)


class PrototypeBank(nn.Module):
    def __init__(self, config: CanonicalModelConfig, prototypes: PrototypeIndex):
        super().__init__()
        self.config = config
        maxima = {
            "card": max(prototypes.engine_cards, default=0),
            "attack": max(prototypes.engine_attacks, default=0),
            "skill": max(prototypes.skills, default=0),
            "effect": max(prototypes.effects, default=0),
        }
        limits = {
            "card": config.max_card_id,
            "attack": config.max_attack_id,
            "skill": config.max_skill_id,
            "effect": config.max_effect_id,
        }
        exceeded = {
            name: (maxima[name], limits[name])
            for name in maxima
            if maxima[name] > limits[name]
        }
        if exceeded:
            raise ValueError(f"prototype identities exceed canonical config: {exceeded}")
        d = config.d_model
        self.card_identity = nn.Embedding(config.max_card_id + 1, d, padding_idx=0)
        self.attack_identity = nn.Embedding(config.max_attack_id + 1, d, padding_idx=0)
        self.skill_identity = nn.Embedding(config.max_skill_id + 1, d, padding_idx=0)
        self.effect_identity = nn.Embedding(config.max_effect_id + 1, d, padding_idx=0)
        self.card_cat = CategoricalFields(CARD_CAT_VOCABS, d)
        self.card_num = NumericFields(11, d)
        self.attack_cat = CategoricalFields(ATTACK_CAT_VOCABS, d)
        self.attack_num = NumericFields(36, d)
        self.skill_cat = CategoricalFields(SKILL_CAT_VOCABS, d)
        self.skill_num = NumericFields(12, d)
        self.effect_cat = CategoricalFields(EFFECT_CAT_VOCABS, d)
        self.effect_num = NumericFields(22, d)
        self.norm = nn.LayerNorm(d)
        tables = self._tables(config, prototypes)
        categorical_tables = (
            ("card", tables["card_cat_table"], CARD_CAT_VOCABS),
            ("attack", tables["attack_cat_table"], ATTACK_CAT_VOCABS),
            ("skill", tables["skill_cat_table"], SKILL_CAT_VOCABS),
            ("effect", tables["effect_cat_table"], EFFECT_CAT_VOCABS),
        )
        for name, table, vocabularies in categorical_tables:
            for column, vocabulary in enumerate(vocabularies):
                minimum = int(table[:, column].min().item())
                maximum = int(table[:, column].max().item())
                if minimum < 0 or maximum >= vocabulary:
                    raise ValueError(
                        f"{name} prototype field {column} range "
                        f"[{minimum}, {maximum}] exceeds vocabulary {vocabulary}"
                    )
        for name, value in tables.items():
            self.register_buffer(name, value, persistent=True)

    @staticmethod
    def _tables(config: CanonicalModelConfig, prototypes: PrototypeIndex) -> dict[str, Tensor]:
        card_cat = torch.zeros(config.max_card_id + 1, 6, dtype=torch.long)
        card_num = torch.zeros(config.max_card_id + 1, 11)
        card_skills = torch.zeros(config.max_card_id + 1, 3, dtype=torch.long)
        for identity, full in prototypes.engine_cards.items():
            public = prototypes.cards[identity]
            hp = public["hp"]
            retreat = public["retreat_cost"]
            flags = public["rule_flags"]
            card_cat[identity] = torch.tensor(
                [
                    int(full["card_type"]) + 1,
                    int(full["pokemon_type"]) + 1,
                    int(full["evolution_type"]) + 1,
                    int(public["energy_type"]["value"]) + 1,
                    int(public["weakness_type"]["value"]) + 1,
                    int(public["resistance_type"]["value"]) + 1,
                ]
            )
            card_num[identity] = torch.tensor(
                [
                    float(hp["value"]) / 400.0,
                    float(hp["state"]) / 3.0,
                    float(retreat["value"]) / 5.0,
                    float(retreat["state"]) / 3.0,
                    float(full["energy_count"]) / 10.0,
                    float(flags["ex"]),
                    float(flags["mega_ex"]),
                    float(flags["tera"]),
                    float(flags["ace_spec"]),
                    len(public["skills"]) / 4.0,
                    len(full["attack_ids"]) / 4.0,
                ]
            )
            card_skills[identity] = torch.tensor(
                [full["ability_skill_id"], full["play_skill_id"], full["delay_skill_id"]]
            )

        attack_cat = torch.zeros(config.max_attack_id + 1, 1, dtype=torch.long)
        attack_num = torch.zeros(config.max_attack_id + 1, 36)
        for identity, full in prototypes.engine_attacks.items():
            public = prototypes.attacks[identity]
            energy = torch.bincount(
                torch.tensor(public["energy_types"], dtype=torch.long), minlength=12
            )[:12].float()
            attack_cat[identity] = torch.tensor(
                [min(config.max_card_id, int(full["card_id"]))]
            )
            flag_bits = torch.tensor(
                [float(bool(int(full["attack_flags"]) & (1 << bit))) for bit in range(20)]
            )
            attack_num[identity] = torch.cat(
                (
                    torch.tensor(
                        [
                            float(public["base_damage"]["value"]) / 400.0,
                            float(public["base_damage"]["state"]) / 3.0,
                            len(full["energies"]) / 10.0,
                            (len(full["pre_effects"]) + len(full["post_effects"])) / 16.0,
                        ]
                    ),
                    energy / 4.0,
                    flag_bits,
                )
            )

        skill_cat = torch.zeros(config.max_skill_id + 1, 6, dtype=torch.long)
        skill_num = torch.zeros(config.max_skill_id + 1, 12)
        for identity, item in prototypes.skills.items():
            skill_cat[identity] = torch.tensor(
                [
                    min(config.max_card_id, int(item["card_id"])),
                    int(item["skill_type"]) + 1,
                    int(bool(item["main_ability"])) + 1,
                    int(bool(item["once_turn"])) + 1,
                    int(bool(item["select_activation"])) + 1,
                    int(bool(item["activate_in_discard"])) + 1,
                ]
            )
            skill_num[identity] = torch.tensor(
                [
                    len(item["areas"]) / 8.0,
                    len(item["triggers"]) / 8.0,
                    len(item["effects"]) / 16.0,
                    float(item["priority"]) / 32.0,
                    float(item["first_condition_count"]) / 16.0,
                    float(item["not_stack"]),
                    float(item["attach_bench"]),
                    float(item["ko_self"]),
                    float(item["lucky_bonus"]),
                    float(item["second_effect_start"]) / 16.0,
                    float(item["second_effect_start_enemy"]) / 16.0,
                    float(item["trigger_start"]) / 16.0,
                ]
            )

        effect_cat = torch.zeros(config.max_effect_id + 1, 8, dtype=torch.long)
        effect_num = torch.zeros(config.max_effect_id + 1, 22)
        boolean_names = (
            "enemy_select",
            "random_select",
            "each_selected",
            "each_list",
            "add_check_list",
            "keep_selected_list",
            "exclude_previous_target",
            "keep_target_list",
            "multiply_previous_target_count",
            "multiply_coin_heads",
            "can_select_zero",
            "cannot_select_zero",
            "energy_max_select",
            "select_target_count",
            "select_coin_head_count",
            "select_enemy_energy_count",
            "skip_no_target",
            "seeing_deck",
        )
        for identity, item in prototypes.effects.items():
            target = item["target"]
            effect_cat[identity] = torch.tensor(
                [
                    int(item["parent_kind"]) + 1,
                    int(item["phase"]) + 1,
                    int(item["effect_type"]) + 1,
                    int(item["select_type"]) + 1,
                    int(item["select_context"]) + 1,
                    int(item["condition_type"]) + 1,
                    int(item["comparator_type"]) + 1,
                    int(target["target_player"]) + 1,
                ]
            )
            effect_num[identity] = torch.tensor(
                [
                    float(item["select_count"]) / 16.0,
                    float(item["values"][0]) / 400.0,
                    float(item["values"][1]) / 400.0,
                    len(target["areas"]) / 8.0,
                    *[float(bool(item[name])) for name in boolean_names],
                ]
            )
        return {
            "card_cat_table": card_cat,
            "card_num_table": card_num,
            "card_skill_table": card_skills,
            "attack_cat_table": attack_cat,
            "attack_num_table": attack_num,
            "skill_cat_table": skill_cat,
            "skill_num_table": skill_num,
            "effect_cat_table": effect_cat,
            "effect_num_table": effect_num,
        }

    def card(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_card_id, "card")
        skills = self.card_skill_table[identity]
        return self.norm(
            self.card_identity(identity)
            + self.card_cat(self.card_cat_table[identity])
            + self.card_num(self.card_num_table[identity])
            + self.skill_identity(skills).sum(dim=-2)
        )

    def attack(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_attack_id, "attack")
        return self.norm(
            self.attack_identity(identity)
            + self.attack_cat(self.attack_cat_table[identity])
            + self.attack_num(self.attack_num_table[identity])
        )

    def skill(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_skill_id, "skill")
        return self.norm(
            self.skill_identity(identity)
            + self.skill_cat(self.skill_cat_table[identity])
            + self.skill_num(self.skill_num_table[identity])
        )

    def effect(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_effect_id, "effect")
        return self.norm(
            self.effect_identity(identity)
            + self.effect_cat(self.effect_cat_table[identity])
            + self.effect_num(self.effect_num_table[identity])
        )

    @staticmethod
    def _validate_identity(identity: Tensor, maximum: int, kind: str) -> None:
        if torch.any(identity < 0) or torch.any(identity > maximum):
            raise ValueError(f"{kind} identity exceeds canonical vocabulary {maximum}")


__all__ = ["PrototypeBank"]

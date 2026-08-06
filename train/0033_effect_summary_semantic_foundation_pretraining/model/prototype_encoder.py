"""Type-aware prototype encoder with static engine effect summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..domain.prototypes import FieldState, PrototypeIndex
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
ENERGY_MASK_BITS = 9
TYPE_SPACE_SIZE = ENERGY_MASK_BITS + 1
AREA_TYPE_COUNT = 25
CARD_ATTACK_SLOTS = 2
ATTACK_ENERGY_SLOTS = 5
TRIGGER_SLOTS = 2
TRIGGER_AREA_SLOTS = 2
TRIGGER_CONDITION_SLOTS = 2
EFFECT_SLOTS = 12
EFFECT_NUM_WIDTH = 5
EFFECT_BOOLEAN_FLAGS = (
    "enemy_select", "random_select", "each_selected", "each_list",
    "add_check_list", "keep_selected_list", "exclude_previous_target",
    "keep_target_list", "multiply_previous_target_count", "multiply_coin_heads",
    "can_select_zero", "cannot_select_zero", "energy_max_select",
    "select_target_count", "select_coin_head_count", "select_coin_head_count_x2",
    "select_enemy_energy_count", "skip_no_target", "open", "switch_bench_as_target",
    "active_effect_target", "bench_effect_target", "remove_if_no_effect",
    "seeing_deck", "separator", "fail_skip",
)

EFFECT_CAT_VOCABS = (
    256, 16, 40, 8, 3, 3,
    *([3] * AREA_TYPE_COUNT),
    32, 8,
    *([3] * len(EFFECT_BOOLEAN_FLAGS)),
)


@dataclass(frozen=True, slots=True)
class PrototypeEmbeddings:
    cards: Tensor
    attacks: Tensor
    skills: Tensor

    def card(self, identity: Tensor) -> Tensor:
        return self.cards[identity]

    def attack(self, identity: Tensor) -> Tensor:
        return self.attacks[identity]

    def skill(self, identity: Tensor) -> Tensor:
        return self.skills[identity]


def _symbols(prototypes: PrototypeIndex) -> dict[str, int]:
    """Only retain symbols that express an approved rule relation."""
    values: set[str] = set()
    for card in prototypes.engine_cards.values():
        values.update(str(card.get(name, "")) for name in ("evolves_from", "evolves_from_2"))
    for skill in prototypes.skills.values():
        for trigger in skill.get("triggers", ()):
            for condition in trigger.get("subject", {}).get("conditions", ()):
                values.add(str(condition.get("name", "")))
    values.discard("")
    return {value: index + 1 for index, value in enumerate(sorted(values))}


def _mask_bits(value: Any) -> list[int]:
    raw = int(value)
    return [int(bool(raw & (1 << bit))) + 1 for bit in range(ENERGY_MASK_BITS)]


class OfficialPrototypeEncoder(nn.Module):
    """Encode engine facts, card roles, shared type semantics, and static effects."""

    def __init__(self, config: ModelConfig, prototypes: PrototypeIndex):
        super().__init__()
        self.config = config
        maxima = {
            "card": max(prototypes.engine_cards, default=0),
            "attack": max(prototypes.engine_attacks, default=0),
            "skill": max(prototypes.skills, default=0),
        }
        limits = {
            "card": config.max_card_id,
            "attack": config.max_attack_id,
            "skill": config.max_skill_id,
        }
        exceeded = {
            name: (value, limits[name])
            for name, value in maxima.items()
            if value > limits[name]
        }
        if exceeded:
            raise ValueError(f"prototype identities exceed model limits: {exceeded}")

        symbols = _symbols(prototypes)
        symbol_vocab = max(symbols.values(), default=0) + 1
        self.card_cat_vocabs = (
            8, 6, 5, symbol_vocab, symbol_vocab,
            *([3] * (len(CARD_BOOLEANS) + ENERGY_MASK_BITS * 3)),
        )
        self.attack_cat_vocabs = (
            *([3] * 20),
            *([513] * ATTACK_ENERGY_SLOTS),
            3,
        )
        trigger_vocabs = (
            32, 5, 3, 3,
            *([26] * TRIGGER_AREA_SLOTS),
            *sum(((103, 7, symbol_vocab) for _ in range(TRIGGER_CONDITION_SLOTS)), ()),
        )
        self.skill_cat_vocabs = (
            18,
            *([3] * AREA_TYPE_COUNT),
            *([3] * len(SKILL_BOOLEANS)),
            *(trigger_vocabs * TRIGGER_SLOTS),
        )

        d = config.d_model
        self.card_identity = nn.Embedding(config.max_card_id + 1, d, padding_idx=0)
        self.attack_identity = nn.Embedding(config.max_attack_id + 1, d, padding_idx=0)
        self.skill_identity = nn.Embedding(config.max_skill_id + 1, d, padding_idx=0)
        self.card_cat = CategoricalFields(self.card_cat_vocabs, d)
        self.card_num = NumericFields(3, d)
        self.attack_cat = CategoricalFields(self.attack_cat_vocabs, d)
        self.attack_num = NumericFields(1, d)
        self.skill_cat = CategoricalFields(self.skill_cat_vocabs, d)
        self.skill_num = NumericFields(
            TRIGGER_SLOTS * TRIGGER_CONDITION_SLOTS * 2,
            d,
        )
        self.effect_cat = CategoricalFields(EFFECT_CAT_VOCABS, d)
        self.effect_num = NumericFields(EFFECT_NUM_WIDTH, d)
        self.effect_slot = nn.Embedding(EFFECT_SLOTS + 1, d, padding_idx=0)
        self.type_identity = nn.Embedding(TYPE_SPACE_SIZE + 1, d, padding_idx=0)
        self.type_roles = nn.ModuleList(
            nn.Linear(d, d, bias=False)
            for _ in range(5)
        )
        self.card_type_projections = nn.ModuleList(
            nn.Linear(d, d, bias=False)
            for _ in range(self.card_cat_vocabs[0])
        )
        self.norm = nn.LayerNorm(d)
        for name, table in self._build_tables(config, prototypes, symbols).items():
            self.register_buffer(name, table, persistent=True)
        self._validate_categorical_tables()

    @staticmethod
    def _build_tables(
        config: ModelConfig,
        prototypes: PrototypeIndex,
        symbols: Mapping[str, int],
    ) -> dict[str, Tensor]:
        card_cat_rows: dict[int, list[int]] = {}
        card_num = torch.zeros(config.max_card_id + 1, 3)
        card_skills = torch.zeros(config.max_card_id + 1, 3, dtype=torch.long)
        card_attacks = torch.zeros(
            config.max_card_id + 1,
            CARD_ATTACK_SLOTS,
            dtype=torch.long,
        )
        for identity, card in prototypes.engine_cards.items():
            card_cat_rows[identity] = [
                int(card["card_type"]) + 1,
                int(card["pokemon_type"]) + 1,
                int(card["evolution_type"]) + 1,
                symbols.get(str(card.get("evolves_from", "")), 0),
                symbols.get(str(card.get("evolves_from_2", "")), 0),
                *[int(bool(card[name])) + 1 for name in CARD_BOOLEANS],
                *_mask_bits(card["energy_type_mask"]),
                *_mask_bits(card["weakness"]),
                *_mask_bits(card["resistance"]),
            ]
            card_num[identity] = torch.tensor(
                [float(card["hp"]), float(card["retreat_cost"]), float(card["energy_count"])]
            )
            card_skills[identity] = torch.tensor(
                [
                    int(card["ability_skill_id"]),
                    int(card["play_skill_id"]),
                    int(card["delay_skill_id"]),
                ]
            )
            attacks = list(card["attack_ids"])
            if len(attacks) > CARD_ATTACK_SLOTS:
                raise ValueError(f"card {identity} exceeds audited attack slots")
            if attacks:
                card_attacks[identity, : len(attacks)] = torch.tensor(attacks)

        attack_cat_rows: dict[int, list[int]] = {}
        attack_num = torch.zeros(config.max_attack_id + 1, 1)
        attack_effect_cat = torch.zeros(
            config.max_attack_id + 1,
            EFFECT_SLOTS,
            len(EFFECT_CAT_VOCABS),
            dtype=torch.long,
        )
        attack_effect_num = torch.zeros(
            config.max_attack_id + 1,
            EFFECT_SLOTS,
            EFFECT_NUM_WIDTH,
        )
        attack_effect_state = torch.zeros_like(attack_effect_num, dtype=torch.long)

        def effect_cat(effect: Mapping[str, Any] | None) -> list[int]:
            if effect is None:
                return [0] * len(EFFECT_CAT_VOCABS)
            target = effect.get("target", {}) or {}
            area_set = {int(area) for area in target.get("areas", ())}
            return [
                int(effect.get("effect_type", 0)) + 1,
                int(effect.get("select_type", 0)) + 1,
                int(effect.get("select_context", 0)) + 1,
                int(target.get("target_player", 0)) + 1,
                int(bool(target.get("not_me"))) + 1,
                int(bool(target.get("skip_enemy_target"))) + 1,
                *[int(area in area_set) + 1 for area in range(AREA_TYPE_COUNT)],
                int(effect.get("condition_type", 0)) + 1,
                int(effect.get("comparator_type", 0)) + 1,
                *[int(bool(effect.get(name))) + 1 for name in EFFECT_BOOLEAN_FLAGS],
            ]

        def effect_num(effect: Mapping[str, Any] | None) -> list[float]:
            if effect is None:
                return [0.0] * EFFECT_NUM_WIDTH
            values = list(effect.get("values", ()) or ())
            return [
                float(effect.get("select_count", 0)),
                float(effect.get("loop_count", 0)),
                float(effect.get("priority", 0)),
                float(values[0]) if values else 0.0,
                float(values[1]) if len(values) > 1 else 0.0,
            ]

        def write_effects(
            cat_table: Tensor,
            num_table: Tensor,
            state_table: Tensor,
            identity: int,
            effects: list[Mapping[str, Any]],
            kind: str,
        ) -> None:
            if len(effects) > EFFECT_SLOTS:
                raise ValueError(f"{kind} {identity} exceeds audited effect slots")
            for index, effect in enumerate(effects):
                cat_table[identity, index] = torch.tensor(effect_cat(effect))
                num_table[identity, index] = torch.tensor(effect_num(effect))
                state_table[identity, index] = torch.tensor(
                    [int(FieldState.PRESENT)] * EFFECT_NUM_WIDTH
                )

        for identity, attack in prototypes.engine_attacks.items():
            energies = list(attack["energies"])
            if len(energies) > ATTACK_ENERGY_SLOTS:
                raise ValueError(f"attack {identity} exceeds audited energy slots")
            attack_cat_rows[identity] = [
                *[
                    int(bool(int(attack["attack_flags"]) & (1 << bit))) + 1
                    for bit in range(20)
                ],
                *[
                    int(energies[index]) + 1 if index < len(energies) else 0
                    for index in range(ATTACK_ENERGY_SLOTS)
                ],
                int(bool(attack.get("last_cancel_fail_attack"))) + 1,
            ]
            attack_num[identity, 0] = float(attack["damage"])
            write_effects(
                attack_effect_cat,
                attack_effect_num,
                attack_effect_state,
                identity,
                list(attack.get("pre_effects", ())) + list(attack.get("post_effects", ())),
                "attack",
            )

        def trigger_cat(trigger: Mapping[str, Any] | None) -> list[int]:
            if trigger is None:
                return [0] * 12
            subject = trigger.get("subject", {})
            areas = list(subject.get("areas", ()))
            conditions = list(subject.get("conditions", ()))
            if len(areas) > TRIGGER_AREA_SLOTS or len(conditions) > TRIGGER_CONDITION_SLOTS:
                raise ValueError("trigger subject exceeds audited fixed slots")
            result = [
                int(trigger.get("trigger_type", 0)) + 1,
                int(subject.get("target_player", 0)) + 1,
                int(bool(subject.get("not_me"))) + 1,
                int(bool(subject.get("skip_enemy_target"))) + 1,
                *[
                    int(areas[index]) + 1 if index < len(areas) else 0
                    for index in range(TRIGGER_AREA_SLOTS)
                ],
            ]
            for index in range(TRIGGER_CONDITION_SLOTS):
                condition = conditions[index] if index < len(conditions) else None
                result.extend(
                    [
                        int(condition.get("target_type", 0)) + 1 if condition else 0,
                        int(condition.get("comparator_type", 0)) + 1 if condition else 0,
                        symbols.get(str(condition.get("name", "")), 0) if condition else 0,
                    ]
                )
            return result

        def trigger_num(trigger: Mapping[str, Any] | None) -> list[float]:
            conditions = list(trigger.get("subject", {}).get("conditions", ())) if trigger else []
            result: list[float] = []
            for index in range(TRIGGER_CONDITION_SLOTS):
                condition = conditions[index] if index < len(conditions) else None
                result.extend(
                    [
                        float(condition.get("value", 0)) if condition else 0.0,
                        float(condition.get("value2", 0)) if condition else 0.0,
                    ]
                )
            return result

        skill_cat_rows: dict[int, list[int]] = {}
        skill_num = torch.zeros(
            config.max_skill_id + 1,
            TRIGGER_SLOTS * TRIGGER_CONDITION_SLOTS * 2,
        )
        skill_num_state = torch.full_like(skill_num, 3, dtype=torch.long)
        skill_effect_cat = torch.zeros(
            config.max_skill_id + 1,
            EFFECT_SLOTS,
            len(EFFECT_CAT_VOCABS),
            dtype=torch.long,
        )
        skill_effect_num = torch.zeros(
            config.max_skill_id + 1,
            EFFECT_SLOTS,
            EFFECT_NUM_WIDTH,
        )
        skill_effect_state = torch.zeros_like(skill_effect_num, dtype=torch.long)
        for identity, skill in prototypes.skills.items():
            areas = list(skill["areas"])
            triggers = list(skill["triggers"])
            if any(not 0 <= int(area) < AREA_TYPE_COUNT for area in areas):
                raise ValueError(f"skill {identity} has an unsupported area")
            if len(triggers) > TRIGGER_SLOTS:
                raise ValueError(f"skill {identity} exceeds audited trigger slots")
            area_set = {int(area) for area in areas}
            skill_cat_rows[identity] = [
                int(skill["skill_type"]) + 1,
                *[int(area in area_set) + 1 for area in range(AREA_TYPE_COUNT)],
                *[int(bool(skill[name])) + 1 for name in SKILL_BOOLEANS],
                *trigger_cat(triggers[0] if triggers else None),
                *trigger_cat(triggers[1] if len(triggers) > 1 else None),
            ]
            skill_num[identity] = torch.tensor(
                [
                    *trigger_num(triggers[0] if triggers else None),
                    *trigger_num(triggers[1] if len(triggers) > 1 else None),
                ]
            )
            state_values: list[int] = []
            for trigger_index in range(TRIGGER_SLOTS):
                trigger = triggers[trigger_index] if trigger_index < len(triggers) else None
                conditions = list(trigger.get("subject", {}).get("conditions", ())) if trigger else []
                for condition_index in range(TRIGGER_CONDITION_SLOTS):
                    state_values.extend([1, 1] if condition_index < len(conditions) else [3, 3])
            skill_num_state[identity] = torch.tensor(state_values)
            write_effects(
                skill_effect_cat,
                skill_effect_num,
                skill_effect_state,
                identity,
                list(skill.get("effects", ())),
                "skill",
            )

        def table(rows: Mapping[int, list[int]], count: int, width: int) -> Tensor:
            output = torch.zeros(count + 1, width, dtype=torch.long)
            for identity, values in rows.items():
                output[identity] = torch.tensor(values)
            return output

        return {
            "card_cat_table": table(card_cat_rows, config.max_card_id, 5 + len(CARD_BOOLEANS) + ENERGY_MASK_BITS * 3),
            "card_num_table": card_num,
            "card_skill_table": card_skills,
            "card_attack_table": card_attacks,
            "attack_cat_table": table(
                attack_cat_rows,
                config.max_attack_id,
                20 + ATTACK_ENERGY_SLOTS + 1,
            ),
            "attack_num_table": attack_num,
            "attack_effect_cat_table": attack_effect_cat,
            "attack_effect_num_table": attack_effect_num,
            "attack_effect_state_table": attack_effect_state,
            "skill_cat_table": table(
                skill_cat_rows,
                config.max_skill_id,
                len(next(iter(skill_cat_rows.values()), []))
                or (1 + AREA_TYPE_COUNT + len(SKILL_BOOLEANS) + 12 * TRIGGER_SLOTS),
            ),
            "skill_num_table": skill_num,
            "skill_num_state_table": skill_num_state,
            "skill_effect_cat_table": skill_effect_cat,
            "skill_effect_num_table": skill_effect_num,
            "skill_effect_state_table": skill_effect_state,
        }

    def _validate_categorical_tables(self) -> None:
        for name, table, vocabularies in (
            ("card", self.card_cat_table, self.card_cat_vocabs),
            ("attack", self.attack_cat_table, self.attack_cat_vocabs),
            ("skill", self.skill_cat_table, self.skill_cat_vocabs),
            ("attack_effect", self.attack_effect_cat_table.flatten(0, 1), EFFECT_CAT_VOCABS),
            ("skill_effect", self.skill_effect_cat_table.flatten(0, 1), EFFECT_CAT_VOCABS),
        ):
            if table.shape[1] != len(vocabularies):
                raise ValueError(f"{name} prototype width mismatch")
            for column, vocabulary in enumerate(vocabularies):
                minimum = int(table[:, column].min())
                maximum = int(table[:, column].max())
                if minimum < 0 or maximum >= vocabulary:
                    raise ValueError(
                        f"{name} field {column} range [{minimum}, {maximum}] exceeds {vocabulary}"
                    )

    @staticmethod
    def _validate_identity(identity: Tensor, maximum: int, kind: str) -> None:
        if identity.device.type == "cpu" and (
            torch.any(identity < 0) or torch.any(identity > maximum)
        ):
            raise ValueError(f"{kind} identity outside configured range")

    def _card_base(self, identity: Tensor) -> Tensor:
        base = (
            self.card_identity(identity)
            + self.card_cat(self.card_cat_table[identity])
            + self.card_num(self.card_num_table[identity])
            + self._card_shared_type_embedding(identity)
        )
        return self._project_card_type(base, self.card_cat_table[identity][..., 0])

    def _attack_base(self, identity: Tensor) -> Tensor:
        return (
            self.attack_identity(identity)
            + self.attack_cat(self.attack_cat_table[identity])
            + self.attack_num(self.attack_num_table[identity])
            + self._attack_shared_type_embedding(identity)
            + self._effect_summary(
                self.attack_effect_cat_table[identity],
                self.attack_effect_num_table[identity],
                self.attack_effect_state_table[identity],
            )
        )

    def _skill_base(self, identity: Tensor) -> Tensor:
        return (
            self.skill_identity(identity)
            + self.skill_cat(self.skill_cat_table[identity])
            + self.skill_num(
                self.skill_num_table[identity],
                self.skill_num_state_table[identity],
            )
            + self._effect_summary(
                self.skill_effect_cat_table[identity],
                self.skill_effect_num_table[identity],
                self.skill_effect_state_table[identity],
            )
        )

    def _mask_type_embedding(self, mask_bits: Tensor) -> Tensor:
        indices = torch.arange(
            2,
            TYPE_SPACE_SIZE + 1,
            device=mask_bits.device,
            dtype=torch.long,
        )
        embedded = self.type_identity(indices)
        active = mask_bits.eq(2).to(embedded.dtype)
        return (active.unsqueeze(-1) * embedded).sum(dim=-2)

    def _raw_energy_mask_embedding(self, encoded_slots: Tensor) -> Tensor:
        raw_mask = (encoded_slots - 1).clamp(min=0)
        present = encoded_slots.gt(0)
        colorless = present & raw_mask.eq(0)
        colorless_embedding = (
            colorless.unsqueeze(-1).to(self.type_identity.weight.dtype)
            * self.type_identity.weight[1]
        )
        indices = torch.arange(
            2,
            TYPE_SPACE_SIZE + 1,
            device=raw_mask.device,
            dtype=torch.long,
        )
        bits = torch.arange(ENERGY_MASK_BITS, device=raw_mask.device, dtype=torch.long)
        active = raw_mask.unsqueeze(-1).bitwise_and(1 << bits).ne(0)
        primitive_embedding = (
            active.unsqueeze(-1).to(self.type_identity.weight.dtype)
            * self.type_identity(indices)
        ).sum(dim=-2)
        return colorless_embedding + primitive_embedding

    def _card_shared_type_embedding(self, identity: Tensor) -> Tensor:
        table = self.card_cat_table[identity]
        pokemon_type = (table[..., 1]).clamp(min=0, max=TYPE_SPACE_SIZE)
        energy_start = 5 + len(CARD_BOOLEANS)
        weakness_start = energy_start + ENERGY_MASK_BITS
        resistance_start = weakness_start + ENERGY_MASK_BITS
        pokemon = self.type_roles[0](self.type_identity(pokemon_type))
        energy = self.type_roles[1](
            self._mask_type_embedding(table[..., energy_start:weakness_start])
        )
        weakness = self.type_roles[2](
            self._mask_type_embedding(table[..., weakness_start:resistance_start])
        )
        resistance = self.type_roles[3](
            self._mask_type_embedding(table[..., resistance_start:resistance_start + ENERGY_MASK_BITS])
        )
        return pokemon + energy + weakness + resistance

    def _attack_shared_type_embedding(self, identity: Tensor) -> Tensor:
        energy_slots = self.attack_cat_table[identity][
            ..., 20:20 + ATTACK_ENERGY_SLOTS
        ]
        return self.type_roles[4](self._raw_energy_mask_embedding(energy_slots).sum(dim=-2))

    def _project_card_type(self, values: Tensor, card_type: Tensor) -> Tensor:
        projected = values.new_zeros(values.shape)
        for index, projection in enumerate(self.card_type_projections):
            active = card_type.eq(index).unsqueeze(-1)
            projected = projected + projection(values) * active.to(values.dtype)
        return projected

    def _effect_summary(self, cat: Tensor, num: Tensor, state: Tensor) -> Tensor:
        effects = self.effect_cat(cat) + self.effect_num(num, state)
        mask = cat[..., 0].ne(0)
        slots = torch.arange(1, EFFECT_SLOTS + 1, device=cat.device, dtype=torch.long)
        effects = effects + self.effect_slot(slots) * mask.unsqueeze(-1)
        return (effects * mask.unsqueeze(-1)).sum(dim=-2)

    def card(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_card_id, "card")
        encoded = self.norm(
            self._card_base(identity)
            + self.skill(self.card_skill_table[identity]).sum(dim=-2)
            + self.attack(self.card_attack_table[identity]).sum(dim=-2)
        )
        return encoded * identity.ne(0).unsqueeze(-1)

    def attack(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_attack_id, "attack")
        encoded = self.norm(self._attack_base(identity))
        return encoded * identity.ne(0).unsqueeze(-1)

    def skill(self, identity: Tensor) -> Tensor:
        self._validate_identity(identity, self.config.max_skill_id, "skill")
        encoded = self.norm(self._skill_base(identity))
        return encoded * identity.ne(0).unsqueeze(-1)

    def encode_all(self) -> PrototypeEmbeddings:
        device = self.card_identity.weight.device
        attack_ids = torch.arange(self.config.max_attack_id + 1, device=device)
        skill_ids = torch.arange(self.config.max_skill_id + 1, device=device)
        card_ids = torch.arange(self.config.max_card_id + 1, device=device)
        attacks = self.attack(attack_ids)
        skills = self.skill(skill_ids)
        cards = self.norm(
            self._card_base(card_ids)
            + skills[self.card_skill_table[card_ids]].sum(dim=-2)
            + attacks[self.card_attack_table[card_ids]].sum(dim=-2)
        )
        cards = cards * card_ids.ne(0).unsqueeze(-1)
        return PrototypeEmbeddings(cards=cards, attacks=attacks, skills=skills)


__all__ = [
    "AREA_TYPE_COUNT",
    "CARD_BOOLEANS",
    "EFFECT_BOOLEAN_FLAGS",
    "EFFECT_CAT_VOCABS",
    "EFFECT_SLOTS",
    "OfficialPrototypeEncoder",
    "PrototypeEmbeddings",
    "SKILL_BOOLEANS",
    "TYPE_SPACE_SIZE",
]

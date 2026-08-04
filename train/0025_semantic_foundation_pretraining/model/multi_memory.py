"""Multi-memory semantic pointer policy skeleton for 0025."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..features.prototypes import FieldState, PrototypeIndex


@dataclass(frozen=True)
class SemanticModelConfig:
    d_model: int = 320
    heads: int = 8
    max_card_id: int = 2048
    max_attack_id: int = 4096
    max_skill_ref: int = 20000
    max_effect_ref: int = 8192
    dropout: float = 0.0

    @property
    def max_options(self) -> int:
        return 128

    @property
    def max_action_steps(self) -> int:
        return 64


class MemoryQuery(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, query: Tensor, memory: Tensor, mask: Tensor) -> Tensor:
        null = self.null.expand(query.size(0), -1, -1)
        memory = torch.cat((null, memory), dim=1)
        valid = torch.cat((torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device), mask), dim=1)
        context, _ = self.attention(query, memory, memory, key_padding_mask=~valid, need_weights=False)
        return self.norm(query + torch.tanh(self.gate) * context)


class SemanticFoundationPolicy(nn.Module):
    """Options retrieve from typed memories independently before pointer scoring."""

    def __init__(self, config: SemanticModelConfig, prototypes: PrototypeIndex):
        super().__init__()
        self.config = config
        d = config.d_model
        self.card = nn.Embedding(config.max_card_id + 1, d, padding_idx=0)
        self.attack = nn.Embedding(config.max_attack_id + 1, d, padding_idx=0)
        self.skill = nn.Embedding(config.max_skill_ref + 1, d, padding_idx=0)
        self.effect = nn.Embedding(config.max_effect_ref + 1, d, padding_idx=0)
        self.small = nn.Embedding(4097, d, padding_idx=0)
        self.field_state = nn.Embedding(len(FieldState), d, padding_idx=int(FieldState.PAD))
        self.entity_num = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.entity_cat = nn.Linear(6, d, bias=False)
        self.global_num = nn.Sequential(nn.Linear(17, d), nn.GELU(), nn.Linear(d, d))
        self.option_num = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.Linear(d, d))
        self.option_cat = nn.Linear(7, d, bias=False)
        self.card_static = nn.Linear(18, d)
        self.attack_static = nn.Linear(20, d)
        self.skill_static = nn.Linear(10, d)
        self.effect_static = nn.Linear(16, d)
        self.ledger_num = nn.Sequential(nn.Linear(15, d), nn.GELU(), nn.Linear(d, d))
        self.event_num = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.deck_num = nn.Linear(1, d)
        self.board_encoder = self._encoder(d, config)
        self.prototype_encoder = self._encoder(d, config)
        self.resource_encoder = self._encoder(d, config)
        self.event_encoder = self._encoder(d, config)
        self.board_query = MemoryQuery(d, config.heads, config.dropout)
        self.prototype_query = MemoryQuery(d, config.heads, config.dropout)
        self.resource_query = MemoryQuery(d, config.heads, config.dropout)
        self.event_query = MemoryQuery(d, config.heads, config.dropout)
        self.option_norm = nn.LayerNorm(d)
        self.decoder_init = nn.Linear(d, d)
        self.decoder = nn.GRUCell(d, d)
        self.pointer_query = nn.Linear(d, d, bias=False)
        self.pointer_key = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1)
        self.stop = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        card_table, attack_table, skill_table, effect_table = self._prototype_tables(prototypes, config)
        self.register_buffer("card_feature_table", card_table, persistent=True)
        self.register_buffer("attack_feature_table", attack_table, persistent=True)
        self.register_buffer("skill_feature_table", skill_table, persistent=True)
        self.register_buffer("effect_feature_table", effect_table, persistent=True)

    @staticmethod
    def _encoder(d: int, config: SemanticModelConfig) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d, config.heads, d * 3, config.dropout, "gelu", batch_first=True, norm_first=True
        )
        return nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(d))

    @staticmethod
    def _prototype_tables(prototypes: PrototypeIndex, config: SemanticModelConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cards = torch.zeros(config.max_card_id + 1, 18)
        for card_id, card in prototypes.cards.items():
            if card_id > config.max_card_id:
                continue
            hp, retreat = card["hp"], card["retreat_cost"]
            stage, flags = card["stage"], card["rule_flags"]
            values = [
                float(card["card_type"]) / 6.0, float(hp["value"]) / 400.0,
                float(hp["state"]) / 3.0, float(retreat["value"]) / 5.0,
                float(retreat["state"]) / 3.0, float(card["energy_type"]["value"]) / 11.0,
                float(card["weakness_type"]["value"]) / 11.0,
                float(card["resistance_type"]["value"]) / 11.0,
                float(stage["basic"]), float(stage["stage1"]), float(stage["stage2"]),
                float(flags["ex"]), float(flags["mega_ex"]), float(flags["tera"]),
                float(flags["ace_spec"]), len(card["skills"]) / 4.0,
                len(card["attack_ids"]) / 4.0, float(card["evolves_from"] is not None),
            ]
            cards[card_id] = torch.tensor(values)
        attacks = torch.zeros(config.max_attack_id + 1, 20)
        for attack_id, attack in prototypes.attacks.items():
            if attack_id > config.max_attack_id:
                continue
            counts = torch.bincount(torch.tensor(attack["energy_types"], dtype=torch.long), minlength=12)[:12].float()
            full = prototypes.engine_attacks[attack_id]
            effects = full["pre_effects"] + full["post_effects"]
            extra = torch.tensor([
                len(full["pre_effects"]) / 8.0, len(full["post_effects"]) / 8.0,
                float(bool(full["attack_flags"])), len({item["effect_type"] for item in effects}) / 16.0,
                sum(len(item["target"]["conditions"]) for item in effects) / 8.0,
                float(any(item["select_type"] for item in effects)),
            ])
            attacks[attack_id] = torch.cat((torch.tensor([
                attack["base_damage"]["value"] / 400.0, attack["energy_count"] / 10.0
            ]), counts / 4.0, extra))
        skills = torch.zeros(config.max_skill_ref + 1, 10)
        for skill_id, skill in prototypes.skills.items():
            if skill_id <= config.max_skill_ref:
                skills[skill_id] = torch.tensor([
                    skill["card_id"] / float(config.max_card_id), skill["skill_type"] / 16.0,
                    float(skill["main_ability"]), float(skill["once_turn"]),
                    float(skill["select_activation"]), len(skill["areas"]) / 2.0,
                    len(skill["triggers"]) / 4.0, len(skill["effects"]) / 16.0,
                    skill["priority"] / 16.0, skill["first_condition_count"] / 8.0,
                ])
        effects = torch.zeros(config.max_effect_ref + 1, 16)
        for effect_ref, item in prototypes.effects.items():
            if effect_ref <= config.max_effect_ref:
                effects[effect_ref] = torch.tensor([
                    item["parent_kind"] / 2.0, item["phase"] / 2.0,
                    item["effect_type"] / 128.0, item["select_type"] / 32.0,
                    item["select_count"] / 8.0, item["select_context"] / 64.0,
                    item["values"][0] / 400.0, item["values"][1] / 400.0,
                    item["condition_type"] / 128.0, item["comparator_type"] / 16.0,
                    item["target"]["target_player"] / 4.0, len(item["target"]["areas"]) / 4.0,
                    len(item["target"]["conditions"]) / 8.0, float(item["enemy_select"]),
                    float(item["random_select"]), float(item["seeing_deck"]),
                ])
        return cards, attacks, skills, effects

    @staticmethod
    def _encode_masked(encoder: nn.Module, values: Tensor, mask: Tensor) -> Tensor:
        safe_mask = mask.clone()
        safe_mask[:, 0] = True
        encoded = encoder(values, src_key_padding_mask=~safe_mask)
        return encoded * mask.unsqueeze(-1)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        entity = batch["entity_cat"]
        entity_summary = torch.stack((
            entity[..., 1].float() / 2, entity[..., 2].float() / 24,
            entity[..., 3].float() / 64, entity[..., 4].float() / 7,
            entity[..., 5].float() / 31, entity[..., 6].float() / 192,
        ), dim=-1)
        board = self.card(entity[..., 0]) + self.entity_cat(entity_summary) + self.entity_num(batch["entity_num"])
        board = self._encode_masked(self.board_encoder, board, batch["entity_mask"])

        card_refs = batch["prototype_card_refs"]
        attack_refs = batch["prototype_attack_refs"]
        skill_refs = batch["prototype_skill_refs"]
        effect_refs = batch["prototype_effect_refs"]
        card_memory = self.card(card_refs) + self.card_static(self.card_feature_table[card_refs])
        attack_memory = self.attack(attack_refs) + self.attack_static(self.attack_feature_table[attack_refs])
        skill_memory = self.skill(skill_refs) + self.skill_static(self.skill_feature_table[skill_refs])
        effect_memory = self.effect(effect_refs) + self.effect_static(self.effect_feature_table[effect_refs])
        prototype = torch.cat((card_memory, attack_memory, skill_memory, effect_memory), dim=1)
        prototype_mask = torch.cat((batch["prototype_card_mask"], batch["prototype_attack_mask"], batch["prototype_skill_mask"], batch["prototype_effect_mask"]), dim=1)
        prototype = self._encode_masked(self.prototype_encoder, prototype, prototype_mask)

        ledger = batch["ledger_cat"]
        resource = self.card(ledger[..., 0]) + self.small(ledger[..., 1].clamp_max(4096))
        resource = resource + self.small(ledger[..., 2].clamp_max(4096)) + self.ledger_num(batch["ledger_num"])
        deck = self.card(batch["deck_card_ids"]) + self.deck_num(batch["deck_multiplicity"].unsqueeze(-1) / 4.0)
        resource = torch.cat((resource, deck), dim=1)
        resource_mask = torch.cat((batch["ledger_mask"], batch["deck_mask"]), dim=1)
        resource = self._encode_masked(self.resource_encoder, resource, resource_mask)

        events = batch["event_cat"]
        event_summary = events.float().sum(dim=-1).clamp_max(4096).long()
        event_memory = self.small(event_summary) + self.event_num(batch["event_num"])
        known = self.card(batch["known_hand_ids"])
        event_memory = torch.cat((event_memory, known), dim=1)
        event_mask = torch.cat((batch["event_mask"], batch["known_hand_mask"]), dim=1)
        event_memory = self._encode_masked(self.event_encoder, event_memory, event_mask)

        option = batch["option_cat"]
        semantic = batch["semantic_option_cat"]
        categorical_summary = torch.stack((
            option[..., 0].float() / 64, option[..., 1].float() / 32,
            option[..., 2].float() / 32, option[..., 3].float() / 2,
            semantic[..., 13].float() / 128, semantic[..., 14].float() / 128,
            semantic[..., 15].float() / 3,
        ), dim=-1)
        option_repr = (
            self.option_cat(categorical_summary) + self.card(option[..., 4]) + self.card(option[..., 5])
            + self.attack(semantic[..., 1].clamp_max(self.config.max_attack_id))
            + self.card(semantic[..., 2].clamp_max(self.config.max_card_id))
            + self.option_num(batch["semantic_option_num"])
            + self.field_state(batch["semantic_option_state"]).sum(dim=-2)
        )
        option_repr = self.option_norm(option_repr)
        option_repr = self.board_query(option_repr, board, batch["entity_mask"])
        option_repr = self.prototype_query(option_repr, prototype, prototype_mask)
        option_repr = self.resource_query(option_repr, resource, resource_mask)
        option_repr = self.event_query(option_repr, event_memory, event_mask)
        global_features = torch.cat((batch["global_num"], batch["turn_budget"]), dim=-1)
        state = self.global_num(global_features)
        return state, option_repr * batch["option_mask"].unsqueeze(-1)

    def _logits(
        self,
        batch: dict[str, Tensor],
        options: Tensor,
        hidden: Tensor,
        available: Tensor,
        selected_count: Tensor,
    ) -> Tensor:
        query = self.pointer_query(hidden).unsqueeze(1)
        logits = (query * self.pointer_key(options)).sum(dim=-1) / self.config.d_model ** 0.5
        logits = logits + self.option_bias(options).squeeze(-1)
        can_select = selected_count.lt(batch["max_count"])
        logits = logits.masked_fill(
            ~(available & can_select.unsqueeze(1)), torch.finfo(logits.dtype).min
        )
        stop = self.stop(hidden)
        can_stop = selected_count.ge(batch["min_count"])
        stop = stop.masked_fill(~can_stop.unsqueeze(1), torch.finfo(stop.dtype).min)
        return torch.cat((logits, stop), dim=1)

    def _consume(
        self,
        options: Tensor,
        hidden: Tensor,
        available: Tensor,
        selected_count: Tensor,
        raw_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        option_count = options.size(1)
        valid = raw_index.ge(0) & raw_index.lt(option_count)
        index = raw_index.clamp(min=0, max=option_count - 1)
        chosen = options.gather(
            1, index[:, None, None].expand(-1, 1, options.size(-1))
        ).squeeze(1)
        updated = self.decoder(chosen, hidden)
        hidden = torch.where(valid.unsqueeze(-1), updated, hidden)
        batch_indices = torch.arange(options.size(0), device=options.device)
        available = available.clone()
        available[batch_indices[valid], index[valid]] = False
        selected_count = selected_count + valid.long()
        return hidden, available, selected_count

    def forward(self, batch: dict[str, Tensor], selected_prefix: Tensor | None = None) -> Tensor:
        state, options = self.encode(batch)
        hidden = torch.tanh(self.decoder_init(state))
        selected_count = torch.zeros(options.size(0), dtype=torch.long, device=options.device)
        available = batch["option_mask"].clone()
        if selected_prefix is not None:
            for step in range(selected_prefix.size(1)):
                hidden, available, selected_count = self._consume(
                    options,
                    hidden,
                    available,
                    selected_count,
                    selected_prefix[:, step],
                )
        return self._logits(batch, options, hidden, available, selected_count)

    def teacher_logits(self, batch: dict[str, Tensor]) -> Tensor:
        state, options = self.encode(batch)
        hidden = torch.tanh(self.decoder_init(state))
        selected_count = torch.zeros(options.size(0), dtype=torch.long, device=options.device)
        available = batch["option_mask"].clone()
        outputs: list[Tensor] = []
        for step in range(batch["targets"].size(1)):
            outputs.append(
                self._logits(batch, options, hidden, available, selected_count)
            )
            hidden, available, selected_count = self._consume(
                options,
                hidden,
                available,
                selected_count,
                batch["targets"][:, step],
            )
        return torch.stack(outputs, dim=1)


__all__ = ["SemanticFoundationPolicy", "SemanticModelConfig"]

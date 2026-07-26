from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

try:
    from .base_model import IDOnlyConfig, IDOnlyPointerPolicy
except ImportError:  # Repository training path; candidate packages include base_model.py.
    from train.0010_alakazam_sota_model.model import IDOnlyConfig, IDOnlyPointerPolicy


@dataclass(frozen=True)
class FeatureModelConfig(IDOnlyConfig):
    """0010-compatible shapes plus one explicitly named representation variant."""

    remove_option_position: bool = False
    role_separated_action: bool = False
    action_primitive_context: bool = False
    action_primitive_dim: int = 48
    action_transition_auxiliary: bool = False
    action_history: bool = False
    max_action_history: int = 16

    def validate(self) -> None:
        super().validate()
        if self.action_primitive_dim <= 0:
            raise ValueError("action_primitive_dim must be positive")
        if self.max_action_history <= 0:
            raise ValueError("max_action_history must be positive")

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class FeatureEngineeringPolicy(IDOnlyPointerPolicy):
    """Action-query policy supporting isolated E1/E2 representation ablations."""

    def __init__(self, config: FeatureModelConfig):
        super().__init__(config)
        self.config = config
        if config.role_separated_action:
            self.source_role = nn.Linear(config.d_model, config.d_model, bias=False)
            self.target_role = nn.Linear(config.d_model, config.d_model, bias=False)
        if config.action_primitive_context:
            p = config.action_primitive_dim
            self.attack_id = nn.Embedding(2049, p, padding_idx=0)
            self.primitive_number = nn.Embedding(129, p, padding_idx=0)
            self.tool_index = nn.Embedding(33, p, padding_idx=0)
            self.context_count = nn.Embedding(17, p, padding_idx=0)
            self.primitive_projection = nn.Linear(p, config.d_model, bias=False)
        if config.action_transition_auxiliary:
            self.transition_trunk = nn.Sequential(
                nn.Linear(2 * config.d_model, 32),
                nn.GELU(),
            )
            self.transition_delta_head = nn.Linear(32, 11)
            self.transition_turn_head = nn.Linear(32, 1)
        if config.action_history:
            self.history_position = nn.Embedding(config.max_action_history + 1, config.d_model)
            self.history_kind = nn.Parameter(torch.zeros(config.d_model))

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if (
            not self.config.remove_option_position
            and not self.config.role_separated_action
            and not self.config.action_primitive_context
            and not self.config.action_history
        ):
            return super().encode(batch)

        entity = batch["entity_cat"]
        entity_repr = (
            self.card(entity[..., 0])
            + self.owner(entity[..., 1])
            + self.zone(entity[..., 2])
            + self.slot(entity[..., 3])
            + self.kind(entity[..., 4])
            + self.status(entity[..., 5])
            + self.parent_slot(entity[..., 6].clamp_max(192))
            + self.entity_num(batch["entity_num"])
        )
        entity_repr = self.entity_norm(entity_repr)
        global_values = batch["global_cat"]
        global_repr = self.global_norm(
            self.select_type(global_values[:, 0])
            + self.select_context(global_values[:, 1])
            + self.first_player(global_values[:, 2])
            + self.flags(global_values[:, 3])
            + self.global_num(batch["global_num"])
        )
        cls = self.cls.expand(entity.size(0), -1, -1) + global_repr.unsqueeze(1)
        sequence_parts = [cls, entity_repr]
        padding_parts = [
            torch.zeros((entity.size(0), 1), dtype=torch.bool, device=entity.device),
            ~batch["entity_mask"],
        ]
        if self.config.action_history:
            history = batch["action_history_cat"]
            history_repr = (
                self.select_type(history[..., 0])
                + self.select_context(history[..., 1])
                + self.option_type(history[..., 2])
                + self.card(history[..., 3])
                + self.card(history[..., 4])
                + self.history_position(history[..., 5])
                + self.history_kind
            )
            sequence_parts.append(history_repr)
            padding_parts.append(~batch["action_history_mask"])
        sequence = torch.cat(sequence_parts, dim=1)
        padding = torch.cat(padding_parts, dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        state_repr = encoded[:, 0]
        encoded_entities = encoded[:, 1 : 1 + entity.size(1)]

        option = batch["option_cat"]
        source = (
            self.area(option[..., 1])
            + self.option_owner(option[..., 3])
            + self.card(option[..., 4])
            + self.slot(option[..., 7])
            + self._gather_entities(encoded_entities, option[..., 9])
        )
        target = (
            self.area(option[..., 2])
            + self.card(option[..., 5])
            + self.slot(option[..., 8])
            + self._gather_entities(encoded_entities, option[..., 10])
        )
        if self.config.role_separated_action:
            source = self.source_role(source)
            target = self.target_role(target)
        option_repr = self.option_type(option[..., 0]) + self.number(option[..., 6])
        option_repr = option_repr + source + target
        if self.config.action_primitive_context:
            primitive = batch["option_primitive_cat"]
            context = batch["action_context_cat"]
            primitive_repr = (
                self.attack_id(primitive[..., 0])
                + self.primitive_number(primitive[..., 1])
                + self.primitive_number(primitive[..., 2])
                + self.tool_index(primitive[..., 3])
                + self.context_count(context[:, 2]).unsqueeze(1)
                + self.context_count(context[:, 3]).unsqueeze(1)
            )
            option_repr = (
                option_repr
                + self.primitive_projection(primitive_repr)
                + self.card(context[:, 0]).unsqueeze(1)
                + self.card(context[:, 1]).unsqueeze(1)
            )
        if not self.config.remove_option_position:
            option_repr = option_repr + self.option_position(option[..., 11])
        option_repr = self.option_norm(option_repr)
        attended, _ = self.option_to_state(
            option_repr,
            encoded,
            encoded,
            key_padding_mask=padding,
            need_weights=False,
        )
        option_repr = self.option_state_norm(option_repr + attended)
        option_repr = self.option_ff_norm(option_repr + self.option_ff(option_repr))
        return state_repr, option_repr

    def greedy_action(self, batch: dict[str, Tensor]) -> list[int]:
        """Decode to the simulator's real count bound, including selections longer than 16."""

        if batch["global_cat"].size(0) != 1:
            raise ValueError("greedy_action supports exactly one observation")
        state, options = self.encode(batch)
        option_count = options.size(1)
        keys = self.pointer_key(options)
        hidden = torch.tanh(self.decoder_init(state))
        chosen = torch.zeros((1, option_count), dtype=torch.bool, device=options.device)
        minimum = int(batch["min_count"].item())
        maximum = min(int(batch["max_count"].item()), option_count)
        if minimum > maximum:
            raise RuntimeError(
                f"simulator minCount {minimum} exceeds available option count {maximum}"
            )
        selected_indices: list[int] = []
        for step in range(maximum):
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / self.config.d_model**0.5
            pointer = pointer + self.option_bias(options).squeeze(-1)
            pointer = pointer.masked_fill(
                ~batch["option_mask"] | chosen,
                torch.finfo(pointer.dtype).min,
            )
            stop = self.stop(hidden).squeeze(1)
            best_option = int(pointer[0].argmax().item())
            if step >= minimum and float(stop[0]) >= float(pointer[0, best_option]):
                break
            if not bool(batch["option_mask"][0, best_option]):
                raise RuntimeError("pointer decoder found no legal option before minCount")
            selected_indices.append(best_option)
            chosen[0, best_option] = True
            hidden = self.decoder(options[:, best_option], hidden)
        if len(selected_indices) < minimum:
            raise RuntimeError(
                f"pointer decoder returned {len(selected_indices)} options below minCount {minimum}"
            )
        return selected_indices

    def teacher_logits_and_transition(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return pointer logits and chosen-action next-decision auxiliary predictions."""

        if not self.config.action_transition_auxiliary:
            raise RuntimeError("transition auxiliary is disabled")
        state, options = self.encode(batch)
        batch_size, option_count, _ = options.shape
        keys = self.pointer_key(options)
        hidden = torch.tanh(self.decoder_init(state))
        chosen = torch.zeros((batch_size, option_count), dtype=torch.bool, device=options.device)
        selected_sum = torch.zeros_like(state)
        selected_count = torch.zeros((batch_size, 1), dtype=state.dtype, device=state.device)
        outputs: list[Tensor] = []
        rows = torch.arange(batch_size, device=options.device)
        for step in range(batch["targets"].size(1)):
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / self.config.d_model**0.5
            pointer = pointer + self.option_bias(options).squeeze(-1)
            pointer = pointer.masked_fill(
                ~batch["option_mask"] | chosen,
                torch.finfo(pointer.dtype).min,
            )
            stop = self.stop(hidden)
            stop = stop.masked_fill(
                (step < batch["min_count"]).unsqueeze(-1),
                torch.finfo(stop.dtype).min,
            )
            outputs.append(torch.cat([pointer, stop], dim=1))
            target = batch["targets"][:, step]
            valid = (target >= 0) & (target < option_count)
            safe = target.clamp(min=0, max=option_count - 1)
            selected = options[rows, safe]
            selected_sum = selected_sum + selected * valid.unsqueeze(-1)
            selected_count = selected_count + valid.unsqueeze(-1)
            hidden = torch.where(valid.unsqueeze(-1), self.decoder(selected, hidden), hidden)
            chosen.scatter_(
                1,
                safe.unsqueeze(1),
                chosen.gather(1, safe.unsqueeze(1)) | valid.unsqueeze(1),
            )
        selected_mean = selected_sum / selected_count.clamp_min(1.0)
        transition = self.transition_trunk(torch.cat([state, selected_mean], dim=-1))
        return (
            torch.stack(outputs, dim=1),
            self.transition_delta_head(transition),
            self.transition_turn_head(transition).squeeze(-1),
        )


def load_baseline_weights(
    model: FeatureEngineeringPolicy,
    reference: IDOnlyPointerPolicy,
) -> None:
    """Strict helper used by tests to prove the baseline path is 0010-identical."""

    model.load_state_dict(reference.state_dict(), strict=True)

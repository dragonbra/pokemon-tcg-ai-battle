"""Faithful ID-only pointer policy from the Yushin daily-winner Notebook.

This project intentionally preserves the reference representation: public card
IDs and structural locations only, without card metadata, deck conditioning,
expert conditioning, reward weighting, or hand-authored card semantics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _card_id(card: Any) -> int:
    if isinstance(card, (int, float)):
        return max(0, _as_int(card))
    return max(0, _as_int(_get(card, "id", _get(card, "cardId", 0))))


def _player(current: dict[str, Any], index: int) -> dict[str, Any]:
    players = _as_list(current.get("players"))
    if 0 <= index < len(players) and isinstance(players[index], dict):
        return players[index]
    return {}


def valid_reference_action(action: Any, select: dict[str, Any]) -> bool:
    options = _as_list(select.get("option"))
    if not isinstance(action, list) or not all(isinstance(value, int) for value in action):
        return False
    if len(set(action)) != len(action):
        return False
    minimum = max(0, _as_int(select.get("minCount"), 0))
    maximum = max(minimum, _as_int(select.get("maxCount"), len(options)))
    return minimum <= len(action) <= maximum and all(
        0 <= value < len(options) for value in action
    )


@dataclass(frozen=True)
class IDOnlyConfig:
    max_card_id: int = 2048
    d_model: int = 320
    heads: int = 8
    encoder_layers: int = 4
    ffn_multiplier: int = 3
    dropout: float = 0.10
    max_entities: int = 192
    max_options: int = 128
    max_action_steps: int = 16

    def validate(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must divide heads")
        if self.encoder_layers < 1 or self.ffn_multiplier < 2:
            raise ValueError("invalid transformer depth or width")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class IDOnlyCodec:
    """Encode the exact public-state contract used by the reference Notebook."""

    OWNER_NONE = 0
    OWNER_SELF = 1
    OWNER_OPP = 2
    ZONES = {
        "own_active": 1,
        "own_bench": 2,
        "own_hand": 3,
        "own_discard": 4,
        "opp_active": 5,
        "opp_bench": 6,
        "opp_discard": 7,
        "stadium": 8,
        "looking": 9,
        "select_deck": 10,
        "own_energy": 11,
        "opp_energy": 12,
        "own_tool": 13,
        "opp_tool": 14,
        "own_evolution": 15,
        "opp_evolution": 16,
    }

    def __init__(self, config: IDOnlyConfig):
        self.config = config

    @staticmethod
    def _status_bits(state: dict[str, Any]) -> int:
        names = ("asleep", "burned", "confused", "paralyzed", "poisoned")
        return sum((1 << index) for index, name in enumerate(names) if state.get(name))

    def encode(
        self,
        observation: dict[str, Any],
        action: list[int] | None,
    ) -> dict[str, Any] | None:
        current = _get(observation, "current", {}) or {}
        select = _get(observation, "select", {}) or {}
        options = _as_list(select.get("option"))
        if not options or (action is not None and not valid_reference_action(action, select)):
            return None
        actor = _as_int(current.get("yourIndex"), -1)
        if actor not in (0, 1):
            return None
        opponent = 1 - actor
        own, opp = _player(current, actor), _player(current, opponent)
        flags = 0
        for index, name in enumerate(
            ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated", "turnEnd")
        ):
            if current.get(name):
                flags |= 1 << index
        global_cat = [
            min(max(_as_int(select.get("type"), -1) + 1, 0), 64),
            min(max(_as_int(select.get("context"), -1) + 1, 0), 128),
            min(max(_as_int(current.get("firstPlayer"), -1) + 1, 0), 3),
            min(flags, 63),
        ]
        global_num = [
            min(max(_as_int(current.get("turn"), 0), 0) / 20.0, 2.0),
            min(max(_as_int(current.get("turnActionCount"), 0), 0) / 50.0, 2.0),
            min(max(_as_int(own.get("deckCount"), 0), 0) / 60.0, 1.0),
            min(max(_as_int(opp.get("deckCount"), 0), 0) / 60.0, 1.0),
            min(max(_as_int(own.get("handCount"), len(_as_list(own.get("hand")))), 0) / 20.0, 2.0),
            min(max(_as_int(opp.get("handCount"), len(_as_list(opp.get("hand")))), 0) / 20.0, 2.0),
            min(len(_as_list(own.get("prize"))) / 6.0, 1.0),
            min(len(_as_list(opp.get("prize"))) / 6.0, 1.0),
            min(len(options) / float(self.config.max_options), 2.0),
            min(max(_as_int(select.get("remainDamageCounter"), 0), 0) / 300.0, 2.0),
            min(max(_as_int(select.get("remainEnergyCost"), 0), 0) / 10.0, 2.0),
            min((len(_as_list(own.get("bench"))) + len(_as_list(opp.get("bench")))) / 10.0, 1.0),
        ]

        entities: list[list[int]] = []
        entity_num: list[list[float]] = []
        location: dict[tuple[int, int, int], int] = {}

        def add_card(
            card: Any,
            owner: int,
            zone: int,
            slot: int,
            kind: int,
            status: int = 0,
            parent: int = -1,
            key: tuple[int, int, int] | None = None,
        ) -> int:
            raw_id = _card_id(card)
            if raw_id <= 0:
                return -1
            card_value = min(raw_id, self.config.max_card_id)
            hp = _as_float(_get(card, "hp", 0))
            max_hp = max(hp, _as_float(_get(card, "maxHp", hp)))
            energy = _as_list(_get(card, "energyCards", _get(card, "energies", [])))
            tools = _as_list(_get(card, "tools", []))
            evolution = _as_list(_get(card, "preEvolution", []))
            index = len(entities)
            entities.append(
                [
                    card_value,
                    owner,
                    zone,
                    min(slot + 1, 64),
                    kind,
                    min(status, 31),
                    max(parent + 1, 0),
                ]
            )
            entity_num.append(
                [
                    min(max(0.0, max_hp - hp) / 400.0, 2.0),
                    min(len(energy) / 10.0, 2.0),
                    min(len(tools) / 4.0, 2.0),
                    min(len(evolution) / 4.0, 2.0),
                    1.0 if _get(card, "appearThisTurn", _get(card, "appear", False)) else 0.0,
                ]
            )
            if key is not None:
                location[key] = index
            return index

        def add_zone(
            player_index: int,
            name: str,
            area: int,
            own_zone: str,
            opp_zone: str,
        ) -> None:
            state = _player(current, player_index)
            owner = self.OWNER_SELF if player_index == actor else self.OWNER_OPP
            zone = self.ZONES[own_zone if owner == self.OWNER_SELF else opp_zone]
            status = self._status_bits(state) if name == "active" else 0
            for slot, card in enumerate(_as_list(state.get(name))):
                parent = add_card(
                    card,
                    owner,
                    zone,
                    slot,
                    2 if name in ("active", "bench") else 1,
                    status,
                    key=(player_index, area, slot),
                )
                if parent < 0 or name not in ("active", "bench"):
                    continue
                children = (
                    ("energyCards", "energies", "own_energy", "opp_energy", 3),
                    ("tools", "tools", "own_tool", "opp_tool", 4),
                    ("preEvolution", "preEvolution", "own_evolution", "opp_evolution", 5),
                )
                for primary, fallback, own_child, opp_child, kind in children:
                    values = _as_list(_get(card, primary, _get(card, fallback, [])))
                    for child_slot, child in enumerate(values):
                        add_card(
                            child,
                            owner,
                            self.ZONES[own_child if owner == self.OWNER_SELF else opp_child],
                            child_slot,
                            kind,
                            parent=parent,
                        )

        for player_index in (actor, opponent):
            add_zone(player_index, "active", 4, "own_active", "opp_active")
            add_zone(player_index, "bench", 5, "own_bench", "opp_bench")
            if player_index == actor:
                add_zone(player_index, "hand", 2, "own_hand", "own_hand")
            add_zone(player_index, "discard", 3, "own_discard", "opp_discard")
        for slot, card in enumerate(_as_list(current.get("stadium"))):
            add_card(card, self.OWNER_NONE, self.ZONES["stadium"], slot, 6, key=(-1, 7, slot))
        for slot, card in enumerate(_as_list(current.get("looking"))):
            add_card(card, self.OWNER_SELF, self.ZONES["looking"], slot, 7, key=(actor, 12, slot))
        for slot, card in enumerate(_as_list(select.get("deck"))):
            add_card(
                card,
                self.OWNER_SELF,
                self.ZONES["select_deck"],
                slot,
                1,
                key=(actor, 1, slot),
            )
        if len(entities) > self.config.max_entities:
            return None

        option_rows: list[list[int]] = []
        for option_index, raw_option in enumerate(options):
            option = raw_option if isinstance(raw_option, dict) else {}
            source_player = _as_int(option.get("playerIndex"), actor)
            source_area = _as_int(option.get("area"), -1)
            if _as_int(option.get("type"), -1) == 7 and source_area < 0:
                source_area = 2
            source_slot = _as_int(option.get("index"), -1)
            source_entity = location.get((source_player, source_area, source_slot), -1)
            if source_area == 7:
                source_entity = location.get((-1, 7, source_slot), source_entity)
            source_card = _as_int(option.get("cardId"), 0)
            if source_card <= 0 and source_entity >= 0:
                source_card = entities[source_entity][0]
            target_area = _as_int(option.get("inPlayArea"), -1)
            target_slot = _as_int(option.get("inPlayIndex"), -1)
            target_player = _as_int(
                option.get(
                    "inPlayPlayerIndex",
                    option.get("targetPlayerIndex", source_player),
                ),
                actor,
            )
            target_entity = location.get((target_player, target_area, target_slot), -1)
            target_card = entities[target_entity][0] if target_entity >= 0 else 0
            if source_player == actor:
                owner = self.OWNER_SELF
            elif source_player in (0, 1):
                owner = self.OWNER_OPP
            else:
                owner = self.OWNER_NONE
            option_rows.append(
                [
                    min(max(_as_int(option.get("type"), -1) + 1, 0), 64),
                    min(max(source_area + 1, 0), 32),
                    min(max(target_area + 1, 0), 32),
                    owner,
                    min(max(source_card, 0), self.config.max_card_id),
                    min(max(target_card, 0), self.config.max_card_id),
                    min(max(_as_int(option.get("number"), -1) + 1, 0), 128),
                    min(max(source_slot + 1, 0), 128),
                    min(max(target_slot + 1, 0), 128),
                    source_entity + 1,
                    target_entity + 1,
                    min(option_index + 1, self.config.max_options),
                ]
            )
        encoded_action = list(action or [])
        if (
            len(option_rows) > self.config.max_options
            or len(encoded_action) > self.config.max_action_steps
        ):
            return None
        return {
            "global_cat": global_cat,
            "global_num": global_num,
            "entity_cat": entities,
            "entity_num": entity_num,
            "option_cat": option_rows,
            "action": encoded_action,
            "min_count": min(
                max(_as_int(select.get("minCount"), 0), 0),
                self.config.max_action_steps,
            ),
            "max_count": min(
                max(_as_int(select.get("maxCount"), 0), len(encoded_action)),
                self.config.max_action_steps,
            ),
        }


class IDOnlyPointerPolicy(nn.Module):
    """Four-layer entity encoder and autoregressive selected-option pointer."""

    def __init__(self, config: IDOnlyConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.card = nn.Embedding(config.max_card_id + 1, d, padding_idx=0)
        self.owner = nn.Embedding(3, d)
        self.zone = nn.Embedding(24, d)
        self.slot = nn.Embedding(129, d)
        self.kind = nn.Embedding(8, d)
        self.status = nn.Embedding(32, d)
        self.parent_slot = nn.Embedding(193, d)
        self.entity_num = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.entity_norm = nn.LayerNorm(d)
        self.select_type = nn.Embedding(65, d)
        self.select_context = nn.Embedding(129, d)
        self.first_player = nn.Embedding(4, d)
        self.flags = nn.Embedding(64, d)
        self.global_num = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
        self.global_norm = nn.LayerNorm(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.heads,
            dim_feedforward=d * config.ffn_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(d),
        )
        self.option_type = nn.Embedding(65, d)
        self.area = nn.Embedding(33, d)
        self.number = nn.Embedding(129, d)
        self.option_owner = nn.Embedding(3, d)
        self.option_position = nn.Embedding(config.max_options + 1, d)
        self.option_norm = nn.LayerNorm(d)
        self.option_to_state = nn.MultiheadAttention(
            d,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.option_state_norm = nn.LayerNorm(d)
        self.option_ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * d, d),
        )
        self.option_ff_norm = nn.LayerNorm(d)
        self.decoder_init = nn.Linear(d, d)
        self.decoder = nn.GRUCell(d, d)
        self.pointer_query = nn.Linear(d, d, bias=False)
        self.pointer_key = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1)
        self.stop = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    @staticmethod
    def _gather_entities(entities: Tensor, references: Tensor) -> Tensor:
        safe = (references - 1).clamp_min(0)
        gathered = entities.gather(1, safe.unsqueeze(-1).expand(-1, -1, entities.size(-1)))
        return gathered * references.gt(0).unsqueeze(-1)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
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
        sequence = torch.cat([cls, entity_repr], dim=1)
        padding = torch.cat(
            [
                torch.zeros((entity.size(0), 1), dtype=torch.bool, device=entity.device),
                ~batch["entity_mask"],
            ],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        state_repr, encoded_entities = encoded[:, 0], encoded[:, 1:]

        option = batch["option_cat"]
        option_repr = (
            self.option_type(option[..., 0])
            + self.area(option[..., 1])
            + self.area(option[..., 2])
            + self.option_owner(option[..., 3])
            + self.card(option[..., 4])
            + self.card(option[..., 5])
            + self.number(option[..., 6])
            + self.slot(option[..., 7])
            + self.slot(option[..., 8])
            + self.option_position(option[..., 11])
            + self._gather_entities(encoded_entities, option[..., 9])
            + self._gather_entities(encoded_entities, option[..., 10])
        )
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

    def teacher_logits(self, batch: dict[str, Tensor]) -> Tensor:
        state, options = self.encode(batch)
        batch_size, option_count, _ = options.shape
        keys = self.pointer_key(options)
        hidden = torch.tanh(self.decoder_init(state))
        chosen = torch.zeros(
            (batch_size, option_count),
            dtype=torch.bool,
            device=options.device,
        )
        outputs: list[Tensor] = []
        for step in range(batch["targets"].size(1)):
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / math.sqrt(self.config.d_model)
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
            if valid.any():
                safe = target.clamp(min=0, max=option_count - 1)
                selected = options[torch.arange(batch_size, device=options.device), safe]
                hidden = torch.where(
                    valid.unsqueeze(-1),
                    self.decoder(selected, hidden),
                    hidden,
                )
                chosen.scatter_(
                    1,
                    safe.unsqueeze(1),
                    chosen.gather(1, safe.unsqueeze(1)) | valid.unsqueeze(1),
                )
        return torch.stack(outputs, dim=1)

    def greedy_action(self, batch: dict[str, Tensor]) -> list[int]:
        """Decode one legal option sequence while enforcing simulator count bounds."""
        if batch["global_cat"].size(0) != 1:
            raise ValueError("greedy_action supports exactly one observation")
        state, options = self.encode(batch)
        option_count = options.size(1)
        keys = self.pointer_key(options)
        hidden = torch.tanh(self.decoder_init(state))
        chosen = torch.zeros((1, option_count), dtype=torch.bool, device=options.device)
        minimum = int(batch["min_count"].item())
        maximum = min(int(batch["max_count"].item()), self.config.max_action_steps)
        selected_indices: list[int] = []
        for step in range(maximum + 1):
            if step >= maximum:
                break
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / math.sqrt(self.config.d_model)
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


def collate_id_only(rows: list[dict[str, Any]]) -> dict[str, Tensor]:
    if not rows:
        raise ValueError("cannot collate an empty ID-only batch")
    batch_size = len(rows)
    # Keep one masked padding slot so batches containing no public card entities
    # still have a valid tensor shape for entity-reference gathers.
    entity_count = max(1, max(len(row["entity_cat"]) for row in rows))
    option_count = max(len(row["option_cat"]) for row in rows)
    target_steps = max(len(row["action"]) + 1 for row in rows)
    entity_cat = torch.zeros((batch_size, entity_count, 7), dtype=torch.long)
    entity_num = torch.zeros((batch_size, entity_count, 5), dtype=torch.float32)
    entity_mask = torch.zeros((batch_size, entity_count), dtype=torch.bool)
    option_cat = torch.zeros((batch_size, option_count, 12), dtype=torch.long)
    option_mask = torch.zeros((batch_size, option_count), dtype=torch.bool)
    targets = torch.full((batch_size, target_steps), -100, dtype=torch.long)
    global_cat = torch.zeros((batch_size, 4), dtype=torch.long)
    global_num = torch.zeros((batch_size, 12), dtype=torch.float32)
    min_count = torch.zeros(batch_size, dtype=torch.long)
    max_count = torch.zeros(batch_size, dtype=torch.long)
    for index, row in enumerate(rows):
        entities = len(row["entity_cat"])
        options = len(row["option_cat"])
        if entities:
            entity_cat[index, :entities] = torch.tensor(row["entity_cat"], dtype=torch.long)
            entity_num[index, :entities] = torch.tensor(row["entity_num"], dtype=torch.float32)
            entity_mask[index, :entities] = True
        option_cat[index, :options] = torch.tensor(row["option_cat"], dtype=torch.long)
        option_mask[index, :options] = True
        action = list(row["action"])
        if action:
            targets[index, : len(action)] = torch.tensor(action, dtype=torch.long)
        targets[index, len(action)] = option_count
        global_cat[index] = torch.tensor(row["global_cat"], dtype=torch.long)
        global_num[index] = torch.tensor(row["global_num"], dtype=torch.float32)
        min_count[index] = int(row["min_count"])
        max_count[index] = int(row["max_count"])
    return {
        "global_cat": global_cat,
        "global_num": global_num,
        "entity_cat": entity_cat,
        "entity_num": entity_num,
        "entity_mask": entity_mask,
        "option_cat": option_cat,
        "option_mask": option_mask,
        "targets": targets,
        "min_count": min_count,
        "max_count": max_count,
    }

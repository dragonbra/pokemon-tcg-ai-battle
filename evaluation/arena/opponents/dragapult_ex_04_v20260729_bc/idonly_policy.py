"""Train the independent ID-only pointer policy from v1's encoded output.

This kernel deliberately mounts the completed v1 kernel output. It never reads
or re-extracts the ten original episode datasets.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

try:
    import ijson  # type: ignore
except ImportError:
    ijson = None


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
RUNNING_ON_KAGGLE = KAGGLE_INPUT.exists() and KAGGLE_WORKING.exists()
ROOT = KAGGLE_WORKING if RUNNING_ON_KAGGLE else Path.cwd()
DEFAULT_DATES = (
    "2026-07-13,2026-07-14,2026-07-15,2026-07-16,2026-07-17,"
    "2026-07-18,2026-07-19,2026-07-20,2026-07-21,2026-07-22"
)
DATE_RE = re.compile(r"2026-\d{2}-\d{2}")


def getv(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def card_id(card: Any) -> int:
    if isinstance(card, (int, float)):
        return max(0, as_int(card))
    return max(0, as_int(getv(card, "id", getv(card, "cardId", 0))))


def player(current: dict[str, Any], index: int) -> dict[str, Any]:
    players = as_list(current.get("players"))
    return players[index] if 0 <= index < len(players) and isinstance(players[index], dict) else {}


def normalize_team(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def is_winner(rewards: list[Any], index: int) -> bool:
    if not 0 <= index < len(rewards) or not isinstance(rewards[index], (int, float)):
        return False
    numeric = [float(value) for value in rewards if isinstance(value, (int, float))]
    return bool(numeric) and float(rewards[index]) == max(numeric)


def valid_action(action: Any, select: dict[str, Any]) -> bool:
    options = as_list(select.get("option"))
    if not isinstance(action, list) or not all(isinstance(value, int) for value in action):
        return False
    if len(set(action)) != len(action):
        return False
    low = max(0, as_int(select.get("minCount"), 0))
    high = max(low, as_int(select.get("maxCount"), len(options)))
    return low <= len(action) <= high and all(0 <= value < len(options) for value in action)


def episode_date(path: Path) -> str:
    match = DATE_RE.search(str(path).replace("\\", "/"))
    return match.group(0) if match else "unknown"


def visual_decision_frames(data: dict[str, Any]) -> list[tuple[int, dict[str, Any], Any]]:
    traces: list[list[Any]] = []
    singletons: list[Any] = []
    for step in as_list(data.get("steps")):
        for record in as_list(step):
            raw = getv(record, "visualize")
            if raw is None:
                raw = getv(record, "visual")
            if isinstance(raw, list) and raw:
                traces.append(raw)
            elif isinstance(raw, dict):
                singletons.append(raw)
    frames = max(traces, key=len) if traces else singletons
    return [
        (index, frame["obs"], frame.get("selected"))
        for index, frame in enumerate(frames)
        if isinstance(frame, dict) and isinstance(frame.get("obs"), dict)
    ]


def scan_episode_teams(path: Path) -> list[str]:
    """Read only the team header before touching a full replay trace."""
    if ijson is not None:
        with path.open("rb") as handle:
            teams = next(ijson.items(handle, "info.TeamNames"), [])
        return [str(value) for value in as_list(teams)]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [str(value) for value in as_list(getv(data.get("info"), "TeamNames"))]


def scan_episode_rewards(path: Path) -> list[Any]:
    if ijson is not None:
        with path.open("rb") as handle:
            return as_list(next(ijson.items(handle, "rewards"), []))
    with path.open("r", encoding="utf-8") as handle:
        return as_list(json.load(handle).get("rewards"))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ModelConfig:
    max_card_id: int = 2048
    d_model: int = 320
    heads: int = 8
    encoder_layers: int = 4
    ffn_multiplier: int = 3
    dropout: float = 0.10
    max_entities: int = 192
    max_options: int = 128
    max_action_steps: int = 16
    decoder_layers: int = 1

    def validate(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must divide heads")
        if self.encoder_layers < 1 or self.ffn_multiplier < 2:
            raise ValueError("invalid transformer depth or width")
        if self.decoder_layers < 1 or self.decoder_layers > 3:
            raise ValueError("decoder_layers must be in [1, 3]")


class IDOnlyCodec:
    """Public state + card IDs; excludes all hand-authored card semantics."""

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

    def __init__(self, config: ModelConfig):
        self.config = config

    @staticmethod
    def _status_bits(state: dict[str, Any]) -> int:
        names = ("asleep", "burned", "confused", "paralyzed", "poisoned")
        return sum((1 << index) for index, name in enumerate(names) if bool(state.get(name)))

    def encode(self, obs: dict[str, Any], action: list[int]) -> dict[str, Any] | None:
        current = getv(obs, "current", {}) or {}
        select = getv(obs, "select", {}) or {}
        options = as_list(select.get("option"))
        if not options or not valid_action(action, select):
            return None
        actor = as_int(current.get("yourIndex"), -1)
        if actor not in (0, 1):
            return None
        opponent = 1 - actor
        own, opp = player(current, actor), player(current, opponent)
        flags = 0
        for index, name in enumerate(("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated", "turnEnd")):
            if bool(current.get(name)):
                flags |= 1 << index
        own_prize = len(as_list(own.get("prize")))
        opp_prize = len(as_list(opp.get("prize")))
        global_cat = [
            min(max(as_int(select.get("type"), -1) + 1, 0), 64),
            min(max(as_int(select.get("context"), -1) + 1, 0), 128),
            min(max(as_int(current.get("firstPlayer"), -1) + 1, 0), 3),
            min(flags, 63),
        ]
        global_num = [
            min(max(as_int(current.get("turn"), 0), 0) / 20.0, 2.0),
            min(max(as_int(current.get("turnActionCount"), 0), 0) / 50.0, 2.0),
            min(max(as_int(own.get("deckCount"), 0), 0) / 60.0, 1.0),
            min(max(as_int(opp.get("deckCount"), 0), 0) / 60.0, 1.0),
            min(max(as_int(own.get("handCount"), len(as_list(own.get("hand")))), 0) / 20.0, 2.0),
            min(max(as_int(opp.get("handCount"), len(as_list(opp.get("hand")))), 0) / 20.0, 2.0),
            min(own_prize / 6.0, 1.0),
            min(opp_prize / 6.0, 1.0),
            min(len(options) / float(self.config.max_options), 2.0),
            min(max(as_int(select.get("remainDamageCounter"), 0), 0) / 300.0, 2.0),
            min(max(as_int(select.get("remainEnergyCost"), 0), 0) / 10.0, 2.0),
            min((len(as_list(own.get("bench"))) + len(as_list(opp.get("bench")))) / 10.0, 1.0),
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
            raw_id = card_id(card)
            if raw_id <= 0:
                return -1
            cid = min(raw_id, self.config.max_card_id)
            hp = as_float(getv(card, "hp", 0))
            max_hp = max(hp, as_float(getv(card, "maxHp", hp)))
            damage = max(0.0, max_hp - hp)
            energy = as_list(getv(card, "energyCards", getv(card, "energies", [])))
            tools = as_list(getv(card, "tools", []))
            evolution = as_list(getv(card, "preEvolution", []))
            index = len(entities)
            entities.append([cid, owner, zone, min(slot + 1, 64), kind, min(status, 31), max(parent + 1, 0)])
            entity_num.append([
                min(damage / 400.0, 2.0),
                min(len(energy) / 10.0, 2.0),
                min(len(tools) / 4.0, 2.0),
                min(len(evolution) / 4.0, 2.0),
                1.0 if bool(getv(card, "appearThisTurn", getv(card, "appear", False))) else 0.0,
            ])
            if key is not None:
                location[key] = index
            return index

        def add_zone(player_index: int, name: str, area: int, own_zone: str, opp_zone: str) -> None:
            state = player(current, player_index)
            owner = self.OWNER_SELF if player_index == actor else self.OWNER_OPP
            zone = self.ZONES[own_zone if owner == self.OWNER_SELF else opp_zone]
            active_status = self._status_bits(state) if name == "active" else 0
            for slot, card in enumerate(as_list(state.get(name))):
                parent = add_card(card, owner, zone, slot, 2 if name in ("active", "bench") else 1, active_status, key=(player_index, area, slot))
                if parent < 0 or name not in ("active", "bench"):
                    continue
                for child_slot, child in enumerate(as_list(getv(card, "energyCards", getv(card, "energies", [])))):
                    add_card(child, owner, self.ZONES["own_energy" if owner == self.OWNER_SELF else "opp_energy"], child_slot, 3, parent=parent)
                for child_slot, child in enumerate(as_list(getv(card, "tools", []))):
                    add_card(child, owner, self.ZONES["own_tool" if owner == self.OWNER_SELF else "opp_tool"], child_slot, 4, parent=parent)
                for child_slot, child in enumerate(as_list(getv(card, "preEvolution", []))):
                    add_card(child, owner, self.ZONES["own_evolution" if owner == self.OWNER_SELF else "opp_evolution"], child_slot, 5, parent=parent)

        for player_index in (actor, opponent):
            add_zone(player_index, "active", 4, "own_active", "opp_active")
            add_zone(player_index, "bench", 5, "own_bench", "opp_bench")
            if player_index == actor:
                add_zone(player_index, "hand", 2, "own_hand", "own_hand")
            add_zone(player_index, "discard", 3, "own_discard", "opp_discard")
        for slot, card in enumerate(as_list(current.get("stadium"))):
            add_card(card, self.OWNER_NONE, self.ZONES["stadium"], slot, 6, key=(-1, 7, slot))
        for slot, card in enumerate(as_list(current.get("looking"))):
            add_card(card, self.OWNER_SELF, self.ZONES["looking"], slot, 7, key=(actor, 12, slot))
        for slot, card in enumerate(as_list(select.get("deck"))):
            add_card(card, self.OWNER_SELF, self.ZONES["select_deck"], slot, 1, key=(actor, 1, slot))
        if len(entities) > self.config.max_entities:
            return None

        option_rows: list[list[int]] = []
        for option_index, option in enumerate(options):
            if not isinstance(option, dict):
                option = {}
            source_player = as_int(option.get("playerIndex"), actor)
            source_area = as_int(option.get("area"), -1)
            if as_int(option.get("type"), -1) == 7 and source_area < 0:
                source_area = 2
            source_slot = as_int(option.get("index"), -1)
            source_entity = location.get((source_player, source_area, source_slot), -1)
            if source_area == 7:
                source_entity = location.get((-1, 7, source_slot), source_entity)
            source_card = as_int(option.get("cardId"), 0)
            if source_card <= 0 and source_entity >= 0:
                source_card = entities[source_entity][0]
            target_area = as_int(option.get("inPlayArea"), -1)
            target_slot = as_int(option.get("inPlayIndex"), -1)
            target_player = as_int(option.get("inPlayPlayerIndex", option.get("targetPlayerIndex", source_player)), actor)
            target_entity = location.get((target_player, target_area, target_slot), -1)
            target_card = entities[target_entity][0] if target_entity >= 0 else 0
            owner = self.OWNER_SELF if source_player == actor else self.OWNER_OPP if source_player in (0, 1) else self.OWNER_NONE
            option_rows.append([
                min(max(as_int(option.get("type"), -1) + 1, 0), 64),
                min(max(source_area + 1, 0), 32),
                min(max(target_area + 1, 0), 32),
                owner,
                min(max(source_card, 0), self.config.max_card_id),
                min(max(target_card, 0), self.config.max_card_id),
                min(max(as_int(option.get("number"), -1) + 1, 0), 128),
                min(max(source_slot + 1, 0), 128),
                min(max(target_slot + 1, 0), 128),
                source_entity + 1,
                target_entity + 1,
                min(option_index + 1, self.config.max_options),
            ])
        if len(option_rows) > self.config.max_options or len(action) > self.config.max_action_steps:
            return None
        return {
            "global_cat": global_cat,
            "global_num": global_num,
            "entity_cat": entities,
            "entity_num": entity_num,
            "option_cat": option_rows,
            "action": action,
            "min_count": min(max(as_int(select.get("minCount"), 0), 0), self.config.max_action_steps),
            "max_count": min(max(as_int(select.get("maxCount"), 0), len(action)), self.config.max_action_steps),
        }


class IDOnlyPointerPolicy(nn.Module):
    """State encoder plus a configurable stacked selected-option GRU decoder."""

    def __init__(self, config: ModelConfig):
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers, norm=nn.LayerNorm(d))
        self.option_type = nn.Embedding(65, d)
        self.area = nn.Embedding(33, d)
        self.number = nn.Embedding(129, d)
        self.option_owner = nn.Embedding(3, d)
        self.option_position = nn.Embedding(config.max_options + 1, d)
        self.option_norm = nn.LayerNorm(d)
        self.option_to_state = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.option_state_norm = nn.LayerNorm(d)
        self.option_ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(2 * d, d))
        self.option_ff_norm = nn.LayerNorm(d)
        self.decoder_init = nn.Linear(d, d * config.decoder_layers)
        self.decoder_cells = nn.ModuleList(nn.GRUCell(d, d) for _ in range(config.decoder_layers))
        self.pointer_query = nn.Linear(d, d, bias=False)
        self.pointer_key = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1)
        self.stop = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    @staticmethod
    def _gather_entities(entities: Tensor, refs: Tensor) -> Tensor:
        safe = (refs - 1).clamp_min(0)
        gathered = entities.gather(1, safe.unsqueeze(-1).expand(-1, -1, entities.size(-1)))
        return gathered * refs.gt(0).unsqueeze(-1)

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
        g = batch["global_cat"]
        global_repr = self.global_norm(
            self.select_type(g[:, 0])
            + self.select_context(g[:, 1])
            + self.first_player(g[:, 2])
            + self.flags(g[:, 3])
            + self.global_num(batch["global_num"])
        )
        cls = self.cls.expand(entity.size(0), -1, -1) + global_repr.unsqueeze(1)
        sequence = torch.cat([cls, entity_repr], dim=1)
        padding = torch.cat([torch.zeros((entity.size(0), 1), dtype=torch.bool, device=entity.device), ~batch["entity_mask"]], dim=1)
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
        attended, _ = self.option_to_state(option_repr, encoded, encoded, key_padding_mask=padding, need_weights=False)
        option_repr = self.option_state_norm(option_repr + attended)
        option_repr = self.option_ff_norm(option_repr + self.option_ff(option_repr))
        return state_repr, option_repr

    def decode_step(self, selected: Tensor, hidden: Tensor) -> Tensor:
        """Advance a stacked GRU using the teacher-selected option embedding."""
        layer_input = selected
        next_hidden: list[Tensor] = []
        for index, cell in enumerate(self.decoder_cells):
            layer_output = cell(layer_input, hidden[:, index])
            next_hidden.append(layer_output)
            layer_input = layer_output
        return torch.stack(next_hidden, dim=1)

    def teacher_logits(self, batch: dict[str, Tensor]) -> Tensor:
        state, options = self.encode(batch)
        batch_size, option_count, _ = options.shape
        keys = self.pointer_key(options)
        hidden = torch.tanh(self.decoder_init(state)).view(batch_size, self.config.decoder_layers, self.config.d_model)
        chosen = torch.zeros((batch_size, option_count), dtype=torch.bool, device=options.device)
        outputs: list[Tensor] = []
        for step in range(batch["targets"].size(1)):
            pointer = (self.pointer_query(hidden[:, -1]).unsqueeze(1) * keys).sum(-1) / math.sqrt(self.config.d_model)
            pointer = pointer + self.option_bias(options).squeeze(-1)
            pointer = pointer.masked_fill(~batch["option_mask"] | chosen, torch.finfo(pointer.dtype).min)
            stop = self.stop(hidden[:, -1])
            stop = stop.masked_fill((step < batch["min_count"]).unsqueeze(-1), torch.finfo(stop.dtype).min)
            logits = torch.cat([pointer, stop], dim=1)
            outputs.append(logits)
            target = batch["targets"][:, step]
            valid = (target >= 0) & (target < option_count)
            if valid.any():
                safe = target.clamp(min=0, max=option_count - 1)
                selected = options[torch.arange(batch_size, device=options.device), safe]
                next_hidden = self.decode_step(selected, hidden)
                hidden = torch.where(valid.view(-1, 1, 1), next_hidden, hidden)
                chosen.scatter_(1, safe.unsqueeze(1), chosen.gather(1, safe.unsqueeze(1)) | valid.unsqueeze(1))
        return torch.stack(outputs, dim=1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.teacher_logits(batch)


def collate_examples(rows: list[dict[str, Any]]) -> dict[str, Tensor]:
    batch_size = len(rows)
    # A legal selection can expose no card entity. Keep one masked sentinel so
    # entity-reference gathers remain well-defined for those rows.
    entity_count = max(1, max(len(row["entity_cat"]) for row in rows))
    option_count = max(1, max(len(row["option_cat"]) for row in rows))
    target_steps = max(len(row["action"]) + 1 for row in rows)
    entity_cat = np.zeros((batch_size, entity_count, 7), dtype=np.int64)
    entity_num = np.zeros((batch_size, entity_count, 5), dtype=np.float32)
    entity_mask = np.zeros((batch_size, entity_count), dtype=np.bool_)
    option_cat = np.zeros((batch_size, option_count, 12), dtype=np.int64)
    option_mask = np.zeros((batch_size, option_count), dtype=np.bool_)
    targets = np.full((batch_size, target_steps), -100, dtype=np.int64)
    global_cat = np.zeros((batch_size, 4), dtype=np.int64)
    global_num = np.zeros((batch_size, 12), dtype=np.float32)
    min_count = np.zeros(batch_size, dtype=np.int64)
    for index, row in enumerate(rows):
        entities = len(row["entity_cat"])
        options = len(row["option_cat"])
        if entities:
            entity_cat[index, :entities] = np.asarray(row["entity_cat"], dtype=np.int64)
            entity_num[index, :entities] = np.asarray(row["entity_num"], dtype=np.float32)
            entity_mask[index, :entities] = True
        if options:
            option_cat[index, :options] = np.asarray(row["option_cat"], dtype=np.int64)
            option_mask[index, :options] = True
        action = list(row["action"])
        targets[index, : len(action)] = np.asarray(action, dtype=np.int64)
        targets[index, len(action)] = option_count
        global_cat[index] = np.asarray(row["global_cat"], dtype=np.int64)
        global_num[index] = np.asarray(row["global_num"], dtype=np.float32)
        min_count[index] = int(row["min_count"])
    return {
        "global_cat": torch.from_numpy(global_cat),
        "global_num": torch.from_numpy(global_num),
        "entity_cat": torch.from_numpy(entity_cat),
        "entity_num": torch.from_numpy(entity_num),
        "entity_mask": torch.from_numpy(entity_mask),
        "option_cat": torch.from_numpy(option_cat),
        "option_mask": torch.from_numpy(option_mask),
        "targets": torch.from_numpy(targets),
        "min_count": torch.from_numpy(min_count),
    }


class JsonlShardDataset(IterableDataset[dict[str, Any]]):
    def __init__(self, paths: list[Path], *, shuffle: bool, seed: int, buffer_size: int = 4096):
        self.paths = paths
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed + self.epoch)
        paths = list(self.paths)
        if self.shuffle:
            rng.shuffle(paths)
        buffer: list[dict[str, Any]] = []
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not self.shuffle:
                        yield row
                        continue
                    buffer.append(row)
                    if len(buffer) >= self.buffer_size:
                        yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer))) if self.shuffle else buffer.pop(0)


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def unwrap_model(model: nn.Module) -> IDOnlyPointerPolicy:
    """Return portable weights when training uses CUDA DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model  # type: ignore[return-value]


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def episode_files(root: Path, dates: set[str]) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if episode_date(path) in dates:
            yield path


def stable_split(episode_id: str, valid_dates: set[str], date: str) -> str:
    if date in valid_dates:
        return "valid"
    return "train"


def write_rows(handle: Any, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        count += 1
    return count


def count_jsonl_gz(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def find_preencoded_dataset(input_root: Path) -> dict[str, Any]:
    candidates: list[tuple[Path, Path]] = []
    for train_path in input_root.rglob("train_000.jsonl.gz"):
        valid_path = train_path.with_name("valid_000.jsonl.gz")
        if valid_path.is_file():
            candidates.append((train_path, valid_path))
    if not candidates:
        raise FileNotFoundError(
            "No pre-encoded v1 data found. Attach the completed "
            "horizen12/ptcg-yushin-id-only-bc-v1 kernel as a source."
        )
    candidates.sort(key=lambda pair: ("ptcg_yushin_idonly_bc_v1_output" not in str(pair[0]), str(pair[0])))
    train_path, valid_path = candidates[0]
    return {
        "source": "mounted_v1_preencoded_output",
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "decisions_train": count_jsonl_gz(train_path),
        "decisions_valid": count_jsonl_gz(valid_path),
    }



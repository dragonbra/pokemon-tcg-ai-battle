from __future__ import annotations

import math
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pure_policy_codec_v1 import (
    ENTITY_CAT_DIM,
    ENTITY_NUM_DIM,
    FAMILY_NAMES,
    GLOBAL_CAT_DIM,
    GLOBAL_NUM_DIM,
    OPTION_CAT_DIM,
    OPTION_NUM_DIM,
    OWNER_OPP,
    OWNER_SELF,
    PolicyCodecV1,
    ZONE_OPP_ACTIVE,
    ZONE_OPP_BENCH,
    ZONE_OPP_DISCARD,
    ZONE_OPP_ENERGY,
    ZONE_OPP_EVOLUTION,
    ZONE_OPP_TOOL,
    ZONE_OWN_ACTIVE,
    ZONE_OWN_BENCH,
    ZONE_OWN_DISCARD,
    ZONE_OWN_ENERGY,
    ZONE_OWN_HAND,
    ZONE_OWN_PRIZE,
    normalize_model_action,
)
from card_interaction_semantics import (
    ATTACHED_PREDICATE_MASK,
    ATTACK_CHANNEL_BITS,
    INTERACTION_PREDICATE_BITS,
    INTERACTION_GATE_GROUP_FEATURES,
    PROTECTION_MODE_BLOCK,
    PROTECTION_MODE_REDUCE,
    PROTECTION_SCOPE_IDS,
    RELATION_FEATURE_NAMES,
    load_compiled_interaction_index,
)
from card_rule_targets import BLOCK_REASON_NAMES, DAMAGE_BUCKET_EDGES, IN_PLAY_ZONES


MODEL_VERSION = "entity_pointer_policy_v1"
OPTIONAL_AUXILIARY_HEAD_PREFIXES = (
    "kill_line_head.",
    "safe_draw_head.",
    "future_family_head.",
    "attack_interaction_head.",
    "rule_aux_",
)
# Zone summaries are an optional architectural branch.  Keeping their state
# keys optional lets an old V1/V4 checkpoint load unchanged, while an upgraded
# config can initialize the branch from the same shared backbone.
OPTIONAL_ZONE_SUMMARY_PREFIXES = (
    "zone_summary_",
)
OPTIONAL_CARD_RULE_PREFIXES = (
    "card_rule_",
    "attack_rule_",
    "rule_semantic_",
    "rule_modifier_",
    "rule_relation_",
    "rule_aux_",
)
# Eight stable summaries, deliberately matching the public state abstractions
# used by the experiment plan.  The opponent-board summary pools its public
# active/bench/energy/tool/evolution entities into one token.
ZONE_SUMMARY_GROUPS: tuple[tuple[int, ...], ...] = (
    (ZONE_OWN_ACTIVE,),
    (ZONE_OWN_BENCH,),
    (ZONE_OWN_HAND,),
    (ZONE_OWN_DISCARD,),
    (ZONE_OWN_PRIZE,),
    (ZONE_OWN_ENERGY,),
    (ZONE_OPP_ACTIVE, ZONE_OPP_BENCH, ZONE_OPP_ENERGY, ZONE_OPP_TOOL, ZONE_OPP_EVOLUTION),
    (ZONE_OPP_DISCARD,),
)
ZONE_SUMMARY_NAMES = (
    "own_active",
    "own_bench",
    "own_hand",
    "own_discard",
    "own_prizes",
    "own_energy",
    "opponent_board",
    "opponent_discard",
)
ZONE_SUMMARY_IMPLEMENTATION = "zone_latent_attention_pool_v1"
# Keep V4 -> zone-summary migration deterministic.  The forked RNG means
# attaching the branch does not perturb the caller's training/random stream,
# and candidate/reference models receive identical new zone parameters.
ZONE_SUMMARY_INIT_SEED = 2026072101
CARD_SEMANTIC_INIT_SEED = 2026072102
CARD_INTERACTION_INIT_SEED = 2026072201
CARD_RULE_SEMANTIC_INIT_SEED = 2026072202
HISTORY_KEYS = (
    "history_family",
    "history_source_card",
    "history_target_card",
    "history_attack_id",
    "history_turn",
    "history_own_prizes",
    "history_opp_prizes",
)


def collate_history_records(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate optional action history without rebuilding the full policy batch."""
    if not rows or not any(any(row.get(key) for key in HISTORY_KEYS) for row in rows):
        return {}
    batch_size = len(rows)
    max_history = max(1, max(len(list(row.get("history_family") or [])) for row in rows))
    history_arrays = {
        key: np.zeros((batch_size, max_history), dtype=np.int64)
        for key in HISTORY_KEYS
    }
    history_mask = np.zeros((batch_size, max_history), dtype=np.bool_)
    for row_index, row in enumerate(rows):
        length = min(max_history, len(list(row.get("history_family") or [])))
        if not length:
            continue
        start = max_history - length
        for key in HISTORY_KEYS:
            values = list(row.get(key) or [])[-length:]
            if len(values) != length:
                raise ValueError(f"history field {key} has inconsistent length")
            history_arrays[key][row_index, start:] = np.asarray(values, dtype=np.int64)
        history_mask[row_index, start:] = True
    batch = {key: torch.from_numpy(value) for key, value in history_arrays.items()}
    batch["history_mask"] = torch.from_numpy(history_mask)
    return batch


@dataclass
class EntityPointerConfig:
    max_card_id: int = 4096
    max_attack_id: int = 4096
    d_model: int = 512
    heads: int = 8
    encoder_layers: int = 8
    decoder_layers: int = 2
    dropout: float = 0.05
    opponent_classes: int = 16
    finish_reason_classes: int = 32
    card_semantic_vocab_size: int = 0
    card_semantic_max_features: int = 0
    card_semantic_total_features: int = 0
    card_semantic_gate_classes: int = 1
    card_interaction_relation_features: int = 0
    card_interaction_gate_classes: int = 1
    card_interaction_split_gates: bool = False
    # Version 1 is the legacy seven-card bitmask branch. Version 2 is the
    # generated full-pool attack/protection/target relation runtime.
    card_interaction_runtime_version: int = 1
    card_rule_semantic_vocab_size: int = 0
    card_rule_card_total_features: int = 0
    card_rule_attack_total_features: int = 0
    card_rule_gate_classes: int = 1
    # Zone summaries are opt-in so legacy checkpoints retain their exact
    # sequence layout.  `zone_summary_gate_classes` mirrors semantic gates and
    # is normally set to num_cores for deck-conditioned policies.
    zone_summary_enabled: bool = False
    zone_summary_gate_classes: int = 1

    def validate(self) -> None:
        if self.d_model % self.heads:
            raise ValueError(f"d_model={self.d_model} must be divisible by heads={self.heads}")
        if self.decoder_layers < 1:
            raise ValueError("decoder_layers must be positive")
        semantic_dims = (
            self.card_semantic_vocab_size,
            self.card_semantic_max_features,
            self.card_semantic_total_features,
        )
        if any(value < 0 for value in semantic_dims):
            raise ValueError("card semantic dimensions must be non-negative")
        if any(semantic_dims) and not all(semantic_dims):
            raise ValueError("card semantic dimensions must be all zero or positive")
        if self.card_semantic_gate_classes < 1:
            raise ValueError("card semantic gate classes must be positive")
        if self.card_interaction_relation_features < 0:
            raise ValueError("card interaction relation features must be non-negative")
        if self.card_interaction_gate_classes < 1:
            raise ValueError("card interaction gate classes must be positive")
        if self.card_interaction_runtime_version not in {1, 2}:
            raise ValueError("unsupported card interaction runtime version")
        if self.card_interaction_split_gates and (
            self.card_interaction_runtime_version != 2
            or self.card_interaction_relation_features <= 0
        ):
            raise ValueError("split interaction gates require the full-pool runtime")
        rule_dims = (
            self.card_rule_semantic_vocab_size,
            self.card_rule_card_total_features,
            self.card_rule_attack_total_features,
        )
        if any(value < 0 for value in rule_dims):
            raise ValueError("card rule semantic dimensions must be non-negative")
        if any(rule_dims) and not all(rule_dims):
            raise ValueError("card rule semantic dimensions must be all zero or positive")
        if self.card_rule_gate_classes < 1:
            raise ValueError("card rule semantic gate classes must be positive")
        if self.zone_summary_gate_classes < 1:
            raise ValueError("zone summary gate classes must be positive")


def load_card_semantic_index(
    path: str | Path,
    *,
    max_card_id: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build a compact card-id -> semantic-feature index from the generated card map."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cards = payload.get("cards") or {}
    labels = sorted({str(label) for values in cards.values() for label in values})
    label_to_id = {label: index + 1 for index, label in enumerate(labels)}
    max_features = max((len(values) for values in cards.values()), default=0)
    index = torch.zeros((max_card_id + 1, max_features), dtype=torch.long)
    mask = torch.zeros((max_card_id + 1, max_features), dtype=torch.bool)
    for raw_card_id, values in cards.items():
        card_id = int(raw_card_id)
        if not 0 <= card_id <= max_card_id:
            continue
        feature_ids = [label_to_id[str(value)] for value in values]
        if feature_ids:
            index[card_id, : len(feature_ids)] = torch.tensor(feature_ids, dtype=torch.long)
            mask[card_id, : len(feature_ids)] = True
    metadata = {
        "version": str(payload.get("version", "")),
        "source_csv": str(payload.get("source_csv", "")),
        "card_count": len(cards),
        "feature_count": len(labels),
        "max_features_per_card": max_features,
        "total_feature_assignments": int(mask.sum().item()),
    }
    return index, mask, metadata


def load_card_rule_semantic_index(
    path: str | Path,
    *,
    max_card_id: int,
    max_attack_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build shared compact card/attack rule-atom indexes.

    Empty IDs receive the padding atom so EmbeddingBag returns an exact zero
    without borrowing the following row's first feature.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cards = dict(payload.get("cards") or {})
    attacks = dict(payload.get("attacks") or {})
    labels = sorted(
        {
            str(label)
            for mapping in (cards, attacks)
            for values in mapping.values()
            for label in values
        }
    )
    label_to_id = {label: index + 1 for index, label in enumerate(labels)}

    def compact(
        mapping: dict[str, list[str]], max_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        values: list[int] = []
        offsets = torch.zeros(max_id + 2, dtype=torch.long)
        assignments = 0
        for item_id in range(max_id + 1):
            item_values = [
                label_to_id[str(label)]
                for label in mapping.get(str(item_id), [])
                if str(label) in label_to_id
            ]
            assignments += len(item_values)
            values.extend(item_values or [0])
            offsets[item_id + 1] = len(values)
        return torch.tensor(values, dtype=torch.long), offsets, assignments

    card_flat, card_offsets, card_assignments = compact(cards, max_card_id)
    attack_flat, attack_offsets, attack_assignments = compact(attacks, max_attack_id)
    metadata = {
        "version": str(payload.get("version", "")),
        "card_count": len(cards),
        "attack_count": len(attacks),
        "feature_count": len(labels),
        "card_feature_assignments": card_assignments,
        "attack_feature_assignments": attack_assignments,
        "statistics": dict(payload.get("statistics") or {}),
    }
    return card_flat, card_offsets, attack_flat, attack_offsets, metadata


def collate_decision_records(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("cannot collate an empty decision batch")
    batch_size = len(rows)
    entity_lengths: list[int] = []
    option_lengths: list[int] = []
    min_counts: list[int] = []
    max_counts: list[int] = []
    for row in rows:
        global_cat_values = row["global_cat"]
        global_num_values = row["global_num"]
        entity_parent_values = row["entity_parent"]
        option_equiv_values = row["option_equiv"]
        entity_count = len(entity_parent_values)
        option_count = len(option_equiv_values)
        if len(global_cat_values) != GLOBAL_CAT_DIM or len(global_num_values) != GLOBAL_NUM_DIM:
            raise ValueError("bad global feature width")
        if len(row["entity_cat"]) != entity_count * ENTITY_CAT_DIM:
            raise ValueError("bad flattened entity categorical width")
        if len(row["entity_num"]) != entity_count * ENTITY_NUM_DIM:
            raise ValueError("bad flattened entity numeric width")
        if len(row["option_cat"]) != option_count * OPTION_CAT_DIM:
            raise ValueError("bad flattened option categorical width")
        if len(row["option_num"]) != option_count * OPTION_NUM_DIM:
            raise ValueError("bad flattened option numeric width")
        min_count = int(row["min_count"])
        max_count = int(row["max_count"])
        if min_count < 0 or max_count < min_count:
            raise ValueError(f"invalid select bounds {min_count}/{max_count}")
        entity_lengths.append(entity_count)
        option_lengths.append(option_count)
        min_counts.append(min_count)
        max_counts.append(max_count)

    max_entities = max(1, max(entity_lengths))
    max_options = max(option_lengths)
    max_action = max(
        1,
        max(
            max(len(list(row.get("teacher_action") or [])), min_counts[index])
            for index, row in enumerate(rows)
        ),
    )

    actions_by_row = [
        normalize_model_action(
            list(row.get("teacher_action") or []),
            option_lengths[index],
            min_counts[index],
            max_counts[index],
        )
        for index, row in enumerate(rows)
    ]
    action_lengths = [len(action) for action in actions_by_row]

    global_cat_array = np.asarray([row["global_cat"] for row in rows], dtype=np.int64)
    global_num_array = np.asarray([row["global_num"] for row in rows], dtype=np.float32)
    entity_cat_array = np.zeros((batch_size, max_entities, ENTITY_CAT_DIM), dtype=np.int64)
    entity_num_array = np.zeros((batch_size, max_entities, ENTITY_NUM_DIM), dtype=np.float32)
    entity_parent_array = np.full((batch_size, max_entities), -1, dtype=np.int64)
    entity_mask_array = np.zeros((batch_size, max_entities), dtype=np.bool_)
    option_cat_array = np.zeros((batch_size, max_options, OPTION_CAT_DIM), dtype=np.int64)
    option_num_array = np.zeros((batch_size, max_options, OPTION_NUM_DIM), dtype=np.float32)
    option_equiv_array = np.full((batch_size, max_options), -1, dtype=np.int64)
    option_mask_array = np.zeros((batch_size, max_options), dtype=np.bool_)
    actions_array = np.full((batch_size, max_action), -1, dtype=np.int64)

    for index, row in enumerate(rows):
        entity_count = entity_lengths[index]
        if entity_count:
            entity_cat_array[index, :entity_count] = np.asarray(row["entity_cat"], dtype=np.int64).reshape(
                entity_count, ENTITY_CAT_DIM
            )
            entity_num_array[index, :entity_count] = np.asarray(row["entity_num"], dtype=np.float32).reshape(
                entity_count, ENTITY_NUM_DIM
            )
            entity_parent_array[index, :entity_count] = row["entity_parent"]
            entity_mask_array[index, :entity_count] = True
        option_count = option_lengths[index]
        if option_count:
            option_cat_array[index, :option_count] = np.asarray(row["option_cat"], dtype=np.int64).reshape(
                option_count, OPTION_CAT_DIM
            )
            option_num_array[index, :option_count] = np.asarray(row["option_num"], dtype=np.float32).reshape(
                option_count, OPTION_NUM_DIM
            )
            option_equiv_array[index, :option_count] = row["option_equiv"]
            option_mask_array[index, :option_count] = True
        action_count = action_lengths[index]
        if action_count:
            actions_array[index, :action_count] = actions_by_row[index]

    global_cat = torch.from_numpy(global_cat_array)
    global_num = torch.from_numpy(global_num_array)
    entity_cat = torch.from_numpy(entity_cat_array)
    entity_num = torch.from_numpy(entity_num_array)
    entity_parent = torch.from_numpy(entity_parent_array)
    entity_mask = torch.from_numpy(entity_mask_array)
    option_cat = torch.from_numpy(option_cat_array)
    option_num = torch.from_numpy(option_num_array)
    option_equiv = torch.from_numpy(option_equiv_array)
    option_mask = torch.from_numpy(option_mask_array)
    actions = torch.from_numpy(actions_array)
    action_len = torch.from_numpy(np.asarray(action_lengths, dtype=np.int64))
    min_count = torch.from_numpy(np.asarray(min_counts, dtype=np.int64))
    max_count = torch.from_numpy(np.asarray(max_counts, dtype=np.int64))
    value_target = torch.from_numpy(
        np.asarray([float(row.get("value_target", 0.0) or 0.0) for row in rows], dtype=np.float32)
    )
    next_prize_target = torch.from_numpy(
        np.asarray([float(row.get("next_prize_target", 0.0) or 0.0) for row in rows], dtype=np.float32)
    )
    finish_reason = torch.from_numpy(
        np.asarray([max(0, int(row.get("finish_reason", 0) or 0)) for row in rows], dtype=np.int64)
    )
    opponent_class = torch.from_numpy(
        np.asarray([max(0, int(row.get("opponent_class", 0) or 0)) for row in rows], dtype=np.int64)
    )
    action_family = torch.from_numpy(
        np.asarray([int(row.get("action_family", 0)) for row in rows], dtype=np.int64)
    )
    action_weight = torch.from_numpy(
        np.asarray(
            [max(0.0, float(row.get("action_weight", 1.0) or 0.0)) for row in rows],
            dtype=np.float32,
        )
    )
    value_weight = torch.from_numpy(
        np.asarray(
            [max(0.0, float(row.get("value_weight", 1.0) or 0.0)) for row in rows],
            dtype=np.float32,
        )
    )

    batch = {
        "global_cat": global_cat,
        "global_num": global_num,
        "entity_cat": entity_cat,
        "entity_num": entity_num,
        "entity_parent": entity_parent,
        "entity_mask": entity_mask,
        "option_cat": option_cat,
        "option_num": option_num,
        "option_equiv": option_equiv,
        "option_mask": option_mask,
        "actions": actions,
        "action_len": action_len,
        "min_count": min_count,
        "max_count": max_count,
        "value_target": value_target,
        "next_prize_target": next_prize_target,
        "finish_reason": finish_reason,
        "opponent_class": opponent_class,
        "action_family": action_family,
        "action_weight": action_weight,
        "value_weight": value_weight,
    }
    batch.update(collate_history_records(rows))
    return batch


def move_batch(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class EntityPointerPolicy(nn.Module):
    def __init__(self, config: EntityPointerConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model

        self.card_embedding = nn.Embedding(config.max_card_id + 1, d, padding_idx=0)
        if config.card_semantic_vocab_size > 0:
            self.card_semantic_embedding: nn.EmbeddingBag | None = nn.EmbeddingBag(
                config.card_semantic_vocab_size + 1,
                d,
                mode="mean",
                padding_idx=0,
            )
            gate_shape = () if config.card_semantic_gate_classes == 1 else (config.card_semantic_gate_classes,)
            self.semantic_gate = nn.Parameter(torch.zeros(gate_shape))
            self.register_buffer(
                "card_semantic_ids",
                torch.zeros(
                    (config.max_card_id + 1, config.card_semantic_max_features),
                    dtype=torch.long,
                ),
            )
            self.register_buffer(
                "card_semantic_mask",
                torch.zeros(
                    (config.max_card_id + 1, config.card_semantic_max_features),
                    dtype=torch.bool,
                ),
            )
            self.register_buffer(
                "card_semantic_flat_ids",
                torch.zeros(config.card_semantic_total_features, dtype=torch.long),
            )
            self.register_buffer(
                "card_semantic_offsets",
                torch.zeros(config.max_card_id + 2, dtype=torch.long),
            )
        else:
            self.card_semantic_embedding = None
        if config.card_interaction_relation_features > 0:
            expected_relation_features = (
                5
                if config.card_interaction_runtime_version == 1
                else len(RELATION_FEATURE_NAMES)
            )
            if config.card_interaction_relation_features != expected_relation_features:
                raise ValueError("card interaction relation feature count does not match runtime")
            self.interaction_relation_embedding: nn.Embedding | None = nn.Embedding(
                config.card_interaction_relation_features,
                d,
            )
            gate_shape = (
                ()
                if config.card_interaction_gate_classes == 1
                else (config.card_interaction_gate_classes,)
            )
            self.interaction_relation_gate = nn.Parameter(torch.zeros(gate_shape))
            if config.card_interaction_split_gates:
                self.interaction_block_gate = nn.Parameter(torch.zeros(gate_shape))
                self.interaction_future_gate = nn.Parameter(torch.zeros(gate_shape))
            else:
                self.interaction_block_gate = None
                self.interaction_future_gate = None
            if config.card_interaction_runtime_version == 1:
                self.register_buffer(
                    "card_interaction_traits",
                    torch.zeros(config.max_card_id + 1, dtype=torch.long),
                )
                self.register_buffer(
                    "attack_interaction_traits",
                    torch.zeros(config.max_attack_id + 1, dtype=torch.long),
                )
                self.interaction_relation_projection = None
                self.interaction_aux_effectiveness_head = None
                self.interaction_aux_block_head = None
                self.interaction_aux_future_head = None
            else:
                for name, size, dtype in (
                    ("card_interaction_predicates", config.max_card_id + 1, torch.long),
                    ("card_interaction_protection_channels", config.max_card_id + 1, torch.long),
                    ("card_interaction_protection_scope", config.max_card_id + 1, torch.long),
                    ("card_interaction_protection_mode", config.max_card_id + 1, torch.long),
                    ("card_interaction_protection_amount", config.max_card_id + 1, torch.float32),
                    ("card_interaction_source_predicates", config.max_card_id + 1, torch.long),
                    ("card_interaction_target_predicates", config.max_card_id + 1, torch.long),
                    ("card_interaction_protection_ambiguous", config.max_card_id + 1, torch.bool),
                    ("attack_interaction_channels", config.max_attack_id + 1, torch.long),
                    ("attack_interaction_future_mode", config.max_attack_id + 1, torch.long),
                ):
                    self.register_buffer(name, torch.zeros(size, dtype=dtype))
                self.interaction_relation_projection = nn.Sequential(
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.Linear(d, d),
                    nn.LayerNorm(d),
                )
                self.interaction_aux_effectiveness_head = nn.Linear(d, 1)
                self.interaction_aux_block_head = nn.Linear(d, 3)
                self.interaction_aux_future_head = (
                    nn.Linear(d, 2) if config.card_interaction_split_gates else None
                )
        else:
            self.interaction_relation_embedding = None
            self.interaction_relation_gate = None
            self.interaction_relation_projection = None
            self.interaction_aux_effectiveness_head = None
            self.interaction_aux_block_head = None
            self.interaction_aux_future_head = None
            self.interaction_block_gate = None
            self.interaction_future_gate = None
        self.interaction_semantics_enabled = True
        self.interaction_auxiliary_outputs_enabled = False
        self.interaction_semantic_diagnostics_enabled = False
        self.interaction_semantic_diagnostics_interval = 64
        self._interaction_semantic_diagnostic_calls = 0
        self._interaction_semantic_diagnostics: dict[str, float] = {}
        if config.card_rule_semantic_vocab_size > 0:
            self.rule_semantic_embedding: nn.EmbeddingBag | None = nn.EmbeddingBag(
                config.card_rule_semantic_vocab_size + 1,
                d,
                mode="mean",
                padding_idx=0,
            )
            gate_shape = (
                ()
                if config.card_rule_gate_classes == 1
                else (config.card_rule_gate_classes,)
            )
            self.rule_semantic_gate = nn.Parameter(torch.zeros(gate_shape))
            self.rule_modifier_gate = nn.Parameter(torch.zeros(gate_shape))
            self.rule_relation_gate = nn.Parameter(torch.zeros(gate_shape))
            self.register_buffer(
                "card_rule_flat_ids",
                torch.zeros(config.card_rule_card_total_features, dtype=torch.long),
            )
            self.register_buffer(
                "card_rule_offsets",
                torch.zeros(config.max_card_id + 2, dtype=torch.long),
            )
            self.register_buffer(
                "attack_rule_flat_ids",
                torch.zeros(config.card_rule_attack_total_features, dtype=torch.long),
            )
            self.register_buffer(
                "attack_rule_offsets",
                torch.zeros(config.max_attack_id + 2, dtype=torch.long),
            )
            self.rule_modifier_projection = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )
            self.rule_relation_projection = nn.Sequential(
                nn.Linear(4 * d, 2 * d),
                nn.GELU(),
                nn.Linear(2 * d, d),
                nn.LayerNorm(d),
            )
            self.rule_aux_effectiveness_head = nn.Linear(d, 1)
            self.rule_aux_block_reason_head = nn.Linear(d, len(BLOCK_REASON_NAMES))
            self.rule_aux_damage_bucket_head = nn.Linear(
                d, len(DAMAGE_BUCKET_EDGES) + 1
            )
            self.rule_aux_atom_head = nn.Linear(
                d, config.card_rule_semantic_vocab_size
            )
        else:
            self.rule_semantic_embedding = None
            self.rule_semantic_gate = None
            self.rule_modifier_gate = None
            self.rule_relation_gate = None
            self.rule_modifier_projection = None
            self.rule_relation_projection = None
            self.rule_aux_effectiveness_head = None
            self.rule_aux_block_reason_head = None
            self.rule_aux_damage_bucket_head = None
            self.rule_aux_atom_head = None
        self.rule_semantics_enabled = True
        self.rule_auxiliary_outputs_enabled = False
        self.rule_semantic_diagnostics_enabled = False
        self.rule_semantic_diagnostics_interval = 64
        self._rule_semantic_diagnostic_calls = 0
        self._rule_semantic_diagnostics: dict[str, dict[str, float]] = {}
        # semantic_enabled is deliberately runtime-only so an off/on ablation
        # can use the exact same checkpoint and architecture.
        self.semantic_enabled = True
        self.semantic_diagnostics_enabled = False
        self.semantic_diagnostics_interval = 64
        self._semantic_diagnostic_calls = 0
        self._semantic_diagnostics: dict[str, dict[str, float]] = {}
        # Zone summaries are an opt-in architectural branch.  Runtime
        # ablations can toggle this flag without changing the checkpoint.
        self.zone_summaries_enabled = bool(config.zone_summary_enabled)
        self.zone_summary_diagnostics_enabled = False
        self._zone_summary_diagnostics: dict[str, float] = {}
        self.attack_embedding = nn.Embedding(config.max_attack_id + 1, d, padding_idx=0)
        self.owner_embedding = nn.Embedding(4, d)
        self.zone_embedding = nn.Embedding(32, d)
        self.slot_embedding = nn.Embedding(65, d)
        self.kind_embedding = nn.Embedding(8, d)
        self.status_embedding = nn.Embedding(64, d)

        self.select_type_embedding = nn.Embedding(128, d)
        self.select_context_embedding = nn.Embedding(256, d)
        self.small_embedding = nn.Embedding(128, d)
        self.global_num_projection = nn.Sequential(nn.Linear(GLOBAL_NUM_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.entity_num_projection = nn.Sequential(nn.Linear(ENTITY_NUM_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.parent_projection = nn.Linear(d, d, bias=False)
        self.entity_norm = nn.LayerNorm(d)
        self.global_norm = nn.LayerNorm(d)

        if config.zone_summary_enabled:
            zone_count = len(ZONE_SUMMARY_GROUPS)
            gate_shape = () if config.zone_summary_gate_classes == 1 else (config.zone_summary_gate_classes,)
            self.zone_summary_gate = nn.Parameter(torch.zeros(gate_shape))
            self.zone_summary_query = nn.Parameter(torch.empty(zone_count, d))
            self.zone_summary_key = nn.Linear(d, d, bias=False)
            self.zone_summary_value = nn.Linear(d, d, bias=False)
            self.zone_summary_type_embedding = nn.Parameter(torch.zeros(zone_count, d))
            self.zone_summary_empty_embedding = nn.Parameter(torch.zeros(zone_count, d))
            self.zone_summary_norm = nn.LayerNorm(d)
            self.zone_summary_projection = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
            self.zone_summary_count_projection = nn.Sequential(
                nn.Linear(1, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
            group_by_zone = torch.full((32,), -1, dtype=torch.long)
            for group_index, zone_ids in enumerate(ZONE_SUMMARY_GROUPS):
                for zone_id in zone_ids:
                    group_by_zone[int(zone_id)] = group_index
            self.register_buffer("zone_summary_group_by_zone", group_by_zone)
            # Zero gate gives a safe old-policy initialization while retaining
            # a non-zero derivative for the dedicated zone probe.
            nn.init.xavier_uniform_(self.zone_summary_query)
        else:
            # Do not register placeholder parameters in legacy configs.  This
            # preserves old state dictionaries and exact sequence layouts.
            self.zone_summary_gate = None
            self.zone_summary_query = None
            self.zone_summary_key = None
            self.zone_summary_value = None
            self.zone_summary_type_embedding = None
            self.zone_summary_empty_embedding = None
            self.zone_summary_norm = None
            self.zone_summary_projection = None
            self.zone_summary_count_projection = None

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.heads,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers, norm=nn.LayerNorm(d))

        self.option_type_embedding = nn.Embedding(65, d)
        self.area_embedding = nn.Embedding(33, d)
        self.number_embedding = nn.Embedding(129, d)
        self.option_position_embedding = nn.Embedding(129, d)
        self.option_num_projection = nn.Sequential(nn.Linear(OPTION_NUM_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.option_fusion = nn.Sequential(
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )

        self.decoder_init = nn.ModuleList([nn.Linear(d, d) for _ in range(config.decoder_layers)])
        self.decoder_cells = nn.ModuleList([nn.GRUCell(d, d) for _ in range(config.decoder_layers)])
        self.pointer_query = nn.Linear(d, d, bias=False)
        self.pointer_key = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1)
        self.stop_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.next_prize_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.finish_reason_head = nn.Linear(d, config.finish_reason_classes)
        self.opponent_head = nn.Linear(d, config.opponent_classes)
        self.family_head = nn.Linear(d, len(FAMILY_NAMES))
        # Hindsight representation probes. These heads are auxiliary only and
        # do not alter the action distribution or dense reward.
        self.kill_line_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.safe_draw_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.future_family_head = nn.Linear(d, len(FAMILY_NAMES))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(CARD_INTERACTION_INIT_SEED + 1)
            self.attack_interaction_head = nn.Sequential(
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Linear(d // 2, 1),
            )

    def set_semantic_enabled(self, enabled: bool) -> None:
        self.semantic_enabled = bool(enabled)

    def set_interaction_semantics_enabled(self, enabled: bool) -> None:
        self.interaction_semantics_enabled = bool(enabled)

    def set_interaction_auxiliary_outputs_enabled(self, enabled: bool) -> None:
        self.interaction_auxiliary_outputs_enabled = bool(enabled)

    def reset_interaction_semantic_diagnostics(self) -> None:
        self._interaction_semantic_diagnostic_calls = 0
        self._interaction_semantic_diagnostics = {}

    @torch.no_grad()
    def _record_interaction_semantic_diagnostics(
        self,
        base: torch.Tensor,
        contribution: torch.Tensor,
        flags: torch.Tensor,
        valid: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        if not self.interaction_semantic_diagnostics_enabled:
            return
        call_index = self._interaction_semantic_diagnostic_calls
        self._interaction_semantic_diagnostic_calls += 1
        if call_index % max(1, int(self.interaction_semantic_diagnostics_interval)):
            return
        selected = valid.detach().bool()
        if not bool(selected.any()):
            return
        base_rows = base.detach().float()[selected]
        contribution_rows = contribution.detach().float()[selected]
        base_norm = base_rows.norm(dim=-1)
        contribution_norm = contribution_rows.norm(dim=-1)
        count = float(base_rows.size(0))
        elements = float(base_rows.numel())
        row = self._interaction_semantic_diagnostics
        row["rows"] = row.get("rows", 0.0) + count
        row["elements"] = row.get("elements", 0.0) + elements
        row["base_norm_sum"] = row.get("base_norm_sum", 0.0) + float(
            base_norm.sum().cpu()
        )
        row["contribution_norm_sum"] = row.get(
            "contribution_norm_sum", 0.0
        ) + float(contribution_norm.sum().cpu())
        row["ratio_sum"] = row.get("ratio_sum", 0.0) + float(
            (contribution_norm / base_norm.clamp_min(1.0e-12)).sum().cpu()
        )
        row["base_square_sum"] = row.get("base_square_sum", 0.0) + float(
            base_rows.square().sum().cpu()
        )
        row["contribution_square_sum"] = row.get(
            "contribution_square_sum", 0.0
        ) + float(contribution_rows.square().sum().cpu())
        row["active_flag_sum"] = row.get("active_flag_sum", 0.0) + float(
            flags.detach().float()[selected].sum().cpu()
        )
        row["active_flag_elements"] = row.get(
            "active_flag_elements", 0.0
        ) + float(flags.detach()[selected].numel())
        row["gate_abs_sum"] = row.get("gate_abs_sum", 0.0) + float(
            gate.detach().abs().sum().cpu()
        )
        row["gate_count"] = row.get("gate_count", 0.0) + float(gate.numel())

    def interaction_semantic_diagnostics_report(
        self, *, reset: bool = False
    ) -> dict[str, Any]:
        row = self._interaction_semantic_diagnostics
        count = max(row.get("rows", 0.0), 1.0)
        elements = max(row.get("elements", 0.0), 1.0)
        flag_elements = max(row.get("active_flag_elements", 0.0), 1.0)
        gate_count = max(row.get("gate_count", 0.0), 1.0)
        report = {
            "enabled": bool(self.interaction_semantics_enabled),
            "runtime_version": int(self.config.card_interaction_runtime_version),
            "rows": int(row.get("rows", 0.0)),
            "contribution_ratio_mean": row.get("ratio_sum", 0.0) / count,
            "base_norm_mean": row.get("base_norm_sum", 0.0) / count,
            "contribution_norm_mean": row.get("contribution_norm_sum", 0.0)
            / count,
            "base_activation_rms": math.sqrt(
                row.get("base_square_sum", 0.0) / elements
            ),
            "contribution_activation_rms": math.sqrt(
                row.get("contribution_square_sum", 0.0) / elements
            ),
            "active_flag_fraction": row.get("active_flag_sum", 0.0)
            / flag_elements,
            "gate_abs_mean": row.get("gate_abs_sum", 0.0) / gate_count,
        }
        if reset:
            self.reset_interaction_semantic_diagnostics()
        return report

    def set_rule_semantics_enabled(self, enabled: bool) -> None:
        self.rule_semantics_enabled = bool(enabled)

    def set_rule_auxiliary_outputs_enabled(self, enabled: bool) -> None:
        self.rule_auxiliary_outputs_enabled = bool(enabled)

    def reset_rule_semantic_diagnostics(self) -> None:
        self._rule_semantic_diagnostic_calls = 0
        self._rule_semantic_diagnostics = {}

    def _record_rule_semantic_diagnostics(
        self,
        name: str,
        base: torch.Tensor,
        contribution: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if not self.rule_semantic_diagnostics_enabled:
            return
        call_index = self._rule_semantic_diagnostic_calls
        self._rule_semantic_diagnostic_calls += 1
        if call_index % max(1, int(self.rule_semantic_diagnostics_interval)):
            return
        selected = valid.detach().bool()
        if not bool(selected.any()):
            return
        base_rows = base.detach().float()[selected]
        contribution_rows = contribution.detach().float()[selected]
        base_norm = base_rows.norm(dim=-1)
        contribution_norm = contribution_rows.norm(dim=-1)
        row = self._rule_semantic_diagnostics.setdefault(
            name,
            {
                "count": 0.0,
                "base_norm_sum": 0.0,
                "contribution_norm_sum": 0.0,
                "ratio_sum": 0.0,
                "base_square_sum": 0.0,
                "contribution_square_sum": 0.0,
                "elements": 0.0,
            },
        )
        count = float(base_rows.size(0))
        row["count"] += count
        row["base_norm_sum"] += float(base_norm.sum().cpu())
        row["contribution_norm_sum"] += float(contribution_norm.sum().cpu())
        row["ratio_sum"] += float(
            (contribution_norm / base_norm.clamp_min(1.0e-12)).sum().cpu()
        )
        row["base_square_sum"] += float(base_rows.square().sum().cpu())
        row["contribution_square_sum"] += float(
            contribution_rows.square().sum().cpu()
        )
        row["elements"] += float(base_rows.numel())

    def rule_semantic_diagnostics_report(self, *, reset: bool = False) -> dict[str, Any]:
        report: dict[str, Any] = {
            "enabled": bool(self.rule_semantics_enabled),
            "sample_interval": max(1, int(self.rule_semantic_diagnostics_interval)),
            "components": {},
        }
        for name, row in sorted(self._rule_semantic_diagnostics.items()):
            count = max(float(row["count"]), 1.0)
            elements = max(float(row["elements"]), 1.0)
            report["components"][name] = {
                "rows": int(row["count"]),
                "contribution_ratio_mean": row["ratio_sum"] / count,
                "base_norm_mean": row["base_norm_sum"] / count,
                "contribution_norm_mean": row["contribution_norm_sum"] / count,
                "base_activation_rms": math.sqrt(row["base_square_sum"] / elements),
                "contribution_activation_rms": math.sqrt(
                    row["contribution_square_sum"] / elements
                ),
            }
        if reset:
            self.reset_rule_semantic_diagnostics()
        return report

    def set_zone_summaries_enabled(self, enabled: bool) -> None:
        """Toggle zone latent tokens for a same-checkpoint ablation.

        Models created from a legacy config do not own the optional branch;
        those models remain safely off.  An enabled model can be switched off
        and on without rebuilding its state encoder or changing pointer
        indices.
        """
        if enabled and self.zone_summary_gate is None:
            raise RuntimeError("zone summaries are not present in this model config")
        self.zone_summaries_enabled = bool(enabled)

    def _conditioned_gate_values(
        self,
        gate: torch.Tensor,
        batch: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select a scalar/per-core gate without synchronizing fixed-core batches."""
        if gate.ndim == 0:
            return gate.expand(batch_size)
        fixed_core_id = getattr(self, "_fixed_core_id", None)
        if fixed_core_id is not None:
            # Deck-conditioned policies validate the core once when enabling
            # their fast path. Gate vocabularies may still be shared/smaller,
            # matching the legacy clamp behavior below.
            gate_index = min(int(fixed_core_id), gate.numel() - 1)
            return gate[gate_index].expand(batch_size)
        core_id = batch.get("core_id")
        if core_id is None:
            core_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        return gate[core_id.long().clamp(0, gate.numel() - 1)]

    def reset_zone_summary_diagnostics(self) -> None:
        self._zone_summary_diagnostics = {}

    @torch.no_grad()
    def _record_zone_summary_diagnostics(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        if not self.zone_summary_diagnostics_enabled:
            return
        token_rms = tokens.detach().float().square().mean(dim=-1).sqrt()
        for index, name in enumerate(ZONE_SUMMARY_NAMES):
            row = self._zone_summary_diagnostics.setdefault(
                name,
                {"batches": 0.0, "count_sum": 0.0, "token_rms_sum": 0.0},
            )
            row["batches"] += float(tokens.size(0))
            row["count_sum"] += float(counts[:, index].detach().float().sum().item())
            row["token_rms_sum"] += float(token_rms[:, index].sum().item())
        self._zone_summary_diagnostics["gate_abs_mean"] = float(gate.detach().abs().mean().item())

    def zone_summary_diagnostics_report(self, *, reset: bool = False) -> dict[str, Any]:
        report: dict[str, Any] = {
            "enabled": bool(self.zone_summaries_enabled),
            "groups": {},
        }
        for name in ZONE_SUMMARY_NAMES:
            row = self._zone_summary_diagnostics.get(name)
            if not row:
                continue
            batches = max(1.0, row["batches"])
            report["groups"][name] = {
                "mean_entity_count": row["count_sum"] / batches,
                "token_activation_rms": row["token_rms_sum"] / batches,
            }
        if "gate_abs_mean" in self._zone_summary_diagnostics:
            report["mean_abs_tanh_gate"] = self._zone_summary_diagnostics["gate_abs_mean"]
        if reset:
            self.reset_zone_summary_diagnostics()
        return report

    def reset_semantic_diagnostics(self) -> None:
        self._semantic_diagnostic_calls = 0
        self._semantic_diagnostics = {}

    @torch.no_grad()
    def _record_semantic_diagnostics(
        self,
        base: torch.Tensor,
        contribution: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> None:
        if not self.semantic_diagnostics_enabled:
            return
        self._semantic_diagnostic_calls += 1
        interval = max(1, int(self.semantic_diagnostics_interval))
        if (self._semantic_diagnostic_calls - 1) % interval:
            return

        valid = batch.get("entity_mask")
        if valid is None:
            valid = torch.ones(base.shape[:2], dtype=torch.bool, device=base.device)
        base_norm = base.detach().float().norm(dim=-1)
        contribution_norm = contribution.detach().float().norm(dim=-1)
        base_rms = base.detach().float().square().mean(dim=-1).sqrt()
        contribution_rms = contribution.detach().float().square().mean(dim=-1).sqrt()
        ratio = contribution_norm / base_norm.clamp_min(1.0e-8)

        def accumulate(name: str, selected: torch.Tensor) -> None:
            selected = selected & valid
            count = int(selected.sum().item())
            if count <= 0:
                return
            row = self._semantic_diagnostics.setdefault(
                name,
                {
                    "count": 0.0,
                    "ratio_sum": 0.0,
                    "base_norm_sum": 0.0,
                    "contribution_norm_sum": 0.0,
                    "base_rms_sum": 0.0,
                    "contribution_rms_sum": 0.0,
                },
            )
            row["count"] += float(count)
            row["ratio_sum"] += float(ratio[selected].sum().item())
            row["base_norm_sum"] += float(base_norm[selected].sum().item())
            row["contribution_norm_sum"] += float(contribution_norm[selected].sum().item())
            row["base_rms_sum"] += float(base_rms[selected].sum().item())
            row["contribution_rms_sum"] += float(contribution_rms[selected].sum().item())

        accumulate("all", valid)
        zone = batch["entity_cat"][..., 2]
        for zone_id in torch.unique(zone[valid]).tolist():
            accumulate(f"zone:{int(zone_id)}", zone == int(zone_id))

        for label, key in (("core", "core_id"), ("opponent", "opponent_class")):
            values = batch.get(key)
            if values is None:
                continue
            values = values.long().view(-1)
            for value in torch.unique(values).tolist():
                selected_rows = values == int(value)
                accumulate(
                    f"{label}:{int(value)}",
                    selected_rows.unsqueeze(1).expand_as(valid),
                )

    def semantic_diagnostics_report(self, *, reset: bool = False) -> dict[str, Any]:
        report: dict[str, Any] = {
            "enabled": bool(self.semantic_enabled),
            "sample_interval": max(1, int(self.semantic_diagnostics_interval)),
            "sampled_batches": sum(
                1
                for index in range(self._semantic_diagnostic_calls)
                if index % max(1, int(self.semantic_diagnostics_interval)) == 0
            ),
            "groups": {},
        }
        for name, row in sorted(self._semantic_diagnostics.items()):
            count = max(float(row["count"]), 1.0)
            report["groups"][name] = {
                "entities": int(row["count"]),
                "contribution_ratio_mean": row["ratio_sum"] / count,
                "base_norm_mean": row["base_norm_sum"] / count,
                "contribution_norm_mean": row["contribution_norm_sum"] / count,
                "base_activation_rms": row["base_rms_sum"] / count,
                "contribution_activation_rms": row["contribution_rms_sum"] / count,
            }
        if reset:
            self.reset_semantic_diagnostics()
        return report

    def _compact_rule_embeddings(
        self,
        item_ids: torch.Tensor,
        flat_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self.rule_semantic_embedding is None:
            return torch.zeros(
                (*item_ids.shape, self.config.d_model),
                dtype=self.card_embedding.weight.dtype,
                device=item_ids.device,
            )
        flat_item = item_ids.reshape(-1)
        starts = offsets[flat_item]
        lengths = offsets[flat_item + 1] - starts
        if bool((lengths <= 0).any()):
            raise RuntimeError("card rule semantic index contains an empty compact row")
        bag_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=item_ids.device),
                torch.cumsum(lengths, dim=0)[:-1],
            ]
        )
        bag_starts = torch.repeat_interleave(starts, lengths)
        bag_prefix = torch.repeat_interleave(
            torch.cumsum(lengths, dim=0) - lengths,
            lengths,
        )
        local = torch.arange(bag_starts.numel(), device=item_ids.device) - bag_prefix
        features = flat_ids[bag_starts + local]
        return self.rule_semantic_embedding(features, bag_offsets).view(
            *item_ids.shape,
            self.config.d_model,
        )

    def _card_rule_embeddings(self, card_ids: torch.Tensor) -> torch.Tensor:
        return self._compact_rule_embeddings(
            card_ids.clamp(0, self.config.max_card_id),
            self.card_rule_flat_ids,
            self.card_rule_offsets,
        )

    def _attack_rule_embeddings(self, attack_ids: torch.Tensor) -> torch.Tensor:
        return self._compact_rule_embeddings(
            attack_ids.clamp(0, self.config.max_attack_id),
            self.attack_rule_flat_ids,
            self.attack_rule_offsets,
        )

    def _pooled_child_rule_embeddings(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entity_cards = batch["entity_cat"][..., 0].clamp(0, self.config.max_card_id)
        rule_semantic = self._card_rule_embeddings(entity_cards)
        parents = batch["entity_parent"]
        child_valid = batch["entity_mask"] & (parents >= 0)
        safe_parent = parents.clamp(0, max(entity_cards.size(1) - 1, 0))
        pooled = torch.zeros_like(rule_semantic)
        pooled.scatter_add_(
            1,
            safe_parent.unsqueeze(-1).expand_as(rule_semantic),
            rule_semantic * child_valid.unsqueeze(-1).to(rule_semantic.dtype),
        )
        counts = torch.zeros(
            (*parents.shape, 1),
            dtype=rule_semantic.dtype,
            device=entity_cards.device,
        )
        counts.scatter_add_(
            1,
            safe_parent.unsqueeze(-1),
            child_valid.unsqueeze(-1).to(rule_semantic.dtype),
        )
        return pooled / counts.clamp_min(1.0), counts

    def _entity_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cat = batch["entity_cat"]
        card = cat[..., 0].clamp(0, self.config.max_card_id)
        base = (
            self.card_embedding(card)
            + self.owner_embedding(cat[..., 1].clamp(0, 3))
            + self.zone_embedding(cat[..., 2].clamp(0, 31))
            + self.slot_embedding(cat[..., 3].clamp(0, 64))
            + self.kind_embedding(cat[..., 4].clamp(0, 7))
            + self.status_embedding(cat[..., 5].clamp(0, 63))
            + self.entity_num_projection(batch["entity_num"])
        )
        if self.card_semantic_embedding is not None and self.semantic_enabled:
            if self.card_semantic_flat_ids.numel() > 0:
                # Build one compact bag per entity. This avoids materializing
                # [batch, entities, max_features, d_model] during rollout.
                flat_card = card.reshape(-1)
                starts = self.card_semantic_offsets[flat_card]
                lengths = (self.card_semantic_offsets[flat_card + 1] - starts).clamp_min(1)
                bag_offsets = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long, device=card.device),
                        torch.cumsum(lengths, dim=0)[:-1],
                    ]
                )
                bag_starts = torch.repeat_interleave(starts, lengths)
                bag_prefix = torch.repeat_interleave(torch.cumsum(lengths, dim=0) - lengths, lengths)
                local = torch.arange(bag_starts.numel(), device=card.device) - bag_prefix
                feature_ids = self.card_semantic_flat_ids[
                    (bag_starts + local).clamp_max(self.card_semantic_flat_ids.numel() - 1)
                ]
                semantic = self.card_semantic_embedding(feature_ids, bag_offsets).view(
                    *card.shape, base.size(-1)
                )
            else:
                # Compatibility path for the first scalar-gate semantic smoke
                # checkpoint, whose flat index was not serialized.
                semantic_ids = self.card_semantic_ids[card]
                semantic_mask = self.card_semantic_mask[card].unsqueeze(-1).to(base.dtype)
                semantic = self.card_semantic_embedding(semantic_ids.reshape(-1), torch.arange(
                    0,
                    semantic_ids.numel(),
                    semantic_ids.size(-1),
                    device=card.device,
                    dtype=torch.long,
                )).view(*card.shape, base.size(-1))
                semantic = semantic * (semantic_mask.sum(dim=-2) > 0).to(base.dtype)
            gate = self._conditioned_gate_values(
                self.semantic_gate,
                batch,
                base.size(0),
                base.device,
            )
            contribution = torch.tanh(gate).to(base.dtype).view(-1, 1, 1) * semantic
            self._record_semantic_diagnostics(base, contribution, batch)
            base = base + contribution
        if self.rule_semantic_embedding is not None and self.rule_semantics_enabled:
            rule_semantic = self._card_rule_embeddings(card).to(base.dtype)
            semantic_gate = self._conditioned_gate_values(
                self.rule_semantic_gate,
                batch,
                base.size(0),
                base.device,
            )
            semantic_contribution = torch.tanh(semantic_gate).to(base.dtype).view(
                -1, 1, 1
            ) * rule_semantic
            self._record_rule_semantic_diagnostics(
                "entity_semantic",
                base,
                semantic_contribution,
                batch["entity_mask"],
            )
            base = base + semantic_contribution

            pooled, counts = self._pooled_child_rule_embeddings(batch)
            pooled = pooled.to(base.dtype)
            modifier = self.rule_modifier_projection(pooled)
            modifier = modifier * (counts > 0).to(modifier.dtype)
            modifier_gate = self._conditioned_gate_values(
                self.rule_modifier_gate,
                batch,
                base.size(0),
                base.device,
            )
            modifier_contribution = torch.tanh(modifier_gate).to(base.dtype).view(
                -1, 1, 1
            ) * modifier
            self._record_rule_semantic_diagnostics(
                "attached_modifier",
                base,
                modifier_contribution,
                batch["entity_mask"] & (counts.squeeze(-1) > 0),
            )
            base = base + modifier_contribution
        parents = batch["entity_parent"]
        safe_parent = parents.clamp(min=0, max=max(base.size(1) - 1, 0))
        parent_repr = base.gather(1, safe_parent.unsqueeze(-1).expand(-1, -1, base.size(-1)))
        base = base + self.parent_projection(parent_repr) * (parents >= 0).unsqueeze(-1)
        return self.entity_norm(base)

    def interaction_relation_flags(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return compositional attack/protection flags for every option."""
        cat = batch["option_cat"]
        shape = (*cat.shape[:-1], self.config.card_interaction_relation_features)
        if self.interaction_relation_embedding is None:
            return torch.zeros(shape, dtype=torch.bool, device=cat.device)
        if self.config.card_interaction_runtime_version == 1:
            return self._legacy_interaction_relation_flags(batch)

        attack_id = cat[..., 6].clamp(0, self.config.max_attack_id)
        attack_channels = self.attack_interaction_channels[attack_id]
        channel_flags = [
            (attack_channels & ATTACK_CHANNEL_BITS[name]) != 0
            for name in (
                "normal_damage",
                "attack_effect",
                "damage_counter",
                "non_damage_effect",
            )
        ]
        is_normal, is_effect, is_counter, is_non_damage = channel_flags
        entity_cat = batch["entity_cat"]
        entity_mask = batch["entity_mask"]
        entity_parent = batch["entity_parent"]
        entity_cards = entity_cat[..., 0].clamp(0, self.config.max_card_id)
        entity_traits = self.card_interaction_predicates[entity_cards].clone()
        safe_parent = entity_parent.clamp(0, max(entity_cat.size(1) - 1, 0))
        attached_traits = torch.zeros_like(entity_traits)
        for name in (
            "attached_energy:metal",
            "attached_energy:water",
            "has_special_energy",
        ):
            bit = INTERACTION_PREDICATE_BITS[name]
            child_has_trait = (
                entity_mask & (entity_parent >= 0) & ((entity_traits & bit) != 0)
            )
            counts = torch.zeros_like(entity_traits)
            counts.scatter_add_(1, safe_parent, child_has_trait.long())
            attached_traits |= (counts > 0).long() * bit
        entity_traits |= attached_traits & ATTACHED_PREDICATE_MASK
        bench = (entity_cat[..., 2] == ZONE_OWN_BENCH) | (
            entity_cat[..., 2] == ZONE_OPP_BENCH
        )
        entity_traits |= bench.long() * INTERACTION_PREDICATE_BITS["zone:bench"]

        batch_size, entity_count = entity_cards.shape
        option_count = cat.size(1)
        entity_indices = torch.arange(
            entity_count, device=cat.device, dtype=torch.long
        ).view(1, entity_count)
        own_active = (
            entity_mask
            & (entity_cat[..., 1] == OWNER_SELF)
            & (entity_cat[..., 2] == ZONE_OWN_ACTIVE)
        )
        opponent_active = (
            entity_mask
            & (entity_cat[..., 1] == OWNER_OPP)
            & (entity_cat[..., 2] == ZONE_OPP_ACTIVE)
        )
        own_active_index = (
            torch.where(own_active, entity_indices + 1, torch.zeros_like(entity_indices))
            .amax(dim=-1)
            .sub(1)
        )
        opponent_active_index = (
            torch.where(
                opponent_active, entity_indices + 1, torch.zeros_like(entity_indices)
            )
            .amax(dim=-1)
            .sub(1)
        )
        source_index = cat[..., 8] - 1
        target_index = cat[..., 9] - 1
        source_index = torch.where(
            source_index >= 0,
            source_index,
            own_active_index.unsqueeze(1).expand(-1, option_count),
        )
        target_index = torch.where(
            target_index >= 0,
            target_index,
            opponent_active_index.unsqueeze(1).expand(-1, option_count),
        )
        source_valid = (source_index >= 0) & (source_index < entity_count)
        target_valid = (target_index >= 0) & (target_index < entity_count)
        safe_source = source_index.clamp(0, max(entity_count - 1, 0))
        safe_target = target_index.clamp(0, max(entity_count - 1, 0))

        def gather_entity(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return values.gather(1, indices)

        attacker_traits = gather_entity(entity_traits, safe_source)
        target_traits = gather_entity(entity_traits, safe_target)
        target_owner = gather_entity(entity_cat[..., 1], safe_target)
        target_zone = gather_entity(entity_cat[..., 2], safe_target)

        protection_channels = self.card_interaction_protection_channels[entity_cards]
        protection_scope = self.card_interaction_protection_scope[entity_cards]
        protection_mode = self.card_interaction_protection_mode[entity_cards]
        protection_amount = self.card_interaction_protection_amount[entity_cards]
        source_requirements = self.card_interaction_source_predicates[entity_cards]
        target_requirements = self.card_interaction_target_predicates[entity_cards]
        protection_ambiguous = self.card_interaction_protection_ambiguous[
            entity_cards
        ]
        in_play = torch.zeros_like(entity_mask)
        for zone in IN_PLAY_ZONES:
            in_play |= entity_cat[..., 2] == int(zone)
        protection_source = (
            entity_mask & in_play & (protection_scope != PROTECTION_SCOPE_IDS["none"])
        )

        source_owner = entity_cat[..., 1]
        same_owner = source_owner.unsqueeze(1) == target_owner.unsqueeze(2)
        target_on_bench = (target_zone == ZONE_OWN_BENCH) | (
            target_zone == ZONE_OPP_BENCH
        )
        source_grid = entity_indices.view(1, 1, entity_count)
        scope_matches = torch.zeros(
            (batch_size, option_count, entity_count),
            dtype=torch.bool,
            device=cat.device,
        )
        scope_matches |= (
            protection_scope.unsqueeze(1)
            == PROTECTION_SCOPE_IDS["attached_pokemon"]
        ) & (entity_parent.unsqueeze(1) == target_index.unsqueeze(2))
        scope_matches |= (
            protection_scope.unsqueeze(1) == PROTECTION_SCOPE_IDS["bench"]
        ) & target_on_bench.unsqueeze(2)
        scope_matches |= protection_scope.unsqueeze(1) == PROTECTION_SCOPE_IDS[
            "both_all"
        ]
        for scope_name in ("own_all", "own_basic_team_rocket"):
            scope_matches |= (
                protection_scope.unsqueeze(1) == PROTECTION_SCOPE_IDS[scope_name]
            ) & same_owner
        scope_matches |= (
            protection_scope.unsqueeze(1) == PROTECTION_SCOPE_IDS["self"]
        ) & (source_grid == target_index.unsqueeze(2))

        source_predicates_match = (
            attacker_traits.unsqueeze(2) & source_requirements.unsqueeze(1)
        ) == source_requirements.unsqueeze(1)
        target_predicates_match = (
            target_traits.unsqueeze(2) & target_requirements.unsqueeze(1)
        ) == target_requirements.unsqueeze(1)
        applies = (
            protection_source.unsqueeze(1)
            & source_valid.unsqueeze(2)
            & target_valid.unsqueeze(2)
            & scope_matches
            & source_predicates_match
            & target_predicates_match
        )
        attack_channel_grid = attack_channels.unsqueeze(2)
        protection_channel_grid = protection_channels.unsqueeze(1)
        affected = (attack_channel_grid & protection_channel_grid) != 0
        exact_applies = applies & ~protection_ambiguous.unsqueeze(1)
        relevant = exact_applies & affected
        blocking = relevant & (
            protection_mode.unsqueeze(1) == PROTECTION_MODE_BLOCK
        )
        reducing = relevant & (
            protection_mode.unsqueeze(1) == PROTECTION_MODE_REDUCE
        ) & (protection_amount.unsqueeze(1) > 0)

        def blocked(channel: str) -> torch.Tensor:
            bit = ATTACK_CHANNEL_BITS[channel]
            return (
                blocking
                & ((protection_channels.unsqueeze(1) & bit) != 0)
                & ((attack_channels.unsqueeze(2) & bit) != 0)
            ).any(dim=-1)

        normal_blocked = blocked("normal_damage")
        effect_blocked = blocked("attack_effect")
        counter_blocked = blocked("damage_counter") | effect_blocked
        normal_reduced = (
            reducing
            & ((protection_channels.unsqueeze(1) & ATTACK_CHANNEL_BITS["normal_damage"]) != 0)
        ).any(dim=-1)
        any_effective = (
            (is_normal & ~normal_blocked)
            | (is_effect & ~effect_blocked)
            | (is_counter & ~counter_blocked)
            | (is_non_damage & ~effect_blocked)
        )
        any_meaningful = is_normal | is_effect | is_counter | is_non_damage
        normal_passthrough = is_normal & (
            applies
            & ((protection_channels.unsqueeze(1) & ATTACK_CHANNEL_BITS["attack_effect"]) != 0)
            & ((protection_channels.unsqueeze(1) & ATTACK_CHANNEL_BITS["normal_damage"]) == 0)
        ).any(dim=-1)
        future_mode = self.attack_interaction_future_mode[attack_id]
        boardwide_scopes = {
            PROTECTION_SCOPE_IDS["both_all"],
            PROTECTION_SCOPE_IDS["own_all"],
            PROTECTION_SCOPE_IDS["own_basic_team_rocket"],
        }
        boardwide = torch.zeros_like(relevant)
        for scope_id in boardwide_scopes:
            boardwide |= protection_scope.unsqueeze(1) == scope_id
        return torch.stack(
            [
                is_normal,
                is_effect,
                is_counter,
                is_non_damage,
                protection_source.any(dim=-1).unsqueeze(1).expand(-1, option_count),
                (protection_source.unsqueeze(1) & scope_matches).any(dim=-1),
                (
                    protection_source.unsqueeze(1) & source_predicates_match
                ).any(dim=-1),
                (
                    protection_source.unsqueeze(1)
                    & scope_matches
                    & source_predicates_match
                    & target_predicates_match
                ).any(dim=-1),
                normal_blocked,
                effect_blocked,
                counter_blocked,
                normal_reduced,
                (
                    relevant
                    & (
                        protection_scope.unsqueeze(1)
                        == PROTECTION_SCOPE_IDS["attached_pokemon"]
                    )
                ).any(dim=-1),
                (relevant & boardwide).any(dim=-1),
                (
                    relevant
                    & (
                        protection_scope.unsqueeze(1)
                        == PROTECTION_SCOPE_IDS["self"]
                    )
                ).any(dim=-1),
                (applies & protection_ambiguous.unsqueeze(1) & affected).any(dim=-1),
                any_effective,
                any_meaningful & ~any_effective,
                normal_passthrough,
                future_mode == PROTECTION_MODE_BLOCK,
                future_mode == PROTECTION_MODE_REDUCE,
            ],
            dim=-1,
        )

    def _legacy_interaction_relation_flags(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Evaluate the original five-feature Articuno-only branch."""
        card_stage_basic = 1 << 0
        card_team_rocket = 1 << 2
        card_self_protection = 1 << 3
        card_rocket_aura = 1 << 4
        attack_effect = 1 << 0
        attack_normal = 1 << 2
        cat = batch["option_cat"]
        attack_id = cat[..., 6].clamp(0, self.config.max_attack_id)
        attack_bits = self.attack_interaction_traits[attack_id]
        is_effect = (attack_bits & attack_effect) != 0
        is_normal = (attack_bits & attack_normal) != 0
        entity_cat = batch["entity_cat"]
        entity_mask = batch["entity_mask"]
        entity_cards = entity_cat[..., 0].clamp(0, self.config.max_card_id)
        entity_traits = self.card_interaction_traits[entity_cards]
        opponent_board = (
            entity_mask
            & (entity_cat[..., 1] == OWNER_OPP)
            & (
                (entity_cat[..., 2] == ZONE_OPP_ACTIVE)
                | (entity_cat[..., 2] == ZONE_OPP_BENCH)
            )
        )
        aura_visible = (
            opponent_board & ((entity_traits & card_rocket_aura) != 0)
        ).any(dim=-1)
        target_card = cat[..., 5].clamp(0, self.config.max_card_id)
        opponent_active = (
            entity_mask
            & (entity_cat[..., 1] == OWNER_OPP)
            & (entity_cat[..., 2] == ZONE_OPP_ACTIVE)
        )
        active_card = torch.where(
            opponent_active, entity_cards, torch.zeros_like(entity_cards)
        ).amax(dim=-1)
        target_card = torch.where(
            target_card > 0,
            target_card,
            active_card.unsqueeze(1).expand_as(target_card),
        )
        target_traits = self.card_interaction_traits[target_card]
        target_matches_scope = (
            ((target_traits & card_stage_basic) != 0)
            & ((target_traits & card_team_rocket) != 0)
        )
        self_protected = (target_traits & card_self_protection) != 0
        protection_visible = self_protected | aura_visible.unsqueeze(1)
        protection_applies = self_protected | (
            aura_visible.unsqueeze(1) & target_matches_scope
        )
        blocked = is_effect & protection_applies
        normal_passthrough = is_normal & protection_applies
        return torch.stack(
            [
                is_effect,
                protection_visible,
                target_matches_scope,
                blocked,
                normal_passthrough,
            ],
            dim=-1,
        )

    def _zone_summary_tokens(
        self,
        entity_repr: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention-pool semantic-enriched entities into fixed zone tokens.

        The returned entity order is unchanged; callers append these tokens
        after the entity sequence so source/target pointer indices remain
        valid.  Empty zones still receive a learned empty/type representation,
        which lets the policy distinguish an absent bench from a missing
        observation.
        """
        batch_size, _entity_count, d = entity_repr.shape
        if self.zone_summary_gate is None:
            return (
                entity_repr.new_zeros((batch_size, 0, d)),
                torch.zeros((batch_size, 0), dtype=torch.bool, device=entity_repr.device),
            )
        if not self.zone_summaries_enabled:
            # Runtime off is an information ablation, not a sequence-layout
            # ablation: every branch still has the same eight latent slots.
            # This makes paired comparisons isolate the pooled content while
            # keeping pointer positions and Transformer shapes identical.
            return (
                entity_repr.new_zeros((batch_size, len(ZONE_SUMMARY_GROUPS), d)),
                torch.ones(
                    (batch_size, len(ZONE_SUMMARY_GROUPS)),
                    dtype=torch.bool,
                    device=entity_repr.device,
                ),
            )

        zone = batch["entity_cat"][..., 2].clamp(0, 31)
        entity_mask = batch["entity_mask"].bool()
        group_by_zone = self.zone_summary_group_by_zone.to(zone.device)
        groups = group_by_zone[zone]
        group_index = torch.arange(
            len(ZONE_SUMMARY_GROUPS), device=zone.device, dtype=torch.long
        ).view(1, -1, 1)
        group_mask = entity_mask.unsqueeze(1) & (groups.unsqueeze(1) == group_index)
        counts = group_mask.sum(dim=-1)

        keys = self.zone_summary_key(entity_repr)
        values = self.zone_summary_value(entity_repr)
        scores = torch.einsum("bnd,gd->bgn", keys, self.zone_summary_query)
        scores = scores / math.sqrt(float(d))
        scores = scores.masked_fill(~group_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(group_mask, weights, torch.zeros_like(weights))
        pooled = torch.einsum("bgn,bnd->bgd", weights, values)

        empty = counts == 0
        empty_repr = self.zone_summary_empty_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        pooled = torch.where(empty.unsqueeze(-1), empty_repr, pooled)
        pooled = self.zone_summary_projection(self.zone_summary_norm(pooled))
        count_feature = torch.log1p(counts.to(pooled.dtype)).unsqueeze(-1)
        pooled = pooled + self.zone_summary_count_projection(count_feature)
        pooled = pooled + self.zone_summary_type_embedding.unsqueeze(0)

        gate = self._conditioned_gate_values(
            self.zone_summary_gate,
            batch,
            batch_size,
            entity_repr.device,
        )
        tokens = torch.tanh(gate).to(pooled.dtype).view(-1, 1, 1) * pooled
        self._record_zone_summary_diagnostics(tokens, counts, torch.tanh(gate))
        valid = torch.ones(
            (batch_size, len(ZONE_SUMMARY_GROUPS)), dtype=torch.bool, device=entity_repr.device
        )
        return tokens, valid

    def _global_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cat = batch["global_cat"]
        out = (
            self.select_type_embedding(cat[:, 0].clamp(0, 127))
            + self.select_context_embedding(cat[:, 1].clamp(0, 255))
            + self.small_embedding(cat[:, 2].clamp(0, 127))
            + self.small_embedding(cat[:, 3].clamp(0, 127))
            + self.small_embedding(cat[:, 4].clamp(0, 127))
            + self.small_embedding(cat[:, 5].clamp(0, 127))
            + self.small_embedding(cat[:, 6].clamp(0, 127))
            + self.small_embedding(cat[:, 7].clamp(0, 127))
            + self.global_num_projection(batch["global_num"])
        )
        return self.global_norm(out)

    def _encode_state_tokens(
        self,
        state_tokens: torch.Tensor,
        state_valid: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del batch
        return self.state_encoder(state_tokens, src_key_padding_mask=~state_valid)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_base = self._global_embedding(batch)
        entity_base = self._entity_embeddings(batch)
        zone_tokens, zone_valid = self._zone_summary_tokens(entity_base, batch)
        # Keep entities immediately after the global token.  Zone tokens are
        # appended, so option_cat source/target entity pointers retain their
        # original indices under the new architecture.
        state_tokens = torch.cat([global_base.unsqueeze(1), entity_base, zone_tokens], dim=1)
        global_valid = torch.ones((state_tokens.size(0), 1), dtype=torch.bool, device=state_tokens.device)
        state_valid = torch.cat([global_valid, batch["entity_mask"], zone_valid], dim=1)
        state_encoded = self._encode_state_tokens(state_tokens, state_valid, batch)
        state_repr = state_encoded[:, 0]
        entity_count = int(entity_base.size(1))
        entity_encoded = state_encoded[:, 1 : 1 + entity_count]
        zone_encoded = state_encoded[:, 1 + entity_count :]

        cat = batch["option_cat"]
        source_index = cat[..., 8] - 1
        target_index = cat[..., 9] - 1
        safe_source = source_index.clamp(min=0, max=max(entity_encoded.size(1) - 1, 0))
        safe_target = target_index.clamp(min=0, max=max(entity_encoded.size(1) - 1, 0))
        source_repr = entity_encoded.gather(1, safe_source.unsqueeze(-1).expand(-1, -1, entity_encoded.size(-1)))
        target_repr = entity_encoded.gather(1, safe_target.unsqueeze(-1).expand(-1, -1, entity_encoded.size(-1)))
        source_repr = source_repr * (source_index >= 0).unsqueeze(-1)
        target_repr = target_repr * (target_index >= 0).unsqueeze(-1)

        option_base = (
            self.option_type_embedding(cat[..., 0].clamp(0, 64))
            + self.area_embedding(cat[..., 1].clamp(0, 32))
            + self.area_embedding(cat[..., 2].clamp(0, 32))
            + self.owner_embedding(cat[..., 3].clamp(0, 3))
            + self.card_embedding(cat[..., 4].clamp(0, self.config.max_card_id))
            + self.card_embedding(cat[..., 5].clamp(0, self.config.max_card_id))
            + self.attack_embedding(cat[..., 6].clamp(0, self.config.max_attack_id))
            + self.number_embedding(cat[..., 7].clamp(0, 128))
            + self.option_position_embedding(cat[..., 10].clamp(0, 128))
            + self.option_position_embedding(cat[..., 11].clamp(0, 128))
            + self.option_num_projection(batch["option_num"])
        )
        if self.rule_semantic_embedding is not None and self.rule_semantics_enabled:
            source_card_rule = self._card_rule_embeddings(
                cat[..., 4].clamp(0, self.config.max_card_id)
            )
            target_card_rule = self._card_rule_embeddings(
                cat[..., 5].clamp(0, self.config.max_card_id)
            )
            attack_rule = self._attack_rule_embeddings(
                cat[..., 6].clamp(0, self.config.max_attack_id)
            )
            pooled_children, child_counts = self._pooled_child_rule_embeddings(batch)
            projected_children = self.rule_modifier_projection(pooled_children)
            projected_children = projected_children * (child_counts > 0).to(
                projected_children.dtype
            )
            source_modifier = projected_children.gather(
                1,
                safe_source.unsqueeze(-1).expand(
                    -1, -1, projected_children.size(-1)
                ),
            )
            target_modifier = projected_children.gather(
                1,
                safe_target.unsqueeze(-1).expand(
                    -1, -1, projected_children.size(-1)
                ),
            )
            source_modifier = source_modifier * (source_index >= 0).unsqueeze(-1)
            target_modifier = target_modifier * (target_index >= 0).unsqueeze(-1)
            option_rule = (
                source_card_rule
                + target_card_rule
                + attack_rule
                + source_modifier
                + target_modifier
            ).to(option_base.dtype)
            semantic_gate = self._conditioned_gate_values(
                self.rule_semantic_gate,
                batch,
                option_base.size(0),
                option_base.device,
            )
            option_semantic_contribution = torch.tanh(semantic_gate).to(
                option_base.dtype
            ).view(-1, 1, 1) * option_rule
            self._record_rule_semantic_diagnostics(
                "option_semantic",
                option_base,
                option_semantic_contribution,
                batch["option_mask"],
            )
            option_base = option_base + option_semantic_contribution

            entity_cat = batch["entity_cat"]
            entity_mask = batch["entity_mask"]
            own_active_mask = (
                entity_mask
                & (entity_cat[..., 1] == OWNER_SELF)
                & (entity_cat[..., 2] == ZONE_OWN_ACTIVE)
            )
            opponent_active_mask = (
                entity_mask
                & (entity_cat[..., 1] == OWNER_OPP)
                & (entity_cat[..., 2] == ZONE_OPP_ACTIVE)
            )
            own_active_repr = (
                entity_encoded
                * own_active_mask.unsqueeze(-1).to(entity_encoded.dtype)
            ).sum(dim=1)
            opponent_active_repr = (
                entity_encoded
                * opponent_active_mask.unsqueeze(-1).to(entity_encoded.dtype)
            ).sum(dim=1)
            relation_source = torch.where(
                (source_index >= 0).unsqueeze(-1),
                source_repr,
                own_active_repr.unsqueeze(1),
            )
            relation_target = torch.where(
                (target_index >= 0).unsqueeze(-1),
                target_repr,
                opponent_active_repr.unsqueeze(1),
            )
            state_expand = state_repr.unsqueeze(1).expand(
                -1, option_base.size(1), -1
            )
            relation = self.rule_relation_projection(
                torch.cat(
                    [option_rule, relation_source, relation_target, state_expand],
                    dim=-1,
                )
            )
            relation_gate = self._conditioned_gate_values(
                self.rule_relation_gate,
                batch,
                option_base.size(0),
                option_base.device,
            )
            relation_contribution = torch.tanh(relation_gate).to(
                option_base.dtype
            ).view(-1, 1, 1) * relation
            self._record_rule_semantic_diagnostics(
                "option_relation",
                option_base,
                relation_contribution,
                batch["option_mask"],
            )
            option_base = option_base + relation_contribution
            rule_atom_repr = option_rule
            rule_relation_repr = relation
        relation_flags = self.interaction_relation_flags(batch)
        interaction_relation_repr = None
        interaction_effectiveness_repr = None
        interaction_block_repr = None
        interaction_future_repr = None
        if (
            self.interaction_relation_embedding is not None
            and self.interaction_semantics_enabled
        ):
            embedding = self.interaction_relation_embedding.weight.to(
                option_base.dtype
            )

            def project_group(feature_names: tuple[str, ...]) -> torch.Tensor:
                indices = [RELATION_FEATURE_NAMES.index(name) for name in feature_names]
                selected_flags = relation_flags[..., indices]
                grouped = torch.matmul(
                    selected_flags.to(option_base.dtype), embedding[indices]
                )
                if self.interaction_relation_projection is not None:
                    grouped = self.interaction_relation_projection(grouped)
                return grouped * selected_flags.any(dim=-1, keepdim=True).to(
                    grouped.dtype
                )

            if self.config.card_interaction_split_gates:
                interaction_effectiveness_repr = project_group(
                    INTERACTION_GATE_GROUP_FEATURES["effectiveness"]
                )
                interaction_block_repr = project_group(
                    INTERACTION_GATE_GROUP_FEATURES["block_immunity"]
                )
                interaction_future_repr = project_group(
                    INTERACTION_GATE_GROUP_FEATURES["future_protection"]
                )
                interaction_relation_repr = (
                    interaction_effectiveness_repr
                    + interaction_block_repr
                    + interaction_future_repr
                )
                gate_specs = (
                    (
                        self.interaction_relation_gate,
                        interaction_effectiveness_repr,
                    ),
                    (self.interaction_block_gate, interaction_block_repr),
                    (self.interaction_future_gate, interaction_future_repr),
                )
                relation_contribution = torch.zeros_like(option_base)
                gate_values: list[torch.Tensor] = []
                for gate_parameter, grouped in gate_specs:
                    gate_value = self._conditioned_gate_values(
                        gate_parameter,
                        batch,
                        option_base.size(0),
                        option_base.device,
                    )
                    gate_values.append(torch.tanh(gate_value))
                    relation_contribution = relation_contribution + torch.tanh(
                        gate_value
                    ).to(option_base.dtype).view(-1, 1, 1) * grouped
                diagnostic_gate = torch.stack(gate_values, dim=-1)
            else:
                relation = torch.matmul(
                    relation_flags.to(option_base.dtype), embedding
                )
                if self.interaction_relation_projection is not None:
                    relation = self.interaction_relation_projection(relation)
                interaction_relation_repr = relation
                interaction_effectiveness_repr = relation
                interaction_block_repr = relation
                relation_gate = self._conditioned_gate_values(
                    self.interaction_relation_gate,
                    batch,
                    option_base.size(0),
                    option_base.device,
                )
                relation_contribution = torch.tanh(relation_gate).to(
                    option_base.dtype
                ).view(-1, 1, 1) * relation
                diagnostic_gate = torch.tanh(relation_gate)
            self._record_interaction_semantic_diagnostics(
                option_base,
                relation_contribution,
                relation_flags,
                batch["option_mask"],
                diagnostic_gate,
            )
            option_base = option_base + relation_contribution
        state_expand = state_repr.unsqueeze(1).expand(-1, option_base.size(1), -1)
        option_repr = self.option_fusion(torch.cat([option_base, source_repr, target_repr, state_expand], dim=-1))
        option_repr = option_repr * batch["option_mask"].unsqueeze(-1)
        outputs = {
            "state_repr": state_repr,
            "entity_repr": entity_encoded,
            "zone_repr": zone_encoded,
            "zone_valid": zone_valid,
            "option_repr": option_repr,
            "value": self.value_head(state_repr).squeeze(-1).tanh(),
            "next_prize": self.next_prize_head(state_repr).squeeze(-1),
            "finish_reason": self.finish_reason_head(state_repr),
            "opponent": self.opponent_head(state_repr),
            "family": self.family_head(state_repr),
            "kill_line": self.kill_line_head(state_repr).squeeze(-1),
            "safe_draw": self.safe_draw_head(state_repr).squeeze(-1),
            "future_family": self.future_family_head(state_repr),
            "attack_interaction_options": self.attack_interaction_head(
                option_repr
            ).squeeze(-1),
        }
        if (
            self.rule_auxiliary_outputs_enabled
            and self.rule_semantics_enabled
            and self.rule_aux_effectiveness_head is not None
        ):
            outputs.update(
                {
                    "rule_effectiveness_options": self.rule_aux_effectiveness_head(
                        rule_relation_repr
                    ).squeeze(-1),
                    "rule_block_reason_options": self.rule_aux_block_reason_head(
                        rule_relation_repr
                    ),
                    "rule_damage_bucket_options": self.rule_aux_damage_bucket_head(
                        rule_relation_repr
                    ),
                    "rule_atom_options": self.rule_aux_atom_head(rule_atom_repr),
                }
            )
        if (
            self.interaction_auxiliary_outputs_enabled
            and self.interaction_semantics_enabled
            and interaction_relation_repr is not None
            and self.interaction_aux_effectiveness_head is not None
        ):
            outputs.update(
                {
                    "interaction_effectiveness_options": (
                        self.interaction_aux_effectiveness_head(
                            interaction_effectiveness_repr
                            + interaction_block_repr
                        ).squeeze(-1)
                    ),
                    "interaction_block_options": self.interaction_aux_block_head(
                        interaction_block_repr
                    ),
                    # PPO derives the same deterministic, all-legal-option
                    # supervision used by Stage A from these runtime flags.
                    # Keeping this tensor behind the auxiliary-output switch
                    # avoids adding inference/checkpoint surface area.
                    "interaction_relation_flags": relation_flags,
                }
            )
            if (
                interaction_future_repr is not None
                and self.interaction_aux_future_head is not None
            ):
                outputs["interaction_future_options"] = (
                    self.interaction_aux_future_head(interaction_future_repr)
                )
        return outputs

    def initial_decoder_state(self, state_repr: torch.Tensor) -> list[torch.Tensor]:
        return [torch.tanh(layer(state_repr)) for layer in self.decoder_init]

    def pointer_logits(
        self,
        option_repr: torch.Tensor,
        hidden: list[torch.Tensor],
        valid_options: torch.Tensor,
        stop_valid: torch.Tensor,
    ) -> torch.Tensor:
        query = self.pointer_query(hidden[-1]).unsqueeze(1)
        keys = self.pointer_key(option_repr)
        logits = (query * keys).sum(-1) / math.sqrt(self.config.d_model)
        logits = logits + self.option_bias(option_repr).squeeze(-1)
        logits = logits.masked_fill(~valid_options, torch.finfo(logits.dtype).min)
        stop = self.stop_head(hidden[-1]).squeeze(-1)
        stop = stop.masked_fill(~stop_valid, torch.finfo(stop.dtype).min)
        return torch.cat([logits, stop.unsqueeze(1)], dim=1)

    def advance_decoder(self, selected_repr: torch.Tensor, hidden: list[torch.Tensor], update_mask: torch.Tensor) -> list[torch.Tensor]:
        next_hidden: list[torch.Tensor] = []
        x = selected_repr
        for cell, old in zip(self.decoder_cells, hidden):
            candidate = cell(x, old)
            new = torch.where(update_mask.unsqueeze(-1), candidate, old)
            next_hidden.append(new)
            x = new
        return next_hidden

    def sequence_log_probs(
        self,
        batch: dict[str, torch.Tensor],
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return joint log-probability and validity for each target sequence.

        Equivalent legal options share probability mass exactly as they do in
        BC.  The STOP decision is included when the demonstrated sequence ends
        before max_count, which makes this suitable for paired DPO training.
        """
        outputs = encoded or self.encode(batch)
        option_repr = outputs["option_repr"]
        option_mask = batch["option_mask"]
        selected_mask = torch.zeros_like(option_mask)
        hidden = self.initial_decoder_state(outputs["state_repr"])
        batch_size, option_count = option_mask.shape
        max_steps = int(batch["max_count"].max().item()) if batch_size else 0
        totals = torch.zeros((batch_size,), dtype=option_repr.dtype, device=option_repr.device)
        sequence_valid = torch.ones((batch_size,), dtype=torch.bool, device=option_repr.device)

        for step in range(max_steps + 1):
            action_len = batch["action_len"]
            target_option = step < action_len
            target_stop = (step == action_len) & (action_len < batch["max_count"])
            active = target_option | target_stop
            if not bool(active.any()):
                break
            valid = option_mask & ~selected_mask
            stop_valid = step >= batch["min_count"]
            logits = self.pointer_logits(option_repr, hidden, valid, stop_valid)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_index = torch.zeros((batch_size,), dtype=torch.long, device=option_repr.device)
            chosen_valid = torch.zeros((batch_size,), dtype=torch.bool, device=option_repr.device)

            for bi in range(batch_size):
                if not bool(active[bi]):
                    continue
                if bool(target_stop[bi]):
                    totals[bi] = totals[bi] + log_probs[bi, option_count]
                    sequence_valid[bi] &= bool(stop_valid[bi])
                    continue
                target = int(batch["actions"][bi, step].item())
                if target < 0 or target >= option_count or not bool(valid[bi, target]):
                    sequence_valid[bi] = False
                    continue
                group = batch["option_equiv"][bi, target]
                positive = valid[bi] & (batch["option_equiv"][bi] == group)
                totals[bi] = totals[bi] + torch.logsumexp(log_probs[bi, :option_count][positive], dim=0)
                chosen_index[bi] = target
                chosen_valid[bi] = True
                selected_mask[bi, target] = True

            safe_index = chosen_index.clamp(0, max(option_count - 1, 0))
            selected_repr = option_repr.gather(
                1,
                safe_index.view(-1, 1, 1).expand(-1, 1, option_repr.size(-1)),
            ).squeeze(1)
            hidden = self.advance_decoder(selected_repr, hidden, chosen_valid)

        return totals, sequence_valid

    def sequence_nll(
        self,
        batch: dict[str, torch.Tensor],
        encoded: dict[str, torch.Tensor] | None = None,
        *,
        return_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = encoded or self.encode(batch)
        option_repr = outputs["option_repr"]
        option_mask = batch["option_mask"]
        selected_mask = torch.zeros_like(option_mask)
        hidden = self.initial_decoder_state(outputs["state_repr"])
        batch_size, option_count = option_mask.shape
        max_steps = int(batch["max_count"].max().item()) if batch_size else 0
        weighted_loss = outputs["state_repr"].sum() * 0.0
        total_weight = batch["action_weight"].sum() * 0.0
        exact_ok = torch.ones((batch_size,), dtype=torch.bool, device=option_mask.device)
        first_ok = torch.zeros((batch_size,), dtype=torch.bool, device=option_mask.device)
        stop_ok = torch.ones((batch_size,), dtype=torch.bool, device=option_mask.device)

        # A target can contain at most max_count option decisions, and STOP is
        # emitted before max_count.  Iterating exactly max_steps avoids an
        # active.any() device synchronization at every decoder step.
        for step in range(max_steps):
            action_len = batch["action_len"]
            target_option = step < action_len
            target_stop = (step == action_len) & (action_len < batch["max_count"])
            valid = option_mask & ~selected_mask
            stop_valid = step >= batch["min_count"]
            logits = self.pointer_logits(option_repr, hidden, valid, stop_valid)
            log_probs = F.log_softmax(logits, dim=-1)
            greedy = logits.argmax(dim=-1)

            if step < batch["actions"].size(1):
                target_index = batch["actions"][:, step]
            else:
                target_index = torch.zeros((batch_size,), dtype=torch.long, device=option_mask.device)
            target_in_range = (target_index >= 0) & (target_index < option_count)
            safe_index = target_index.clamp(0, max(option_count - 1, 0))
            target_available = valid.gather(1, safe_index.unsqueeze(1)).squeeze(1)
            chosen_valid = target_option & target_in_range & target_available

            target_group = batch["option_equiv"].gather(1, safe_index.unsqueeze(1)).squeeze(1)
            positive = valid & (batch["option_equiv"] == target_group.unsqueeze(1))
            positive &= chosen_valid.unsqueeze(1)
            # logsumexp over an empty row has undefined gradients.  Give rows
            # without a legal target a harmless fallback which is masked out
            # of both the numerator and denominator below.
            reduce_mask = positive.clone()
            reduce_mask[:, 0] |= ~chosen_valid
            equivalent_logprob = torch.logsumexp(
                log_probs[:, :option_count].masked_fill(~reduce_mask, -torch.inf),
                dim=1,
            )
            option_nll = torch.where(chosen_valid, -equivalent_logprob, 0.0)
            stop_nll = torch.where(target_stop, -log_probs[:, option_count], 0.0)
            contributes = chosen_valid | target_stop
            step_weight = batch["action_weight"].clamp_min(0.0) * contributes.float()
            weighted_loss = weighted_loss + ((option_nll + stop_nll) * step_weight).sum()
            total_weight = total_weight + step_weight.sum()

            greedy_is_option = greedy < option_count
            safe_greedy = greedy.clamp(0, max(option_count - 1, 0))
            greedy_positive = positive.gather(1, safe_greedy.unsqueeze(1)).squeeze(1)
            option_match = chosen_valid & greedy_is_option & greedy_positive
            stop_match = target_stop & (greedy == option_count)
            if step == 0:
                first_ok = torch.where(chosen_valid, option_match, first_ok)
            exact_ok &= (~chosen_valid | option_match) & (~target_stop | stop_match)
            stop_ok &= ~target_stop | stop_match

            previous_selected = selected_mask.gather(1, safe_index.unsqueeze(1))
            selected_mask.scatter_(
                1,
                safe_index.unsqueeze(1),
                previous_selected | chosen_valid.unsqueeze(1),
            )
            selected_repr = option_repr.gather(
                1,
                safe_index.view(-1, 1, 1).expand(-1, 1, option_repr.size(-1)),
            ).squeeze(1)
            hidden = self.advance_decoder(selected_repr, hidden, chosen_valid)

        loss = weighted_loss / total_weight.clamp_min(1.0e-6)
        metrics: dict[str, float] = {}
        if return_metrics:
            metric_values = torch.stack(
                (exact_ok.float().mean(), first_ok.float().mean(), stop_ok.float().mean())
            ).detach().cpu().tolist()
            metrics = dict(zip(("sequence_exact", "first_action_acc", "stop_acc"), metric_values))
        return loss, metrics

    def compute_bc_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = self.encode(batch)
        policy_loss, sequence_metrics = self.sequence_nll(batch, outputs, return_metrics=return_metrics)
        value_parts = F.smooth_l1_loss(outputs["value"], batch["value_target"].clamp(-1, 1), reduction="none")
        value_loss = (value_parts * batch["value_weight"]).sum() / batch["value_weight"].sum().clamp_min(1.0e-6)
        prize_loss = F.binary_cross_entropy_with_logits(outputs["next_prize"], batch["next_prize_target"].clamp(0, 1))
        finish_target = batch["finish_reason"].clamp(0, self.config.finish_reason_classes - 1)
        finish_loss = F.cross_entropy(outputs["finish_reason"], finish_target)
        opponent_target = batch["opponent_class"].clamp(0, self.config.opponent_classes - 1)
        opponent_loss = F.cross_entropy(outputs["opponent"], opponent_target)
        family_target = batch["action_family"].clamp(0, len(FAMILY_NAMES) - 1)
        family_loss = F.cross_entropy(outputs["family"], family_target)
        loss = (
            policy_loss
            + 0.25 * value_loss
            + 0.10 * prize_loss
            + 0.05 * finish_loss
            + 0.05 * opponent_loss
            + 0.05 * family_loss
        )
        metrics: dict[str, float] = {}
        if return_metrics:
            names = ("loss", "policy_loss", "value_loss", "prize_loss", "finish_loss", "opponent_loss", "family_loss")
            values = torch.stack(
                (loss, policy_loss, value_loss, prize_loss, finish_loss, opponent_loss, family_loss)
            ).detach().float().cpu().tolist()
            metrics = {**dict(zip(names, values)), **sequence_metrics}
        return loss, metrics

    @torch.inference_mode()
    def greedy_decode_batch(self, batch: dict[str, torch.Tensor]) -> list[list[int]]:
        outputs = self.encode(batch)
        option_repr = outputs["option_repr"]
        option_mask = batch["option_mask"]
        selected_mask = torch.zeros_like(option_mask)
        hidden = self.initial_decoder_state(outputs["state_repr"])
        result: list[list[int]] = [[] for _ in range(option_mask.size(0))]
        active = torch.ones((option_mask.size(0),), dtype=torch.bool, device=option_mask.device)
        max_steps = int(batch["max_count"].max().item()) if option_mask.size(0) else 0
        for step in range(max_steps):
            if not bool(active.any()):
                break
            valid = option_mask & ~selected_mask & active.unsqueeze(-1)
            stop_valid = (step >= batch["min_count"]) & active
            logits = self.pointer_logits(option_repr, hidden, valid, stop_valid)
            choice = logits.argmax(dim=-1)
            chosen_valid = active & (choice < option_mask.size(1))
            for bi in range(option_mask.size(0)):
                if not bool(active[bi]):
                    continue
                idx = int(choice[bi].item())
                if idx >= option_mask.size(1):
                    active[bi] = False
                    continue
                result[bi].append(idx)
                selected_mask[bi, idx] = True
                if len(result[bi]) >= int(batch["max_count"][bi].item()):
                    active[bi] = False
            safe = choice.clamp(0, max(option_mask.size(1) - 1, 0))
            selected_repr = option_repr.gather(1, safe.view(-1, 1, 1).expand(-1, 1, option_repr.size(-1))).squeeze(1)
            hidden = self.advance_decoder(selected_repr, hidden, chosen_valid)
        for bi in range(len(result)):
            result[bi] = normalize_model_action(
                result[bi],
                int(option_mask[bi].sum().item()),
                int(batch["min_count"][bi].item()),
                int(batch["max_count"][bi].item()),
            )
        return result

    @torch.inference_mode()
    def sample_decode_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        temperature: float = 1.0,
        compute_entropy: bool = True,
        mode: str = "sample",
    ) -> tuple[list[list[int]], torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Decode legal action sequences and retain rollout policy statistics."""
        if mode not in {"sample", "greedy"}:
            raise ValueError(f"unknown decode mode: {mode}")
        outputs = self.encode(batch)
        option_repr = outputs["option_repr"]
        option_mask = batch["option_mask"]
        selected_mask = torch.zeros_like(option_mask)
        hidden = self.initial_decoder_state(outputs["state_repr"])
        batch_size, option_count = option_mask.shape
        active = torch.ones((batch_size,), dtype=torch.bool, device=option_mask.device)
        action_tensor = torch.full(
            (batch_size, int(batch["max_count"].max().item()) if batch_size else 0),
            -1,
            dtype=torch.long,
            device=option_mask.device,
        )
        action_lengths = torch.zeros((batch_size,), dtype=torch.long, device=option_mask.device)
        logprobs = torch.zeros((batch_size,), dtype=option_repr.dtype, device=option_repr.device)
        entropies = torch.zeros_like(logprobs) if compute_entropy else None
        max_steps = action_tensor.size(1)

        for step in range(max_steps):
            if not bool(active.any()):
                break
            valid = option_mask & ~selected_mask & active.unsqueeze(-1)
            stop_valid = (step >= batch["min_count"]) & active
            logits = self.pointer_logits(option_repr, hidden, valid, stop_valid)
            decode_logits = logits / max(float(temperature), 1.0e-6) if mode == "sample" else logits
            log_probs = F.log_softmax(decode_logits, dim=-1)
            if mode == "sample":
                choice = torch.distributions.Categorical(logits=decode_logits).sample()
            else:
                choice = logits.argmax(dim=-1)
            logprobs = logprobs + log_probs.gather(1, choice.unsqueeze(1)).squeeze(1) * active
            if entropies is not None:
                entropies = entropies - (log_probs.exp() * log_probs).sum(dim=-1) * active
            chosen_valid = active & (choice < option_count)
            safe = choice.clamp(0, max(option_count - 1, 0))
            action_tensor[:, step] = torch.where(chosen_valid, safe, -1)
            previous_selected = selected_mask.gather(1, safe.unsqueeze(1))
            selected_mask.scatter_(
                1,
                safe.unsqueeze(1),
                previous_selected | chosen_valid.unsqueeze(1),
            )
            action_lengths = action_lengths + chosen_valid.long()
            active = chosen_valid & (action_lengths < batch["max_count"])
            selected_repr = option_repr.gather(
                1,
                safe.view(-1, 1, 1).expand(-1, 1, option_repr.size(-1)),
            ).squeeze(1)
            hidden = self.advance_decoder(selected_repr, hidden, chosen_valid)

        raw_actions = action_tensor.cpu().tolist()
        lengths = action_lengths.cpu().tolist()
        bounds = torch.stack(
            [option_mask.sum(dim=1), batch["min_count"], batch["max_count"]],
            dim=1,
        ).cpu().tolist()
        result = [
            normalize_model_action(
                raw_actions[bi][: int(lengths[bi])],
                int(bounds[bi][0]),
                int(bounds[bi][1]),
                int(bounds[bi][2]),
            )
            for bi in range(batch_size)
        ]
        return result, logprobs, entropies, outputs["value"]


def flatten_card_semantic_index(
    semantic_ids: torch.Tensor,
    semantic_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the padded card-feature table into EmbeddingBag offsets."""
    if semantic_ids.shape != semantic_mask.shape:
        raise ValueError("semantic ids and mask must have the same shape")
    rows: list[torch.Tensor] = []
    offsets = torch.zeros(semantic_ids.size(0) + 1, dtype=torch.long)
    total = 0
    for card_id in range(semantic_ids.size(0)):
        values = semantic_ids[card_id][semantic_mask[card_id]]
        rows.append(values)
        total += int(values.numel())
        offsets[card_id + 1] = total
    flat = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long)
    return flat, offsets


def attach_card_semantics(
    model: EntityPointerPolicy,
    semantic_config: str | Path,
) -> tuple[EntityPointerPolicy, dict[str, Any]]:
    """Add card semantic embeddings while preserving all compatible checkpoint weights."""
    semantic_ids, semantic_mask, metadata = load_card_semantic_index(
        semantic_config,
        max_card_id=model.config.max_card_id,
    )
    flat_ids, offsets = flatten_card_semantic_index(semantic_ids, semantic_mask)
    desired_gate_classes = int(getattr(model.config, "num_cores", 1))
    if model.config.card_semantic_vocab_size > 0:
        if model.config.card_semantic_vocab_size != int(metadata["feature_count"]):
            raise ValueError("checkpoint semantic vocabulary does not match the supplied config")
        if model.config.card_semantic_max_features != int(metadata["max_features_per_card"]):
            raise ValueError("checkpoint semantic feature width does not match the supplied config")
        if model.card_semantic_ids.shape != semantic_ids.shape:
            raise ValueError("checkpoint semantic card index shape does not match the supplied config")
        if model.config.card_semantic_total_features != int(flat_ids.numel()):
            raise ValueError("checkpoint semantic flat index shape does not match the supplied config")
        model.card_semantic_ids.copy_(semantic_ids.to(model.card_semantic_ids.device))
        model.card_semantic_mask.copy_(semantic_mask.to(model.card_semantic_mask.device))
        model.card_semantic_flat_ids.copy_(flat_ids.to(model.card_semantic_flat_ids.device))
        model.card_semantic_offsets.copy_(offsets.to(model.card_semantic_offsets.device))
        metadata = dict(metadata, already_enabled=True)
        return model, metadata

    config = replace(
        model.config,
        card_semantic_vocab_size=int(metadata["feature_count"]),
        card_semantic_max_features=int(metadata["max_features_per_card"]),
        card_semantic_total_features=int(flat_ids.numel()),
        card_semantic_gate_classes=desired_gate_classes,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CARD_SEMANTIC_INIT_SEED)
        upgraded = type(model)(config)
    incompatible = upgraded.load_state_dict(model.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected keys while adding card semantics: {incompatible.unexpected_keys}")
    allowed_missing = (
        "card_semantic_embedding.",
        "semantic_gate",
        "card_semantic_ids",
        "card_semantic_mask",
        "card_semantic_flat_ids",
        "card_semantic_offsets",
    )
    invalid_missing = [
        name for name in incompatible.missing_keys if not name.startswith(allowed_missing)
    ]
    if invalid_missing:
        raise RuntimeError(f"unexpected missing keys while adding card semantics: {invalid_missing}")
    upgraded.card_semantic_ids.copy_(semantic_ids)
    upgraded.card_semantic_mask.copy_(semantic_mask)
    upgraded.card_semantic_flat_ids.copy_(flat_ids)
    upgraded.card_semantic_offsets.copy_(offsets)
    device = next(model.parameters()).device
    upgraded.to(device)
    metadata = dict(
        metadata,
        already_enabled=False,
        initialization_seed=CARD_SEMANTIC_INIT_SEED,
    )
    return upgraded, metadata


def attach_card_interaction_semantics(
    model: EntityPointerPolicy,
    interaction_config: str | Path,
) -> tuple[EntityPointerPolicy, dict[str, Any]]:
    """Attach a zero-gated, checkpoint-compatible interaction-semantic branch."""
    compiled = load_compiled_interaction_index(
        interaction_config,
        max_card_id=model.config.max_card_id,
        max_attack_id=model.config.max_attack_id,
    )
    metadata = compiled.metadata
    relation_features = int(metadata["relation_feature_count"])
    desired_gate_classes = int(getattr(model.config, "num_cores", 1))

    buffer_mapping = {
        "card_interaction_predicates": compiled.card_predicates,
        "card_interaction_protection_channels": compiled.card_protection_channels,
        "card_interaction_protection_scope": compiled.card_protection_scope,
        "card_interaction_protection_mode": compiled.card_protection_mode,
        "card_interaction_protection_amount": compiled.card_protection_amount,
        "card_interaction_source_predicates": compiled.card_protection_source_predicates,
        "card_interaction_target_predicates": compiled.card_protection_target_predicates,
        "card_interaction_protection_ambiguous": compiled.card_protection_ambiguous,
        "attack_interaction_channels": compiled.attack_channels,
        "attack_interaction_future_mode": compiled.attack_future_mode,
    }

    def copy_compiled_buffers(target: EntityPointerPolicy) -> None:
        for name, value in buffer_mapping.items():
            destination = getattr(target, name)
            destination.copy_(value.to(destination.device))

    if model.config.card_interaction_relation_features > 0:
        if model.config.card_interaction_runtime_version != 2:
            raise ValueError(
                "cannot replace a legacy interaction branch in place; attach to its parent checkpoint"
            )
        if model.config.card_interaction_relation_features != relation_features:
            raise ValueError(
                "checkpoint interaction relation width does not match supplied config"
            )
        if model.config.card_interaction_split_gates:
            copy_compiled_buffers(model)
            return model, dict(
                metadata,
                already_enabled=True,
                split_gates=True,
                gate_names=(
                    "effectiveness",
                    "block_immunity",
                    "future_protection",
                ),
            )

    config = replace(
        model.config,
        card_interaction_relation_features=relation_features,
        card_interaction_gate_classes=desired_gate_classes,
        card_interaction_runtime_version=2,
        card_interaction_split_gates=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CARD_INTERACTION_INIT_SEED)
        upgraded = type(model)(config)
    incompatible = upgraded.load_state_dict(model.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "unexpected keys while adding interaction semantics: "
            f"{incompatible.unexpected_keys}"
        )
    allowed_missing = ("interaction_", "card_interaction_", "attack_interaction_channels", "attack_interaction_future_mode")
    invalid_missing = [
        name for name in incompatible.missing_keys if not name.startswith(allowed_missing)
    ]
    if invalid_missing:
        raise RuntimeError(
            f"unexpected missing keys while adding interaction semantics: {invalid_missing}"
        )
    copy_compiled_buffers(upgraded)
    device = next(model.parameters()).device
    upgraded.to(device)
    return upgraded, dict(
        metadata,
        already_enabled=False,
        upgraded_existing_branch=bool(
            model.config.card_interaction_relation_features > 0
        ),
        split_gates=True,
        gate_names=(
            "effectiveness",
            "block_immunity",
            "future_protection",
        ),
        initialization_seed=CARD_INTERACTION_INIT_SEED,
        initial_gate=0.0,
    )


def attach_card_rule_semantics(
    model: EntityPointerPolicy,
    rule_config: str | Path,
) -> tuple[EntityPointerPolicy, dict[str, Any]]:
    """Attach the full-pool rule-semantic and option-relation branch."""
    (
        card_flat,
        card_offsets,
        attack_flat,
        attack_offsets,
        metadata,
    ) = load_card_rule_semantic_index(
        rule_config,
        max_card_id=model.config.max_card_id,
        max_attack_id=model.config.max_attack_id,
    )
    desired_gate_classes = int(getattr(model.config, "num_cores", 1))
    if model.config.card_rule_semantic_vocab_size > 0:
        expected = (
            int(metadata["feature_count"]),
            int(card_flat.numel()),
            int(attack_flat.numel()),
        )
        actual = (
            int(model.config.card_rule_semantic_vocab_size),
            int(model.config.card_rule_card_total_features),
            int(model.config.card_rule_attack_total_features),
        )
        if actual != expected:
            raise ValueError(
                f"checkpoint rule-semantic dimensions {actual} != supplied {expected}"
            )
        model.card_rule_flat_ids.copy_(card_flat.to(model.card_rule_flat_ids.device))
        model.card_rule_offsets.copy_(card_offsets.to(model.card_rule_offsets.device))
        model.attack_rule_flat_ids.copy_(
            attack_flat.to(model.attack_rule_flat_ids.device)
        )
        model.attack_rule_offsets.copy_(
            attack_offsets.to(model.attack_rule_offsets.device)
        )
        return model, dict(metadata, already_enabled=True)

    config = replace(
        model.config,
        card_rule_semantic_vocab_size=int(metadata["feature_count"]),
        card_rule_card_total_features=int(card_flat.numel()),
        card_rule_attack_total_features=int(attack_flat.numel()),
        card_rule_gate_classes=desired_gate_classes,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CARD_RULE_SEMANTIC_INIT_SEED)
        upgraded = type(model)(config)
    incompatible = upgraded.load_state_dict(model.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "unexpected keys while adding card rule semantics: "
            f"{incompatible.unexpected_keys}"
        )
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(OPTIONAL_CARD_RULE_PREFIXES)
    ]
    if invalid_missing:
        raise RuntimeError(
            f"unexpected missing keys while adding card rule semantics: {invalid_missing}"
        )
    upgraded.card_rule_flat_ids.copy_(card_flat)
    upgraded.card_rule_offsets.copy_(card_offsets)
    upgraded.attack_rule_flat_ids.copy_(attack_flat)
    upgraded.attack_rule_offsets.copy_(attack_offsets)
    device = next(model.parameters()).device
    upgraded.to(device)
    return upgraded, dict(
        metadata,
        already_enabled=False,
        initialization_seed=CARD_RULE_SEMANTIC_INIT_SEED,
        initial_semantic_gate=0.0,
        initial_modifier_gate=0.0,
        initial_relation_gate=0.0,
    )


def attach_zone_summaries(
    model: EntityPointerPolicy,
    *,
    gate_classes: int | None = None,
    initialization_seed: int = ZONE_SUMMARY_INIT_SEED,
) -> tuple[EntityPointerPolicy, dict[str, Any]]:
    """Upgrade a legacy policy with the opt-in zone-summary token branch.

    Shared embeddings/encoder/decoder weights are copied exactly.  New zone
    parameters use their constructor initialization (zero residual gate), so
    callers can train a dedicated semantic/zone probe from the same parent.
    ``gate_classes`` defaults to the deck-conditioned core count when present.
    """
    if model.config.zone_summary_enabled and model.zone_summary_gate is not None:
        return model, {
            "already_enabled": True,
            "gate_classes": int(model.config.zone_summary_gate_classes),
        }
    desired_gate_classes = int(
        gate_classes if gate_classes is not None else getattr(model.config, "num_cores", 1)
    )
    if desired_gate_classes < 1:
        raise ValueError("zone summary gate_classes must be positive")
    config = replace(
        model.config,
        zone_summary_enabled=True,
        zone_summary_gate_classes=desired_gate_classes,
    )
    # Construct missing tensors under a fixed, local RNG fork.  Policy and KL
    # reference upgrades made from the same legacy parent then receive
    # identical zone initialization, independent of call order; shared old
    # weights are copied immediately below.  The caller can record the seed in
    # provenance or provide a different experiment seed explicitly.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(initialization_seed))
        upgraded = type(model)(config)
    incompatible = upgraded.load_state_dict(model.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected keys while adding zone summaries: {incompatible.unexpected_keys}")
    allowed_missing = OPTIONAL_ZONE_SUMMARY_PREFIXES
    invalid_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith(allowed_missing)
    ]
    if invalid_missing:
        raise RuntimeError(f"unexpected missing keys while adding zone summaries: {invalid_missing}")
    device = next(model.parameters()).device
    upgraded.to(device)
    metadata = {
        "already_enabled": False,
        "gate_classes": desired_gate_classes,
        "init_seed": ZONE_SUMMARY_INIT_SEED,
        "group_count": len(ZONE_SUMMARY_GROUPS),
        "group_names": list(ZONE_SUMMARY_NAMES),
        "initialization_seed": int(initialization_seed),
    }
    return upgraded, metadata


def checkpoint_payload(model: EntityPointerPolicy, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "extra": dict(extra or {}),
    }


def load_state_dict_with_optional_auxiliary_heads(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected checkpoint keys: {incompatible.unexpected_keys[:8]}"
        )
    # Zone tensors are optional only when the *target config* has no zone
    # branch (legacy checkpoints).  Once a checkpoint declares the branch,
    # silently random-initializing missing zone weights would invalidate a
    # purported strict ablation, so require every zone key there.
    allowed_missing_prefixes = OPTIONAL_AUXILIARY_HEAD_PREFIXES
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing:
        raise RuntimeError(f"unexpected missing checkpoint keys: {invalid_missing[:8]}")


def load_policy_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[EntityPointerPolicy, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_version") != MODEL_VERSION:
        raise RuntimeError(f"unsupported model version: {payload.get('model_version')}")
    config = EntityPointerConfig(**payload["model_config"])
    model = EntityPointerPolicy(config)
    load_state_dict_with_optional_auxiliary_heads(model, payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


class PurePolicyRuntime:
    def __init__(self, checkpoint: str | Path, *, threads: int | None = None):
        if threads is None:
            threads = int(os.environ.get("PURE_POLICY_TORCH_THREADS", "1"))
        torch.set_num_threads(max(1, threads))
        self.device = torch.device("cpu")
        self.codec = PolicyCodecV1()
        self.model, self.payload = load_policy_checkpoint(checkpoint, self.device)
        self.fallback_count = 0

    @torch.inference_mode()
    def act(self, obs: dict[str, Any]) -> list[int]:
        encoded = self.codec.encode(obs)
        row = encoded.to_record()
        row.update(
            {
                "teacher_action": [],
                "value_target": 0.0,
                "next_prize_target": 0.0,
                "finish_reason": 0,
                "opponent_class": 0,
            }
        )
        batch = collate_decision_records([row])
        return self.model.greedy_decode_batch(batch)[0]

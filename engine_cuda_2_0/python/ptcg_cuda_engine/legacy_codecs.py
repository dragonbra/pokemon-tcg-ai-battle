from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


POLICY_CODEC_V1_GLOBAL_CAT_WIDTH = 8
POLICY_CODEC_V1_GLOBAL_NUM_WIDTH = 16
POLICY_CODEC_V1_ENTITY_CAT_WIDTH = 6
POLICY_CODEC_V1_ENTITY_NUM_WIDTH = 10
POLICY_CODEC_V1_OPTION_CAT_WIDTH = 12

IDONLY_CODEC_V1_GLOBAL_CAT_WIDTH = 4
IDONLY_CODEC_V1_GLOBAL_NUM_WIDTH = 12
IDONLY_CODEC_V1_ENTITY_CAT_WIDTH = 7
IDONLY_CODEC_V1_ENTITY_NUM_WIDTH = 5
IDONLY_CODEC_V1_OPTION_CAT_WIDTH = 12


def marnie_prize_control_v4_static_fields(
    registered_deck: Sequence[int],
    device: Any,
) -> dict[str, Any]:
    """Build the exact resident prize inputs for the promoted control model.

    The frozen ``marnie_prize_control_v4`` checkpoint has
    ``prize_feature_mode=control``.  Its forward pass discards every dynamic
    PrizeLedger feature except registered count, while card IDs and registered
    counts are invariant properties of the 60-card deck.  These tensors can
    therefore be uploaded once without changing that checkpoint's semantics.
    A future belief-mode checkpoint must use a stateful device ledger instead.
    """

    import torch

    deck = [int(card) for card in registered_deck]
    if len(deck) != 60 or any(card <= 0 for card in deck):
        raise ValueError("registered deck must contain 60 positive card IDs")
    counts = Counter(deck)
    card_ids = sorted(counts)
    prize_card = torch.tensor(
        card_ids, dtype=torch.long, device=device
    ).view(1, -1)
    prize_num = torch.zeros(
        (1, len(card_ids), 12), dtype=torch.float32, device=device
    )
    prize_num[0, :, 0] = torch.tensor(
        [counts[card] / 4.0 for card in card_ids],
        dtype=torch.float32,
        device=device,
    )
    return {
        "prize_card": prize_card,
        "prize_num": prize_num,
        "prize_mask": torch.ones_like(prize_card, dtype=torch.bool),
    }


def _require_shape(name: str, value: Any, rank: int, width: int | None = None) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {value.ndim}")
    if width is not None and value.shape[-1] != width:
        raise ValueError(
            f"{name} must have width {width}, got {value.shape[-1]}"
        )


def policy_codec_v1_to_idonly_codec_v1(
    batch: Mapping[str, Any],
    *,
    max_card_id: int = 2048,
    max_action_steps: int = 16,
    target_entity_capacity: int | None = None,
    target_option_capacity: int | None = None,
) -> dict[str, Any]:
    """Convert the resident official codec to the frozen legacy ID-only ABI.

    Both codecs describe the same public observation, but the legacy model has
    different field widths, zone IDs and option-column order. The conversion
    stays on the input tensor device and compacts revealed Prize entities to
    match the legacy CPU codec, which never emitted Prize-zone entities.
    """

    import torch

    if max_card_id <= 0 or max_action_steps <= 0:
        raise ValueError("legacy codec limits must be positive")

    required = {
        "global_cat",
        "global_num",
        "entity_cat",
        "entity_num",
        "entity_parent",
        "entity_mask",
        "option_cat",
        "option_mask",
        "min_count",
        "max_count",
    }
    missing = required - set(batch)
    if missing:
        raise KeyError(f"PolicyCodecV1 batch is missing fields: {sorted(missing)}")

    global_cat = batch["global_cat"]
    global_num = batch["global_num"]
    entity_cat = batch["entity_cat"]
    entity_num = batch["entity_num"]
    entity_parent = batch["entity_parent"]
    entity_mask = batch["entity_mask"].bool()
    option_cat = batch["option_cat"]
    option_mask = batch["option_mask"].bool()

    _require_shape(
        "global_cat", global_cat, 2, POLICY_CODEC_V1_GLOBAL_CAT_WIDTH
    )
    _require_shape(
        "global_num", global_num, 2, POLICY_CODEC_V1_GLOBAL_NUM_WIDTH
    )
    _require_shape(
        "entity_cat", entity_cat, 3, POLICY_CODEC_V1_ENTITY_CAT_WIDTH
    )
    _require_shape(
        "entity_num", entity_num, 3, POLICY_CODEC_V1_ENTITY_NUM_WIDTH
    )
    _require_shape("entity_parent", entity_parent, 2)
    _require_shape("entity_mask", entity_mask, 2)
    _require_shape(
        "option_cat", option_cat, 3, POLICY_CODEC_V1_OPTION_CAT_WIDTH
    )
    _require_shape("option_mask", option_mask, 2)

    batch_size, source_entity_capacity = entity_mask.shape
    source_option_capacity = option_mask.shape[1]
    if entity_cat.shape[:2] != (batch_size, source_entity_capacity):
        raise ValueError("entity_cat capacity does not match entity_mask")
    if entity_num.shape[:2] != (batch_size, source_entity_capacity):
        raise ValueError("entity_num capacity does not match entity_mask")
    if entity_parent.shape != (batch_size, source_entity_capacity):
        raise ValueError("entity_parent capacity does not match entity_mask")
    if option_cat.shape[:2] != (batch_size, source_option_capacity):
        raise ValueError("option_cat capacity does not match option_mask")
    entity_capacity = (
        source_entity_capacity
        if target_entity_capacity is None
        else int(target_entity_capacity)
    )
    option_capacity = (
        source_option_capacity
        if target_option_capacity is None
        else int(target_option_capacity)
    )
    if entity_capacity < source_entity_capacity:
        raise ValueError(
            "target entity capacity cannot truncate the resident codec"
        )
    if option_capacity < source_option_capacity:
        raise ValueError(
            "target option capacity cannot truncate the resident codec"
        )

    legacy_global_cat = torch.stack(
        (global_cat[:, 0], global_cat[:, 1], global_cat[:, 2], global_cat[:, 7]),
        dim=1,
    )
    bench_count = torch.round(global_num[:, 14] * 5.0) + torch.round(
        global_num[:, 15] * 5.0
    )
    legacy_global_num = torch.stack(
        (
            global_num[:, 0],
            global_num[:, 1],
            global_num[:, 4],
            global_num[:, 5],
            global_num[:, 6],
            global_num[:, 7],
            global_num[:, 8],
            global_num[:, 9],
            global_num[:, 10],
            global_num[:, 11],
            global_num[:, 12],
            bench_count * 0.1,
        ),
        dim=1,
    )

    policy_zone = entity_cat[..., 2]
    keep_entity = entity_mask & policy_zone.ne(5) & policy_zone.ne(9)
    dense_entity_index = keep_entity.long().cumsum(dim=1) - 1
    sink_entity = entity_capacity
    entity_destination = torch.where(
        keep_entity,
        dense_entity_index,
        torch.full_like(dense_entity_index, sink_entity),
    )

    legacy_zone = torch.where(
        policy_zone.ge(10),
        policy_zone - 2,
        torch.where(policy_zone.ge(6), policy_zone - 1, policy_zone),
    )
    safe_parent = entity_parent.long().clamp(
        min=0, max=source_entity_capacity - 1
    )
    parent_exists = entity_parent.ge(0) & keep_entity.gather(1, safe_parent)
    legacy_parent = torch.where(
        parent_exists,
        dense_entity_index.gather(1, safe_parent) + 1,
        torch.zeros_like(safe_parent),
    )
    legacy_entity_values = torch.stack(
        (
            entity_cat[..., 0].clamp(min=0, max=max_card_id),
            entity_cat[..., 1],
            legacy_zone,
            entity_cat[..., 3],
            entity_cat[..., 4],
            entity_cat[..., 5],
            legacy_parent,
        ),
        dim=2,
    )
    legacy_entity_values = torch.where(
        keep_entity.unsqueeze(2),
        legacy_entity_values,
        torch.zeros_like(legacy_entity_values),
    )
    legacy_entity_cat = torch.zeros(
        (
            batch_size,
            entity_capacity + 1,
            IDONLY_CODEC_V1_ENTITY_CAT_WIDTH,
        ),
        dtype=entity_cat.dtype,
        device=entity_cat.device,
    )
    legacy_entity_cat.scatter_(
        1,
        entity_destination.unsqueeze(2).expand(
            -1, -1, IDONLY_CODEC_V1_ENTITY_CAT_WIDTH
        ),
        legacy_entity_values,
    )
    legacy_entity_num_values = torch.stack(
        (
            entity_num[..., 2],
            entity_num[..., 3],
            entity_num[..., 4],
            entity_num[..., 5],
            entity_num[..., 6],
        ),
        dim=2,
    )
    legacy_entity_num_values = torch.where(
        keep_entity.unsqueeze(2),
        legacy_entity_num_values,
        torch.zeros_like(legacy_entity_num_values),
    )
    legacy_entity_num = torch.zeros(
        (
            batch_size,
            entity_capacity + 1,
            IDONLY_CODEC_V1_ENTITY_NUM_WIDTH,
        ),
        dtype=entity_num.dtype,
        device=entity_num.device,
    )
    legacy_entity_num.scatter_(
        1,
        entity_destination.unsqueeze(2).expand(
            -1, -1, IDONLY_CODEC_V1_ENTITY_NUM_WIDTH
        ),
        legacy_entity_num_values,
    )
    legacy_entity_mask = torch.zeros(
        (batch_size, entity_capacity + 1),
        dtype=torch.bool,
        device=entity_mask.device,
    )
    legacy_entity_mask.scatter_(1, entity_destination, keep_entity)

    def remap_option_entity(reference: Any) -> Any:
        old_index = reference.long() - 1
        safe_index = old_index.clamp(min=0, max=source_entity_capacity - 1)
        valid = reference.gt(0) & keep_entity.gather(1, safe_index)
        return torch.where(
            valid,
            dense_entity_index.gather(1, safe_index) + 1,
            torch.zeros_like(safe_index),
        )

    source_entity = remap_option_entity(option_cat[..., 8])
    target_entity = remap_option_entity(option_cat[..., 9])
    option_position = torch.arange(
        1,
        source_option_capacity + 1,
        dtype=option_cat.dtype,
        device=option_cat.device,
    ).view(1, -1).expand(batch_size, -1)
    legacy_option_values = torch.stack(
        (
            option_cat[..., 0],
            option_cat[..., 1],
            option_cat[..., 2],
            option_cat[..., 3],
            option_cat[..., 4].clamp(min=0, max=max_card_id),
            option_cat[..., 5].clamp(min=0, max=max_card_id),
            option_cat[..., 7],
            option_cat[..., 10],
            option_cat[..., 11],
            source_entity,
            target_entity,
            option_position,
        ),
        dim=2,
    )
    legacy_option_values = torch.where(
        option_mask.unsqueeze(2),
        legacy_option_values,
        torch.zeros_like(legacy_option_values),
    )
    legacy_option_cat = torch.zeros(
        (batch_size, option_capacity, IDONLY_CODEC_V1_OPTION_CAT_WIDTH),
        dtype=option_cat.dtype,
        device=option_cat.device,
    )
    legacy_option_cat[:, :source_option_capacity] = legacy_option_values
    legacy_option_mask = torch.zeros(
        (batch_size, option_capacity),
        dtype=torch.bool,
        device=option_mask.device,
    )
    legacy_option_mask[:, :source_option_capacity] = option_mask
    action_min_count = batch["min_count"].long().view(batch_size).clamp(min=0)
    action_max_count = batch["max_count"].long().view(batch_size).clamp(min=0)

    return {
        "global_cat": legacy_global_cat,
        "global_num": legacy_global_num,
        "entity_cat": legacy_entity_cat[:, :entity_capacity],
        "entity_num": legacy_entity_num[:, :entity_capacity],
        "entity_mask": legacy_entity_mask[:, :entity_capacity],
        "option_cat": legacy_option_cat,
        "option_mask": legacy_option_mask,
        "min_count": batch["min_count"]
        .long()
        .view(batch_size)
        .clamp(min=0, max=max_action_steps),
        "max_count": batch["max_count"]
        .long()
        .view(batch_size)
        .clamp(min=0, max=max_action_steps),
        "action_min_count": action_min_count,
        "action_max_count": action_max_count,
    }

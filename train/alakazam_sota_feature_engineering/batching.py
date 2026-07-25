from __future__ import annotations

import torch
from torch import Tensor

from train.alakazam_sota_model.model import collate_id_only as _collate_id_only


def collate_id_only(rows: list[dict[str, object]]) -> dict[str, Tensor]:
    """Collate 0010 rows and append E3 fields when the versioned shard provides them."""

    batch = _collate_id_only(rows)  # type: ignore[arg-type]
    if not rows:
        return batch
    batch_size, option_count = batch["option_mask"].shape
    if "option_primitive_cat" in rows[0]:
        primitive = torch.zeros((batch_size, option_count, 4), dtype=torch.long)
        context = torch.zeros((batch_size, 4), dtype=torch.long)
        for index, row in enumerate(rows):
            values = row.get("option_primitive_cat")
            if not isinstance(values, list):
                raise ValueError("mixed dataset schemas: missing option_primitive_cat")
            if values:
                primitive[index, : len(values)] = torch.tensor(values, dtype=torch.long)
            context[index] = torch.tensor(row["action_context_cat"], dtype=torch.long)
        batch["option_primitive_cat"] = primitive
        batch["action_context_cat"] = context
    if "action_history_cat" in rows[0]:
        history_count = max(1, max(len(row["action_history_cat"]) for row in rows))
        history = torch.zeros((batch_size, history_count, 6), dtype=torch.long)
        history_mask = torch.zeros((batch_size, history_count), dtype=torch.bool)
        for index, row in enumerate(rows):
            values = row["action_history_cat"]
            if values:
                history[index, : len(values)] = torch.tensor(values, dtype=torch.long)
                history_mask[index, : len(values)] = True
        batch["action_history_cat"] = history
        batch["action_history_mask"] = history_mask
    if "transition_delta" in rows[0]:
        batch["transition_delta"] = torch.tensor(
            [row["transition_delta"] for row in rows], dtype=torch.float32
        )
        batch["transition_delta_mask"] = torch.tensor(
            [row["transition_delta_mask"] for row in rows], dtype=torch.bool
        )
        batch["transition_turn_changed"] = torch.tensor(
            [row["transition_turn_changed"] for row in rows], dtype=torch.float32
        )
        batch["transition_next_mask"] = torch.tensor(
            [row["transition_next_mask"] for row in rows], dtype=torch.bool
        )
    return batch


def permute_candidates(batch: dict[str, Tensor], *, seed: int) -> dict[str, Tensor]:
    """Permute real candidates per row and remap expert indices without reordering selections."""

    result = {key: value.clone() for key, value in batch.items()}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    option_cat = result["option_cat"]
    option_mask = result["option_mask"]
    targets = result["targets"]
    padded_option_count = option_cat.size(1)
    old_to_new_all = torch.arange(padded_option_count).repeat(option_cat.size(0), 1)
    new_to_old_all = old_to_new_all.clone()
    for row in range(option_cat.size(0)):
        count = int(option_mask[row].sum().item())
        if count <= 1:
            continue
        new_to_old = torch.randperm(count, generator=generator)
        old_to_new = torch.empty_like(new_to_old)
        old_to_new[new_to_old] = torch.arange(count)
        old_to_new_all[row, :count] = old_to_new
        new_to_old_all[row, :count] = new_to_old
        original = option_cat[row, :count].clone()
        option_cat[row, :count] = original[new_to_old]
        # Every option-aligned feature must follow the same permutation.  Leaving
        # primitive semantics in their original order silently pairs them with a
        # different candidate during augmentation and breaks permutation equivariance.
        if "option_primitive_cat" in result:
            original_primitive = result["option_primitive_cat"][row, :count].clone()
            result["option_primitive_cat"][row, :count] = original_primitive[new_to_old]
        # Keep the raw position field internally consistent for audits. E1 does not consume it.
        option_cat[row, :count, 11] = torch.arange(1, count + 1)
        valid = (targets[row] >= 0) & (targets[row] < count)
        targets[row, valid] = old_to_new[targets[row, valid]]
        stop = targets[row] == padded_option_count
        targets[row, stop] = padded_option_count
    result["permutation_old_to_new"] = old_to_new_all
    result["permutation_new_to_old"] = new_to_old_all
    return result


__all__ = ["collate_id_only", "permute_candidates"]

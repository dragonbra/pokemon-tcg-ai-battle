"""Re-evaluate stored conditional allocations from pre-action visible features."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from ..semantic_runtime.action_boundary.dragapult import StableTargetIdentity, enumerate_allocations
from ..semantic_runtime.action_boundary.macro_planner import MacroPlanner


@dataclass(frozen=True)
class _AllocationEvaluation:
    row: int
    chosen: int
    root_index: int
    features: Tensor
    target_embeddings: Tensor


def _prepare_evaluations(model, validated, state, options: Tensor,
                         macro_actions: tuple[dict | None, ...]) -> list[_AllocationEvaluation]:
    """Prepare macros with one batched device lookup instead of scalar GPU syncs."""
    planner = MacroPlanner(model.allocation_head)
    specifications = []
    for row, macro in enumerate(macro_actions):
        if not macro:
            continue
        serials = tuple(int(value) for value in macro["targets"])
        counters = tuple(int(value) for value in macro["counters"])
        stored_visible = macro.get("visible_targets")
        if stored_visible is not None and len(stored_visible) != len(serials):
            raise ValueError("stored Phantom visible-target snapshot has the wrong length")
        specifications.append((row, macro, serials, counters, stored_visible))
    if not specifications:
        return []

    flat_rows = [
        row for row, _macro, serials, _counters, _stored in specifications
        for _serial in serials
    ]
    flat_serials = [
        serial for _row, _macro, serials, _counters, _stored in specifications
        for serial in serials
    ]
    row_index = torch.tensor(flat_rows, dtype=torch.long, device=options.device)
    serial_index = torch.tensor(flat_serials, dtype=torch.long, device=options.device)
    selected_mask = validated.card_mask.index_select(0, row_index)
    selected_cat_rows = validated.card_cat.index_select(0, row_index)
    matches = (
        selected_mask
        & selected_cat_rows[:, :, 1].eq(serial_index.unsqueeze(1) + 1)
        & selected_cat_rows[:, :, 2].eq(2)
        & selected_cat_rows[:, :, 3].eq(6)
    )
    match_counts = matches.sum(dim=1)
    if not bool(match_counts.eq(1).all()):
        invalid = int(match_counts.ne(1).nonzero(as_tuple=False)[0])
        raise ValueError(
            f"stored Phantom target serial {flat_serials[invalid]} is not uniquely visible"
        )
    card_indices = matches.to(torch.int8).argmax(dim=1)
    selected_cat = selected_cat_rows[
        torch.arange(len(flat_rows), device=options.device), card_indices
    ].detach().cpu()
    selected_num = validated.card_num.index_select(0, row_index)[
        torch.arange(len(flat_rows), device=options.device), card_indices
    ].detach().cpu()
    selected_embeddings = state.cards[row_index, card_indices]

    prepared = []
    offset = 0
    for row, macro, serials, counters, stored_visible in specifications:
        identities = []
        raw_targets = []
        embeddings = []
        for target_offset, serial in enumerate(serials):
            flat_index = offset + target_offset
            cat = selected_cat[flat_index]
            numeric = selected_num[flat_index]
            identities.append(StableTargetIdentity(1, serial, int(cat[0]), int(cat[4]) - 1))
            if stored_visible is None:
                raw_targets.append({
                    "serial": serial, "id": int(cat[0]), "benchSlot": int(cat[4]) - 1,
                    "hp": float(numeric[0]), "maxHp": float(numeric[1]),
                    "energyCards": [None] * int(numeric[2]),
                    "preEvolution": [None] * int(numeric[5]),
                    "statusBits": max(0, int(cat[6]) - 1),
                })
            else:
                visible = dict(stored_visible[target_offset])
                if (
                    int(visible.get("serial", -1)) != serial
                    or int(visible.get("id", -1)) != int(cat[0])
                    or int(visible.get("benchSlot", -1)) != int(cat[4]) - 1
                ):
                    raise ValueError("stored Phantom visible-target identity drifted")
                raw_targets.append(visible)
            embeddings.append(selected_embeddings[flat_index])
        offset += len(serials)
        allocations = enumerate_allocations(identities)
        if len(allocations) == 1:
            continue
        allocation_index = next(
            (index for index, item in enumerate(allocations) if item.counters == counters), None
        )
        if allocation_index is None:
            raise ValueError("stored counters are not a canonical Phantom allocation")
        features = planner.visible_features(raw_targets, allocations).to(
            device=options.device, dtype=options.dtype
        )
        count, targets = features.shape[:2]
        target_embeddings = torch.stack(embeddings).unsqueeze(0).expand(
            count, -1, -1
        )
        prepared.append(_AllocationEvaluation(
            row=row,
            chosen=allocation_index,
            root_index=int(macro["root_index"]),
            features=features,
            target_embeddings=target_embeddings,
        ))
    return prepared


def _evaluate_parameter_actions_scalar(model, validated, state, options: Tensor,
                                       macro_actions: tuple[dict | None, ...]) -> tuple[Tensor, Tensor]:
    """Regression oracle matching the original one-GPU-launch-per-macro path."""
    logprob = torch.zeros(len(macro_actions), device=options.device)
    entropy = torch.zeros_like(logprob)
    for item in _prepare_evaluations(
        model, validated, state, options, macro_actions
    ):
        count, targets = item.features.shape[:2]
        logits = model.allocation_head(
            state.summary[item.row:item.row + 1],
            options[item.row:item.row + 1, item.root_index],
            item.target_embeddings.unsqueeze(0), item.features.unsqueeze(0),
            torch.ones((1, count, targets), dtype=torch.bool, device=options.device),
            torch.ones((1, count), dtype=torch.bool, device=options.device),
        )[0]
        distribution = Categorical(logits=logits.float())
        chosen = torch.tensor(item.chosen, device=options.device)
        logprob[item.row] = distribution.log_prob(chosen)
        entropy[item.row] = distribution.entropy()
    return logprob, entropy


def evaluate_parameter_actions(model, validated, state, options: Tensor,
                               macro_actions: tuple[dict | None, ...]) -> tuple[Tensor, Tensor]:
    prepared = _prepare_evaluations(
        model, validated, state, options, macro_actions
    )
    logprob = torch.zeros(len(macro_actions), device=options.device)
    entropy = torch.zeros_like(logprob)
    groups: dict[tuple[int, int], list[_AllocationEvaluation]] = defaultdict(list)
    for item in prepared:
        groups[tuple(item.features.shape[:2])].append(item)
    for (count, targets), items in groups.items():
        rows = torch.tensor(
            [item.row for item in items], dtype=torch.long, device=options.device
        )
        roots = torch.tensor(
            [item.root_index for item in items], dtype=torch.long,
            device=options.device,
        )
        logits = model.allocation_head(
            state.summary.index_select(0, rows),
            options[rows, roots],
            torch.stack([item.target_embeddings for item in items]),
            torch.stack([item.features for item in items]),
            torch.ones(
                (len(items), count, targets), dtype=torch.bool,
                device=options.device,
            ),
            torch.ones(
                (len(items), count), dtype=torch.bool, device=options.device
            ),
        )
        distribution = Categorical(logits=logits.float())
        chosen = torch.tensor(
            [item.chosen for item in items], dtype=torch.long,
            device=options.device,
        )
        logprob = logprob.index_copy(0, rows, distribution.log_prob(chosen))
        entropy = entropy.index_copy(0, rows, distribution.entropy())
    return logprob, entropy


__all__ = ["evaluate_parameter_actions"]

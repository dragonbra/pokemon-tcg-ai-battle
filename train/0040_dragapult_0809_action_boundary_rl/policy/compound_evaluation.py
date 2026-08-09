"""Re-evaluate stored conditional allocations from pre-action visible features."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical

from ..action_boundary.dragapult import StableTargetIdentity, enumerate_allocations
from ..action_boundary.macro_planner import MacroPlanner


def evaluate_parameter_actions(model, validated, state, options: Tensor,
                               macro_actions: tuple[dict | None, ...]) -> tuple[Tensor, Tensor]:
    logprob = torch.zeros(len(macro_actions), device=options.device)
    entropy = torch.zeros_like(logprob)
    planner = MacroPlanner(model.allocation_head)
    for row, macro in enumerate(macro_actions):
        if not macro:
            continue
        serials = tuple(int(value) for value in macro["targets"])
        counters = tuple(int(value) for value in macro["counters"])
        identities = []
        raw_targets = []
        embeddings = []
        stored_visible = macro.get("visible_targets")
        if stored_visible is not None and len(stored_visible) != len(serials):
            raise ValueError("stored Phantom visible-target snapshot has the wrong length")
        for serial in serials:
            match = (
                validated.card_mask[row]
                & validated.card_cat[row, :, 1].eq(serial + 1)
                & validated.card_cat[row, :, 2].eq(2)
                & validated.card_cat[row, :, 3].eq(6)
            ).nonzero(as_tuple=False).flatten()
            if match.numel() != 1:
                raise ValueError(f"stored Phantom target serial {serial} is not uniquely visible")
            index = int(match[0])
            cat, numeric = validated.card_cat[row, index], validated.card_num[row, index]
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
                visible = dict(stored_visible[len(raw_targets)])
                if (
                    int(visible.get("serial", -1)) != serial
                    or int(visible.get("id", -1)) != int(cat[0])
                    or int(visible.get("benchSlot", -1)) != int(cat[4]) - 1
                ):
                    raise ValueError("stored Phantom visible-target identity drifted")
                raw_targets.append(visible)
            embeddings.append(state.cards[row, index])
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
        target_embeddings = torch.stack(embeddings).unsqueeze(0).expand(count, -1, -1).unsqueeze(0)
        logits = model.allocation_head(
            state.summary[row:row + 1], options[row:row + 1, int(macro["root_index"])],
            target_embeddings, features.unsqueeze(0),
            torch.ones((1, count, targets), dtype=torch.bool, device=options.device),
            torch.ones((1, count), dtype=torch.bool, device=options.device),
        )[0]
        distribution = Categorical(logits=logits.float())
        chosen = torch.tensor(allocation_index, device=options.device)
        logprob[row] = distribution.log_prob(chosen)
        entropy[row] = distribution.entropy()
    return logprob, entropy


__all__ = ["evaluate_parameter_actions"]

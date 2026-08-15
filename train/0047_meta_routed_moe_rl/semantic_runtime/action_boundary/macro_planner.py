"""Conditional Phantom Dive planner operating on one shared state encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Categorical

from .dragapult import DragapultDamageAllocation, enumerate_allocations


@dataclass(frozen=True, slots=True)
class PlannedMacro:
    allocation: DragapultDamageAllocation
    allocation_index: int
    allocation_logprob: float
    allocation_entropy: float


class MacroPlanner:
    """Samples `P(allocation | state, Phantom Dive)` after root selection."""

    def __init__(self, allocation_head) -> None:
        self.allocation_head = allocation_head

    @staticmethod
    def visible_features(targets: list[dict[str, Any]], allocations) -> Tensor:
        rows = []
        for allocation in allocations:
            candidate = []
            for raw, counters in zip(targets, allocation.counters, strict=True):
                hp = max(0.0, float(raw.get("hp", 0)))
                maximum_hp = max(hp, float(raw.get("maxHp", hp)))
                damage = max(0.0, maximum_hp - hp)
                counter_damage = float(counters * 10)
                remaining = max(0.0, hp - counter_damage)
                immediate_ko = float(counters > 0 and remaining <= 0)
                prizes = float(raw.get("prize", raw.get("prizeCount", 1))) if immediate_ko else 0.0
                energies = raw.get("energyCards") if isinstance(raw.get("energyCards"), list) else []
                status = float(raw.get("statusBits", 0))
                headroom = max(0.0, hp - 60.0)
                evolution_risk = float(bool(raw.get("preEvolution")))
                candidate.append([
                    float(counters), remaining, immediate_ko, prizes,
                    max(0.0, hp - counter_damage), float(raw.get("benchSlot", 0)),
                    damage, float(len(energies)), status, evolution_risk,
                    headroom, float(counters) / 6.0,
                ])
            rows.append(candidate)
        return torch.tensor(rows, dtype=torch.float32)

    def plan(
        self,
        *,
        state_summary: Tensor,
        root_option: Tensor,
        target_embeddings: Tensor,
        target_identities,
        visible_targets: list[dict[str, Any]],
        greedy: bool,
        generator: torch.Generator | None = None,
    ) -> PlannedMacro:
        allocations = enumerate_allocations(target_identities)
        if len(allocations) == 1:
            return PlannedMacro(allocations[0], 0, 0.0, 0.0)
        features = self.visible_features(visible_targets, allocations).to(
            device=target_embeddings.device, dtype=target_embeddings.dtype
        )
        count = len(allocations)
        embeddings = target_embeddings.unsqueeze(0).expand(count, -1, -1).unsqueeze(0)
        features = features.unsqueeze(0)
        target_mask = torch.ones(features.shape[:-1], dtype=torch.bool, device=features.device)
        allocation_mask = torch.ones((1, count), dtype=torch.bool, device=features.device)
        with torch.inference_mode():
            logits = self.allocation_head(
                state_summary.unsqueeze(0) if state_summary.ndim == 1 else state_summary,
                root_option.unsqueeze(0) if root_option.ndim == 1 else root_option,
                embeddings, features, target_mask, allocation_mask,
            )[0]
            distribution = Categorical(logits=logits.float())
            if greedy:
                index = logits.argmax()
            else:
                index = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(0)
            return PlannedMacro(
                allocations[int(index)], int(index), float(distribution.log_prob(index)),
                float(distribution.entropy()),
            )


__all__ = ["MacroPlanner", "PlannedMacro"]

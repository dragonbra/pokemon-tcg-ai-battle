"""Conditional, permutation-invariant scorer for Phantom Dive allocations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DragapultAllocationHead(nn.Module):
    """Score canonical allocations without adding probability mass to root actions.

    Inputs are already encoded by the one shared State Encoder forward. Target rows
    are canonically ordered only for storage; summation makes scores invariant to a
    simultaneous permutation of target embeddings, visible features and counters.
    """

    VISIBLE_FEATURES = (
        "counters", "remaining_hp", "immediate_ko", "prizes_available",
        "damage_to_ko", "bench_position", "current_damage", "attached_energy",
        "status_bits", "evolution_risk", "hp_headroom", "allocation_fraction",
    )

    def __init__(self, d_model: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or d_model
        self.target = nn.Sequential(
            nn.Linear(d_model + len(self.VISIBLE_FEATURES), hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.structure = nn.Sequential(nn.Linear(6, hidden), nn.GELU())
        self.score = nn.Sequential(
            nn.Linear(d_model * 2 + hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        state_summary: Tensor,
        root_option: Tensor,
        target_embeddings: Tensor,
        visible_features: Tensor,
        target_mask: Tensor,
        allocation_mask: Tensor,
    ) -> Tensor:
        """Return `[batch, allocations]` logits; invalid rows are `-inf`."""
        if target_embeddings.ndim != 4 or visible_features.shape[:-1] != target_embeddings.shape[:-1]:
            raise ValueError("allocation target tensors must have shape [B, A, N, ...]")
        if visible_features.size(-1) != len(self.VISIBLE_FEATURES):
            raise ValueError("visible allocation feature width mismatch")
        mask = target_mask.to(torch.bool)
        rows = self.target(torch.cat((target_embeddings, visible_features), dim=-1))
        rows = rows * mask.unsqueeze(-1)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        pooled_sum = rows.sum(dim=-2)
        counters = visible_features[..., 0]
        positive = counters.gt(0) & mask
        total = counters.sum(dim=-1).clamp_min(1)
        concentration = (counters.square().sum(dim=-1) / total.square()).unsqueeze(-1)
        maximum = counters.max(dim=-1).values.unsqueeze(-1)
        minimum_positive = torch.where(positive, counters, torch.inf).min(dim=-1).values
        minimum_positive = torch.where(torch.isfinite(minimum_positive), minimum_positive, 0).unsqueeze(-1)
        structure = torch.cat((
            count.to(counters.dtype), positive.sum(dim=-1, keepdim=True).to(counters.dtype),
            total.unsqueeze(-1), maximum, minimum_positive, concentration,
        ), dim=-1)
        structural = self.structure(structure)
        batch, allocations = allocation_mask.shape
        summary = state_summary[:, None, :].expand(batch, allocations, -1)
        root = root_option[:, None, :].expand(batch, allocations, -1)
        logits = self.score(torch.cat((summary, root, pooled_sum, structural), dim=-1)).squeeze(-1)
        return logits.masked_fill(~allocation_mask.to(torch.bool), -torch.inf)


__all__ = ["DragapultAllocationHead"]

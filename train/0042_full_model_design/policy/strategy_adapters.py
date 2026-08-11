"""Zero-gated 0042 Value and policy readout adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Fixed root-decision signals; all Value-side tensors are detached here."""

    side_onehot: Tensor
    z_meta: Tensor
    meta_probabilities: Tensor
    value: Tensor
    own_archetype_id: Tensor

    @classmethod
    def build(
        cls,
        *,
        relative_first_player: Tensor,
        z_meta: Tensor,
        meta_logits: Tensor,
        value: Tensor,
        own_archetype_id: Tensor,
    ) -> "StrategyContext":
        if relative_first_player.ndim != 1:
            raise ValueError("relative_first_player must have shape [B]")
        if not bool(relative_first_player.eq(1).logical_or(relative_first_player.eq(2)).all()):
            raise ValueError("relative_first_player must encode own-first=1 or own-second=2")
        batch, width = z_meta.shape
        if width != 320 or meta_logits.shape != (batch, 15) or value.shape != (batch,):
            raise ValueError("0042 strategic Value/Meta tensor shape mismatch")
        if own_archetype_id.shape != (batch,):
            raise ValueError("own_archetype_id must have shape [B]")
        side = F.one_hot(relative_first_player - 1, num_classes=2).to(z_meta.dtype)
        return cls(
            side,
            z_meta.detach(),
            meta_logits.detach().softmax(dim=-1),
            value.detach(),
            own_archetype_id,
        )

    def index_select(self, indices: Tensor) -> "StrategyContext":
        return StrategyContext(*(
            value.index_select(0, indices)
            for value in (
                self.side_onehot,
                self.z_meta,
                self.meta_probabilities,
                self.value,
                self.own_archetype_id,
            )
        ))


def build_defined_strategy_context(
    *,
    relative_first_player: Tensor,
    z_meta: Tensor,
    meta_logits: Tensor,
    value: Tensor,
    own_archetype_id: Tensor,
) -> tuple[Tensor, StrategyContext | None]:
    """Build context only where the Agent's first/second role is established."""

    if relative_first_player.ndim != 1:
        raise ValueError("relative_first_player must have shape [B]")
    if not bool(
        relative_first_player.eq(0)
        .logical_or(relative_first_player.eq(1))
        .logical_or(relative_first_player.eq(2))
        .all()
    ):
        raise ValueError("relative_first_player contains an invalid category")
    rows = relative_first_player.ne(0).nonzero(as_tuple=False).flatten()
    if not rows.numel():
        return rows, None
    return rows, StrategyContext.build(
        relative_first_player=relative_first_player.index_select(0, rows),
        z_meta=z_meta.index_select(0, rows),
        meta_logits=meta_logits.index_select(0, rows),
        value=value.index_select(0, rows),
        own_archetype_id=own_archetype_id.index_select(0, rows),
    )


class ValueResidualAdapter(nn.Module):
    def __init__(self, width: int = 320, embedding_dim: int = 16) -> None:
        super().__init__()
        self.width = width
        self.norm = nn.LayerNorm(width, elementwise_affine=True)
        self.own_embedding = nn.Embedding(15, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(width + embedding_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, z_value: Tensor, own_archetype_id: Tensor) -> tuple[Tensor, Tensor]:
        deck = self.own_embedding(own_archetype_id)
        delta = self.mlp(torch.cat((self.norm(z_value), deck), dim=-1))
        return z_value + self.gate.tanh() * delta, delta

    def effective_residual_ratio(self, z_value: Tensor, delta: Tensor) -> Tensor:
        residual = self.gate.tanh() * delta
        return residual.norm(dim=-1) / z_value.norm(dim=-1).clamp_min(1.0e-8)


class PolicyStrategyAdapter(nn.Module):
    def __init__(self, width: int = 320, embedding_dim: int = 16) -> None:
        super().__init__()
        self.width = width
        self.hidden_norm = nn.LayerNorm(width, elementwise_affine=True)
        self.meta_context_norm = nn.LayerNorm(width, elementwise_affine=True)
        self.own_embedding = nn.Embedding(15, embedding_dim)
        context_dim = 2 + width + 15 + 1 + embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(width + context_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def context_tensor(self, context: StrategyContext) -> Tensor:
        return torch.cat((
            context.side_onehot,
            self.meta_context_norm(context.z_meta),
            context.meta_probabilities,
            context.value.unsqueeze(-1),
            self.own_embedding(context.own_archetype_id),
        ), dim=-1)

    def forward(self, hidden: Tensor, context: StrategyContext) -> tuple[Tensor, Tensor]:
        delta = self.mlp(torch.cat((self.hidden_norm(hidden), self.context_tensor(context)), dim=-1))
        return hidden + self.gate.tanh() * delta, delta

    def effective_residual_ratio(self, hidden: Tensor, delta: Tensor) -> Tensor:
        residual = self.gate.tanh() * delta
        return residual.norm(dim=-1) / hidden.norm(dim=-1).clamp_min(1.0e-8)


__all__ = [
    "PolicyStrategyAdapter",
    "StrategyContext",
    "ValueResidualAdapter",
]

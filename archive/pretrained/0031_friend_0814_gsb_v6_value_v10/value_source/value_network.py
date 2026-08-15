"""Frozen semantic Encoder with MLP and latent-query Value heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from torch import Tensor, nn

from model_source.contracts.batch import DecisionBatch

Architecture = Literal["raw_pool_mlp", "summary_mlp", "latent_queries"]


@dataclass(frozen=True, slots=True)
class ValueOutputs:
    value_logit: Tensor
    archetype_logits: Tensor
    final_diff_logits: Tensor

    @property
    def win_probability(self) -> Tensor:
        return self.value_logit.sigmoid()

    @property
    def value(self) -> Tensor:
        return 2.0 * self.win_probability - 1.0


class PredictionHeads(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.archetype = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 15))
        self.final_diff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 13))

    def forward(self, value: Tensor, archetype: Tensor, final_diff: Tensor) -> ValueOutputs:
        return ValueOutputs(
            self.value(value).squeeze(-1),
            self.archetype(archetype),
            self.final_diff(final_diff),
        )


class LatentDecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.query_cross_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * ffn_multiplier), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * ffn_multiplier, width), nn.Dropout(dropout),
        )

    def forward(self, queries: Tensor, memory: Tensor, memory_mask: Tensor) -> Tensor:
        normalized_queries = self.query_cross_norm(queries)
        normalized_memory = self.memory_norm(memory)
        crossed, _ = self.cross_attention(
            normalized_queries, normalized_memory, normalized_memory,
            key_padding_mask=~memory_mask, need_weights=False,
        )
        queries = queries + crossed
        normalized_queries = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized_queries, normalized_queries, normalized_queries, need_weights=False
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class LatentQueryValueHead(nn.Module):
    def __init__(self, width: int, *, queries: int = 8, layers: int = 2, heads: int = 8,
                 ffn_multiplier: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if queries < 3 or layers < 1:
            raise ValueError("latent Value head requires at least three queries and one layer")
        self.queries = nn.Parameter(torch.empty(1, queries, width))
        nn.init.normal_(self.queries, std=width ** -0.5)
        self.blocks = nn.ModuleList(
            LatentDecoderBlock(width, heads, ffn_multiplier, dropout) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.heads = PredictionHeads(width)

    def forward(self, memory: Tensor, memory_mask: Tensor) -> ValueOutputs:
        if memory.ndim != 3 or memory_mask.shape != memory.shape[:2]:
            raise ValueError("latent Value memory and mask shapes disagree")
        queries = self.queries.expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, memory_mask)
        queries = self.final_norm(queries)
        return self.heads(queries[:, 0], queries[:, 1], queries[:, 2])


class MLPValueHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.heads = PredictionHeads(width)

    def forward(self, representation: Tensor) -> ValueOutputs:
        return self.heads(representation, representation, representation)


def masked_mean(memory: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(memory.dtype)
    return (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class FrozenEncoderValueNetwork(nn.Module):
    def __init__(self, encoder: nn.Module, *, architecture: Architecture = "latent_queries",
                 queries: int = 8, layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.architecture = architecture
        self.value_config = {
            "architecture": architecture,
            "queries": queries,
            "layers": layers,
            "dropout": dropout,
        }
        width = int(encoder.config.d_model)
        if architecture == "latent_queries":
            self.value_head: nn.Module = LatentQueryValueHead(
                width, queries=queries, layers=layers, heads=int(encoder.config.heads), dropout=dropout
            )
        elif architecture in {"raw_pool_mlp", "summary_mlp"}:
            self.value_head = MLPValueHead(width)
        else:
            raise ValueError(f"unsupported 0036 Value architecture: {architecture}")

    def train(self, mode: bool = True) -> "FrozenEncoderValueNetwork":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, batch: DecisionBatch | Mapping[str, Tensor]) -> ValueOutputs:
        with torch.no_grad():
            validated, state, options = self.encoder.encode(batch)
        if self.architecture == "summary_mlp":
            return self.value_head(state.summary)  # type: ignore[arg-type]
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        if self.architecture == "raw_pool_mlp":
            return self.value_head(masked_mean(memory, memory_mask))  # type: ignore[arg-type]
        return self.value_head(memory, memory_mask)  # type: ignore[arg-type]

    def assert_frozen_encoder(self) -> None:
        trainable = [name for name, parameter in self.encoder.named_parameters() if parameter.requires_grad]
        if trainable:
            raise RuntimeError(f"0036 encoder is not frozen: {trainable[:5]}")


__all__ = ["Architecture", "FrozenEncoderValueNetwork", "LatentDecoderBlock",
           "LatentQueryValueHead", "MLPValueHead", "ValueOutputs", "masked_mean"]

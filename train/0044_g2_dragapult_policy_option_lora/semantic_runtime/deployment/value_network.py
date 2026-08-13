"""Portable copy of the 0038 latent-query Value Network architecture."""

from __future__ import annotations

import torch
from torch import nn


class _PredictionHeads(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.archetype = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 15))
        self.final_diff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 13))


class _LatentDecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_multiplier: int = 4) -> None:
        super().__init__()
        self.query_cross_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(width * ffn_multiplier, width),
            nn.Dropout(0.0),
        )

    def forward(self, queries, memory, memory_mask):
        normalized_queries = self.query_cross_norm(queries)
        normalized_memory = self.memory_norm(memory)
        crossed, _ = self.cross_attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        queries = queries + crossed
        normalized_queries = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class LatentQueryValueHead(nn.Module):
    """Exact 8-query/2-layer Value structure used by 0038 checkpoints."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, 8, width))
        nn.init.normal_(self.queries, std=width ** -0.5)
        self.blocks = nn.ModuleList(
            _LatentDecoderBlock(width, heads) for _ in range(2)
        )
        self.final_norm = nn.LayerNorm(width)
        self.heads = _PredictionHeads(width)

    def decode(self, memory, memory_mask):
        queries = self.queries.expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, memory_mask)
        return self.final_norm(queries)

    def value(self, memory, memory_mask):
        query = self.decode(memory, memory_mask)[:, 0]
        return 2.0 * self.heads.value(query).squeeze(-1).sigmoid() - 1.0


__all__ = ["LatentQueryValueHead"]

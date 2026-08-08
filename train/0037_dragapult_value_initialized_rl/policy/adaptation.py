"""Cheap focal-only last-Option-block adaptation for 0037 PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    lora: bool = False
    layernorm_tuning: bool = False
    rank: int = 4
    alpha: float = 8.0
    option_block: int = 1
    adapter_learning_rate: float = 3.0e-5
    layernorm_learning_rate: float = 1.0e-5

    def validate(self, layer_count: int = 2) -> None:
        if self.rank < 1 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.option_block < layer_count:
            raise ValueError("LoRA Option block index is out of range")
        if self.layernorm_tuning and not self.lora:
            raise ValueError("local output LayerNorm tuning requires LoRA")


class LoRAQVWeight(nn.Module):
    """Add low-rank deltas only to merged MultiheadAttention Q and V rows."""

    def __init__(self, width: int, columns: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.q_a = nn.Parameter(torch.empty(rank, columns))
        self.q_b = nn.Parameter(torch.zeros(width, rank))
        self.v_a = nn.Parameter(torch.empty(rank, columns))
        self.v_b = nn.Parameter(torch.zeros(width, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.q_a, a=5**0.5)
        nn.init.kaiming_uniform_(self.v_a, a=5**0.5)

    def forward(self, base: Tensor) -> Tensor:
        width = base.shape[0] // 3
        q = (self.q_b @ self.q_a).to(base.dtype) * self.scale
        v = (self.v_b @ self.v_a).to(base.dtype) * self.scale
        return base + torch.cat((q, torch.zeros_like(q), v), dim=0)


def _add_qv_lora(attention: nn.MultiheadAttention, config: AdaptationConfig) -> None:
    if attention.in_proj_weight is None:
        raise ValueError("Q/V LoRA requires merged in_proj_weight")
    if parametrize.is_parametrized(attention, "in_proj_weight"):
        adapter = attention.parametrizations.in_proj_weight[0]
        for parameter in adapter.parameters():
            parameter.requires_grad_(True)
        attention.parametrizations.in_proj_weight.original.requires_grad_(False)
        return
    weight = attention.in_proj_weight
    if weight.shape[0] != 3 * attention.embed_dim:
        raise ValueError("unexpected merged QKV shape")
    parametrize.register_parametrization(
        attention,
        "in_proj_weight",
        LoRAQVWeight(attention.embed_dim, weight.shape[1], config.rank, config.alpha),
    )
    attention.parametrizations.in_proj_weight.original.requires_grad_(False)


def apply_focal_adaptation(actor: nn.Module, config: AdaptationConfig) -> dict[str, object]:
    decoder = actor.option_encoder.cross_attention_transformer
    config.validate(len(decoder.layers))
    if config.lora:
        block = decoder.layers[config.option_block]
        _add_qv_lora(block.self_attn, config)
        _add_qv_lora(block.multihead_attn, config)
    if config.layernorm_tuning:
        decoder.norm.weight.requires_grad_(True)
        decoder.norm.bias.requires_grad_(True)
    names = dict(actor.named_parameters())
    adapters = [
        name for name, value in names.items()
        if value.requires_grad and ".parametrizations.in_proj_weight.0." in name
    ]
    norms = [
        name for name, value in names.items()
        if value.requires_grad
        and name.startswith("option_encoder.cross_attention_transformer.norm.")
    ]
    return {
        "lora": config.lora,
        "layernorm_tuning": config.layernorm_tuning,
        "rank": config.rank,
        "alpha": config.alpha,
        "option_block": config.option_block,
        "attention_targets": ["self_attn.qv", "cross_attn.qv"] if config.lora else [],
        "adapter_parameter_names": adapters,
        "adapter_parameters": sum(names[name].numel() for name in adapters),
        "layernorm_parameter_names": norms,
        "layernorm_parameters": sum(names[name].numel() for name in norms),
    }


def adapter_parameters(actor: nn.Module) -> list[nn.Parameter]:
    return [
        value for name, value in actor.named_parameters()
        if value.requires_grad and ".parametrizations.in_proj_weight.0." in name
    ]


def tuned_layernorm_parameters(actor: nn.Module) -> list[nn.Parameter]:
    adapter_ids = {id(value) for value in adapter_parameters(actor)}
    decoder_ids = {id(value) for value in actor.action_decoder.parameters()}
    return [
        value for value in actor.parameters()
        if value.requires_grad and id(value) not in adapter_ids and id(value) not in decoder_ids
    ]


__all__ = [
    "AdaptationConfig",
    "LoRAQVWeight",
    "adapter_parameters",
    "apply_focal_adaptation",
    "tuned_layernorm_parameters",
]

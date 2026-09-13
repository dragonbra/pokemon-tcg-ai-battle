"""StateEncoder LoRA parametrizations required by 0045 V13+ artifacts."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


class MergedQVLoRA(nn.Module):
    def __init__(self, width: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.scale = float(alpha) / int(rank)
        self.q_a = nn.Parameter(torch.empty(rank, width))
        self.q_b = nn.Parameter(torch.zeros(width, rank))
        self.v_a = nn.Parameter(torch.empty(rank, width))
        self.v_b = nn.Parameter(torch.zeros(width, rank))

    def forward(self, base: Tensor) -> Tensor:
        width = base.shape[1]
        q = (self.q_b @ self.q_a).to(base.dtype) * self.scale
        v = (self.v_b @ self.v_a).to(base.dtype) * self.scale
        return base + torch.cat((q, torch.zeros_like(q), v), dim=0)


class LinearLoRA(nn.Module):
    def __init__(self, output: int, input_: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.scale = float(alpha) / int(rank)
        self.a = nn.Parameter(torch.empty(rank, input_))
        self.b = nn.Parameter(torch.zeros(output, rank))

    def forward(self, base: Tensor) -> Tensor:
        return base + (self.b @ self.a).to(base.dtype) * self.scale


def install_shared_encoder_lora(
    actor: nn.Module, *, rank: int = 16, alpha: float = 16.0,
) -> None:
    if (int(rank), float(alpha)) != (16, 16.0):
        raise ValueError("unsupported packaged StateEncoder LoRA contract")
    state = actor.state_encoder
    for encoder in (state.board_encoder, state.event_encoder):
        layer = encoder.layers[-1]
        parametrize.register_parametrization(
            layer.self_attn, "in_proj_weight",
            MergedQVLoRA(int(actor.config.d_model), rank, alpha),
        )
        parametrize.register_parametrization(
            layer.self_attn.out_proj, "weight",
            LinearLoRA(
                layer.self_attn.out_proj.out_features,
                layer.self_attn.out_proj.in_features, rank, alpha,
            ),
        )
    for index in (0, 2):
        linear = state.family_fusion[index]
        parametrize.register_parametrization(
            linear, "weight",
            LinearLoRA(linear.out_features, linear.in_features, rank, alpha),
        )


def install_state_ffn_lora(
    actor: nn.Module, *, rank: int = 16, alpha: float = 16.0,
) -> None:
    if (int(rank), float(alpha)) != (16, 16.0):
        raise ValueError("unsupported packaged StateEncoder FFN LoRA contract")
    layer = actor.state_encoder.board_encoder.layers[-1]
    for field in ("linear1", "linear2"):
        linear = getattr(layer, field)
        parametrize.register_parametrization(
            linear, "weight", LinearLoRA(linear.out_features, linear.in_features, rank, alpha)
        )


__all__ = ["LinearLoRA", "MergedQVLoRA", "install_shared_encoder_lora", "install_state_ffn_lora"]

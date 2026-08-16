"""Rank-16 LoRA parametrizations for the shared 0045 V13 StateEncoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


class MergedQVLoRA(nn.Module):
    def __init__(self, width: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.q_a = nn.Parameter(torch.empty(rank, width))
        self.q_b = nn.Parameter(torch.zeros(width, rank))
        self.v_a = nn.Parameter(torch.empty(rank, width))
        self.v_b = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.q_a, a=5**0.5)
        nn.init.kaiming_uniform_(self.v_a, a=5**0.5)

    def forward(self, base: Tensor) -> Tensor:
        width = base.shape[1]
        if base.shape != (3 * width, width):
            raise ValueError("merged Q/V LoRA requires [3d,d] in_proj_weight")
        q = (self.q_b @ self.q_a).to(base.dtype) * self.scale
        v = (self.v_b @ self.v_a).to(base.dtype) * self.scale
        return base + torch.cat((q, torch.zeros_like(q), v), dim=0)


class LinearLoRA(nn.Module):
    def __init__(self, output: int, input_: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.a = nn.Parameter(torch.empty(rank, input_))
        self.b = nn.Parameter(torch.zeros(output, rank))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    def forward(self, base: Tensor) -> Tensor:
        if base.shape != (self.b.shape[0], self.a.shape[1]):
            raise ValueError("linear LoRA base shape changed")
        return base + (self.b @ self.a).to(base.dtype) * self.scale


@dataclass(frozen=True, slots=True)
class SharedEncoderLoRAInventory:
    rank: int
    alpha: float
    parameter_count: int
    tensor_count: int
    targets: tuple[str, ...]


def install_shared_encoder_lora(
    actor: nn.Module, *, rank: int = 16, alpha: float = 16.0,
) -> SharedEncoderLoRAInventory:
    if rank != 16 or float(alpha) != 16.0:
        raise ValueError("0045 V13 shared StateEncoder LoRA requires r=16 alpha=16")
    state = actor.state_encoder
    if state.architecture != "hierarchical":
        raise ValueError("0045 V13 requires hierarchical StateEncoder")
    targets: list[str] = []
    for family_name, encoder in (
        ("board", state.board_encoder),
        ("event", state.event_encoder),
    ):
        layer = encoder.layers[-1]
        parametrize.register_parametrization(
            layer.self_attn,
            "in_proj_weight",
            MergedQVLoRA(int(actor.config.d_model), rank, alpha),
        )
        parametrize.register_parametrization(
            layer.self_attn.out_proj,
            "weight",
            LinearLoRA(
                layer.self_attn.out_proj.out_features,
                layer.self_attn.out_proj.in_features,
                rank,
                alpha,
            ),
        )
        targets.extend((
            f"state_encoder.{family_name}_encoder.layers.-1.self_attn.qv",
            f"state_encoder.{family_name}_encoder.layers.-1.self_attn.out_proj",
        ))
    for index in (0, 2):
        linear = state.family_fusion[index]
        if not isinstance(linear, nn.Linear):
            raise RuntimeError("StateEncoder family-fusion Linear inventory changed")
        parametrize.register_parametrization(
            linear,
            "weight",
            LinearLoRA(linear.out_features, linear.in_features, rank, alpha),
        )
        targets.append(f"state_encoder.family_fusion.{index}")
    parameters = tuple(shared_encoder_lora_parameters(actor))
    inventory = SharedEncoderLoRAInventory(
        rank=rank,
        alpha=float(alpha),
        parameter_count=sum(value.numel() for value in parameters),
        tensor_count=len(parameters),
        targets=tuple(targets),
    )
    if inventory.parameter_count != 92_160 or inventory.tensor_count != 16:
        raise RuntimeError(f"0045 V13 StateEncoder LoRA inventory changed: {inventory}")
    return inventory


def shared_encoder_lora_parameters(actor: nn.Module):
    for name, parameter in actor.state_encoder.named_parameters():
        if ".parametrizations." in name and not name.endswith(".original"):
            yield parameter


def shared_encoder_lora_named_parameters(actor: nn.Module):
    for name, parameter in actor.state_encoder.named_parameters():
        if ".parametrizations." in name and not name.endswith(".original"):
            yield name, parameter


__all__ = [
    "LinearLoRA", "MergedQVLoRA", "SharedEncoderLoRAInventory",
    "install_shared_encoder_lora", "shared_encoder_lora_named_parameters",
    "shared_encoder_lora_parameters",
]

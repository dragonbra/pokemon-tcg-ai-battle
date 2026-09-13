"""Policy-only Q/V LoRA over the immutable final Option Transformer block."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call


class LoRAQVDelta(nn.Module):
    """Low-rank deltas for the Q and V rows of one merged QKV projection."""

    def __init__(self, width: int, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.q_a = nn.Parameter(torch.empty(rank, width))
        self.q_b = nn.Parameter(torch.zeros(width, rank))
        self.v_a = nn.Parameter(torch.empty(rank, width))
        self.v_b = nn.Parameter(torch.zeros(width, rank))
        self.scale = float(alpha) / int(rank)
        nn.init.kaiming_uniform_(self.q_a, a=5**0.5)
        nn.init.kaiming_uniform_(self.v_a, a=5**0.5)

    def merged_weight(self, base: Tensor) -> Tensor:
        width = base.shape[0] // 3
        if base.shape != (3 * width, width):
            raise ValueError("Option Q/V LoRA requires a square merged QKV projection")
        q = (self.q_b @ self.q_a).to(base.dtype) * self.scale
        v = (self.v_b @ self.v_a).to(base.dtype) * self.scale
        return base + torch.cat((q, torch.zeros_like(q), v), dim=0)


@dataclass(frozen=True, slots=True)
class DualOptionEncoding:
    value_options: Tensor
    policy_options: Tensor


class PolicyOnlyOptionLoRA(nn.Module):
    """Run the same frozen block with policy-only Q/V deltas, without copying it."""

    def __init__(
        self, width: int, *, rank: int = 4, alpha: float = 8.0,
        output_projection: bool = False, ffn_expansion: bool = False,
    ) -> None:
        super().__init__()
        if (rank, float(alpha), output_projection) not in {
            (4, 8.0, False), (16, 16.0, True),
        }:
            raise ValueError("unsupported 0045 Option LoRA contract")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.option_block = 1
        self.output_projection = bool(output_projection)
        self.ffn_expansion = bool(ffn_expansion)
        self.self_attention = LoRAQVDelta(width, rank, alpha)
        self.cross_attention = LoRAQVDelta(width, rank, alpha)
        if self.output_projection:
            from .shared_encoder_lora import LinearLoRA
            self.self_output = LinearLoRA(width, width, rank, alpha)
            self.cross_output = LinearLoRA(width, width, rank, alpha)
        if self.ffn_expansion:
            from .shared_encoder_lora import LinearLoRA
            self.ffn_linear1 = LinearLoRA(3 * width, width, rank, alpha)
            self.ffn_linear2 = LinearLoRA(width, 3 * width, rank, alpha)

    def forward(self, option_encoder, batch, state, prefix: Tensor) -> Tensor:
        transformer = option_encoder.cross_attention_transformer
        if len(transformer.layers) != 2:
            raise RuntimeError("0044 requires exactly two Option Transformer blocks")
        final = transformer.layers[1]
        overrides = {
            "self_attn.in_proj_weight": self.self_attention.merged_weight(
                final.self_attn.in_proj_weight
            ),
            "multihead_attn.in_proj_weight": self.cross_attention.merged_weight(
                final.multihead_attn.in_proj_weight
            ),
        }
        if self.output_projection:
            overrides.update({
                "self_attn.out_proj.weight": self.self_output(
                    final.self_attn.out_proj.weight
                ),
                "multihead_attn.out_proj.weight": self.cross_output(
                    final.multihead_attn.out_proj.weight
                ),
            })
        if self.ffn_expansion:
            overrides.update({
                "linear1.weight": self.ffn_linear1(final.linear1.weight),
                "linear2.weight": self.ffn_linear2(final.linear2.weight),
            })
        encoded = functional_call(
            final,
            overrides,
            (prefix, state.tokens),
            option_encoder._masks(batch, state),
            strict=False,
        )
        if transformer.norm is not None:
            encoded = transformer.norm(encoded)
        return encoded * batch.option_mask.unsqueeze(-1)

    def is_zero_delta(self) -> bool:
        """True only while every zero-initialized B matrix remains exactly zero."""
        return all(
            bool(torch.count_nonzero(parameter.detach()) == 0)
            for name, parameter in self.named_parameters()
            if name.endswith("_b")
        )

    def assert_inventory(self) -> None:
        parameters = tuple(self.named_parameters())
        expected_count = (12 if self.output_projection else 8) + (4 if self.ffn_expansion else 0)
        expected_parameters = (61_440 if self.output_projection else 10_240) + (40_960 if self.ffn_expansion else 0)
        if len(parameters) != expected_count or sum(value.numel() for _, value in parameters) != expected_parameters:
            raise RuntimeError("0045 policy Option LoRA inventory changed")
        expected = {
            "self_attention.q_a", "self_attention.q_b",
            "self_attention.v_a", "self_attention.v_b",
            "cross_attention.q_a", "cross_attention.q_b",
            "cross_attention.v_a", "cross_attention.v_b",
        }
        if self.output_projection:
            expected |= {
                "self_output.a", "self_output.b",
                "cross_output.a", "cross_output.b",
            }
        if self.ffn_expansion:
            expected |= {
                "ffn_linear1.a", "ffn_linear1.b",
                "ffn_linear2.a", "ffn_linear2.b",
            }
        if {name for name, _ in parameters} != expected:
            raise RuntimeError("0044 policy Option LoRA tensor names changed")

    def attention_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("ffn_"):
                yield parameter

    def ffn_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("ffn_"):
                yield parameter


__all__ = ["DualOptionEncoding", "LoRAQVDelta", "PolicyOnlyOptionLoRA"]

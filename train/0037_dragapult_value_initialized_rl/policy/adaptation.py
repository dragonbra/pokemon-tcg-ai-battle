"""Focal-only low-rank Encoder adaptation for 0037 PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    lora: bool = False
    layernorm_tuning: bool = False
    rank: int = 8
    alpha: float = 16.0
    board_layers: tuple[int, ...] = (0, 1, 3)
    adapter_learning_rate: float = 3.0e-5
    layernorm_learning_rate: float = 1.0e-5

    def validate(self, layer_count: int = 4) -> None:
        if self.rank < 1 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not self.board_layers or min(self.board_layers) < 0 or max(self.board_layers) >= layer_count:
            raise ValueError("LoRA board layer index is out of range")
        if self.layernorm_tuning and not self.lora:
            raise ValueError("LayerNorm arm requires LoRA")


class LoRAWeight(nn.Module):
    def __init__(self, rows: int, columns: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, columns))
        self.b = nn.Parameter(torch.zeros(rows, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    def forward(self, base: Tensor) -> Tensor:
        return base + (self.b @ self.a).to(base.dtype) * self.scale


def _add_lora(module: nn.Module, parameter: str, config: AdaptationConfig) -> None:
    if parametrize.is_parametrized(module, parameter):
        low_rank = module.parametrizations[parameter][0]
        low_rank.a.requires_grad_(True)
        low_rank.b.requires_grad_(True)
        module.parametrizations[parameter].original.requires_grad_(False)
        return
    weight = getattr(module, parameter)
    if weight.ndim != 2:
        raise ValueError(f"LoRA target {parameter} is not a matrix")
    parametrize.register_parametrization(
        module,
        parameter,
        LoRAWeight(weight.shape[0], weight.shape[1], config.rank, config.alpha),
    )
    module.parametrizations[parameter].original.requires_grad_(False)


def apply_focal_adaptation(actor: nn.Module, config: AdaptationConfig) -> dict[str, object]:
    config.validate(len(actor.state_encoder.board_encoder.layers))
    if config.lora:
        for index in config.board_layers:
            layer = actor.state_encoder.board_encoder.layers[index]
            _add_lora(layer.self_attn, "in_proj_weight", config)
            _add_lora(layer.self_attn.out_proj, "weight", config)
            _add_lora(layer.linear1, "weight", config)
            _add_lora(layer.linear2, "weight", config)
    if config.layernorm_tuning:
        for root in (actor.state_encoder, actor.option_encoder):
            for name, module in root.named_modules():
                if (
                    isinstance(module, nn.LayerNorm)
                    and name != "prototypes"
                    and not name.startswith("prototypes.")
                ):
                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)
    names = [name for name, value in actor.named_parameters() if value.requires_grad]
    adapter = [name for name in names if ".parametrizations." in name and name.endswith((".a", ".b"))]
    norms = [name for name in names if "_encoder" in name and name.endswith((".weight", ".bias")) and ".norm" in name]
    return {
        "lora": config.lora,
        "layernorm_tuning": config.layernorm_tuning,
        "rank": config.rank,
        "alpha": config.alpha,
        "board_layers": list(config.board_layers),
        "adapter_parameter_names": adapter,
        "adapter_parameters": sum(dict(actor.named_parameters())[name].numel() for name in adapter),
        "layernorm_parameter_names": norms,
        "layernorm_parameters": sum(dict(actor.named_parameters())[name].numel() for name in norms),
    }


def adapter_parameters(actor: nn.Module) -> list[nn.Parameter]:
    return [
        value for name, value in actor.named_parameters()
        if value.requires_grad and ".parametrizations." in name and name.endswith((".a", ".b"))
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
    "LoRAWeight",
    "adapter_parameters",
    "apply_focal_adaptation",
    "tuned_layernorm_parameters",
]

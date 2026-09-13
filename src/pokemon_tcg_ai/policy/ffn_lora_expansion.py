"""Zero-effective r16 FFN LoRA for the final shared State block."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.nn.utils import parametrize

from .shared_encoder_lora import LinearLoRA


@dataclass(frozen=True, slots=True)
class StateFFNLoRAInventory:
    rank: int
    alpha: float
    parameter_count: int
    tensor_count: int
    targets: tuple[str, ...]


def install_state_ffn_lora(
    actor: nn.Module, *, rank: int = 16, alpha: float = 16.0,
) -> StateFFNLoRAInventory:
    if rank != 16 or float(alpha) != 16.0:
        raise ValueError("0045 State FFN expansion requires r=16 alpha=16")
    state = actor.state_encoder
    if state.architecture != "hierarchical":
        raise ValueError("0045 State FFN expansion requires hierarchical StateEncoder")
    layer = state.board_encoder.layers[-1]
    targets = []
    for field in ("linear1", "linear2"):
        linear = getattr(layer, field)
        if not isinstance(linear, nn.Linear):
            raise RuntimeError(f"State final block {field} is not Linear")
        parametrize.register_parametrization(
            linear, "weight",
            LinearLoRA(linear.out_features, linear.in_features, rank, alpha),
        )
        targets.append(f"state_encoder.board_encoder.layers.-1.{field}")
    parameters = tuple(state_ffn_lora_parameters(actor))
    inventory = StateFFNLoRAInventory(
        rank=rank, alpha=float(alpha),
        parameter_count=sum(value.numel() for value in parameters),
        tensor_count=len(parameters), targets=tuple(targets),
    )
    if inventory.parameter_count != 40_960 or inventory.tensor_count != 4:
        raise RuntimeError(f"0045 State FFN LoRA inventory changed: {inventory}")
    return inventory


def state_ffn_lora_named_parameters(actor: nn.Module):
    prefixes = (
        "board_encoder.layers.3.linear1.parametrizations.weight.0.",
        "board_encoder.layers.3.linear2.parametrizations.weight.0.",
    )
    for name, parameter in actor.state_encoder.named_parameters():
        if name.startswith(prefixes) and not name.endswith(".original"):
            yield name, parameter


def state_ffn_lora_parameters(actor: nn.Module):
    for _, parameter in state_ffn_lora_named_parameters(actor):
        yield parameter


__all__ = [
    "StateFFNLoRAInventory", "install_state_ffn_lora",
    "state_ffn_lora_named_parameters", "state_ffn_lora_parameters",
]

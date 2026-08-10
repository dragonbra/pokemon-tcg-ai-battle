"""Fail-closed compatibility boundary for removed 0040 Option adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    """0042 installs no Option LoRA or Option LayerNorm tuning."""

    lora: bool = False
    layernorm_tuning: bool = False

    def validate(self, layer_count: int = 2) -> None:
        del layer_count
        if self.lora or self.layernorm_tuning:
            raise ValueError("0042 forbids Option LoRA and Option LayerNorm tuning")


def apply_focal_adaptation(actor: nn.Module, config: AdaptationConfig) -> dict[str, object]:
    """Retain the old call boundary while making parameter injection impossible."""

    del actor
    config.validate()
    return {
        "lora": False,
        "layernorm_tuning": False,
        "attention_targets": [],
        "adapter_parameter_names": [],
        "adapter_parameters": 0,
        "layernorm_parameter_names": [],
        "layernorm_parameters": 0,
    }


__all__ = ["AdaptationConfig", "apply_focal_adaptation"]

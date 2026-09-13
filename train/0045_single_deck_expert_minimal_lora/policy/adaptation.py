"""0044 policy-only Last Option Q/V LoRA contract."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .option_policy_lora import PolicyOnlyOptionLoRA


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    """Install only the audited policy-side final Option Q/V adapter."""

    lora: bool = True
    layernorm_tuning: bool = False
    rank: int = 4
    alpha: float = 8.0
    option_block: int = 1
    output_projection: bool = False
    shared_state_encoder: bool = False
    option_ffn_lora: bool = False
    state_ffn_lora: bool = False

    def validate(self, layer_count: int = 2) -> None:
        if not self.lora:
            raise ValueError("0045 requires policy-only LoRA")
        if layer_count != 2 or self.option_block != 1:
            raise ValueError("0044 LoRA must target the final block of the two-block Option encoder")
        signature = (
            self.rank, float(self.alpha), self.output_projection,
            self.shared_state_encoder, self.option_ffn_lora,
            self.state_ffn_lora, self.layernorm_tuning,
        )
        allowed = signature in {
            (4, 8.0, False, False, False, False, False),
            (16, 16.0, True, True, False, False, False),
            (16, 16.0, True, True, True, True, True),
        }
        if not allowed:
            raise ValueError("unsupported 0045 adaptation profile")


def apply_focal_adaptation(actor: nn.Module, config: AdaptationConfig) -> PolicyOnlyOptionLoRA:
    config.validate(len(actor.option_encoder.cross_attention_transformer.layers))
    adapter = PolicyOnlyOptionLoRA(
        int(actor.config.d_model), rank=config.rank, alpha=config.alpha,
        output_projection=config.output_projection,
        ffn_expansion=config.option_ffn_lora,
    )
    adapter.layernorm_tuning = config.layernorm_tuning
    adapter.assert_inventory()
    return adapter


def adaptation_inventory(adapter: PolicyOnlyOptionLoRA) -> dict[str, object]:
    adapter.assert_inventory()
    return {
        "lora": True,
        "policy_only": True,
        "layernorm_tuning": bool(getattr(adapter, "layernorm_tuning", False)),
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "option_block": adapter.option_block,
        "attention_targets": (
            ["self_attn.qvo", "cross_attn.qvo"]
            if adapter.output_projection
            else ["self_attn.qv", "cross_attn.qv"]
        ),
        "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "adapter_parameters": sum(value.numel() for value in adapter.parameters()),
        "option_ffn_lora": adapter.ffn_expansion,
        "value_branch": "immutable_lora_free_final_option_block",
        "policy_branch": "same_final_option_block_plus_qv_lora",
    }


__all__ = ["AdaptationConfig", "adaptation_inventory", "apply_focal_adaptation"]

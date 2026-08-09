"""Strict paired-0809 initialization with an allocation-only BC sidecar."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .policy.actor_critic import SemanticActorCritic, SourceIdentity, load_actor_critic
from .policy.adaptation import AdaptationConfig
from .source import (
    ACTOR_CHECKPOINT, ACTOR_SHA256, PROJECT_ID, VALUE_CHECKPOINT, VALUE_SHA256,
)
from .integrated.config import IntegratedFlags


INITIALIZATION_SCHEMA = "0040_paired_0809_update0_initialization_v1"
UPDATE0_RNG_SEED = 400_040_001
COMMON_UPDATE0_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "rl_runs/0038_action_boundary_rl/versions/V3_update0_chance_boundary_fallback/checkpoint/update-000000.pt"
)
COMMON_UPDATE0_SHA256 = "d572c8673fcf16c39fb17d8d261bec6578885a69c227e579bc4e6f616bb22820"
NEW_MODULE_MISSING_PREFIX_ALLOWLIST = (
    "allocation_head.", "prize_aux.", "prize_query_head.",
    "opponent_meta_head.", "opponent_meta_conditioner.", "tempo_aux_head.",
)


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    zero_shot_policy_checkpoint: Path = ACTOR_CHECKPOINT
    zero_shot_value_checkpoint: Path = VALUE_CHECKPOINT
    enable_last_option_qv_lora: bool = True
    enable_local_output_layernorm: bool = False
    lora_rank: int = 4
    lora_alpha: float = 8.0
    option_block: int = 1
    rng_seed: int = UPDATE0_RNG_SEED

    def adaptation(self) -> AdaptationConfig:
        return AdaptationConfig(
            lora=self.enable_last_option_qv_lora,
            layernorm_tuning=self.enable_local_output_layernorm,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            option_block=self.option_block,
        )


def assert_no_rl_checkpoint_source(path: Path, expected_sha256: str) -> None:
    resolved = path.resolve()
    if "versions" in resolved.parts or any(part.startswith("update-") for part in resolved.parts):
        raise ValueError(f"RL-updated checkpoint is forbidden for 0040 initialization: {path}")
    from .policy.actor_critic import _sha256
    digest = _sha256(resolved)
    if digest != expected_sha256:
        raise ValueError(f"zero-shot source hash mismatch for {path}: {digest}")


def assert_only_new_missing_keys(missing_keys: list[str] | tuple[str, ...]) -> None:
    invalid = sorted(
        key for key in missing_keys
        if not key.startswith(NEW_MODULE_MISSING_PREFIX_ALLOWLIST)
    )
    if invalid:
        raise ValueError(f"unexpected missing 0037-compatible core keys: {invalid[:8]}")


def assert_zero_delta_lora(model: SemanticActorCritic) -> None:
    violations = []
    for name, value in model.actor.named_parameters():
        if name.endswith((".q_b", ".v_b")) and torch.count_nonzero(value).item() != 0:
            violations.append(name)
    if violations:
        raise ValueError(f"fresh Last Option LoRA is not zero-delta: {violations}")


def build_update0_model(
    deck: tuple[int, ...],
    *,
    device: str | torch.device = "cpu",
    config: InitializationConfig = InitializationConfig(),
    integrated_flags: IntegratedFlags = IntegratedFlags(),
) -> tuple[SemanticActorCritic, SourceIdentity]:
    assert_no_rl_checkpoint_source(config.zero_shot_policy_checkpoint, ACTOR_SHA256)
    assert_no_rl_checkpoint_source(config.zero_shot_value_checkpoint, VALUE_SHA256)
    if (
        config.lora_rank != 4 or config.lora_alpha != 8.0 or config.option_block != 1
        or not config.enable_last_option_qv_lora
    ):
        raise ValueError("0040 update-0 requires last-block Q/V LoRA r4/a8")
    torch.manual_seed(config.rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.rng_seed)
    model, identity = load_actor_critic(
        config.zero_shot_policy_checkpoint,
        deck,
        device,
        value_checkpoint=config.zero_shot_value_checkpoint,
        adaptation=config.adaptation(),
        integrated_flags=integrated_flags,
    )
    assert_zero_delta_lora(model)
    return model, identity


def initialization_manifest(config: InitializationConfig = InitializationConfig()) -> dict[str, Any]:
    return {
        "schema_version": INITIALIZATION_SCHEMA,
        "project_id": PROJECT_ID,
        "zero_shot_policy_checkpoint": str(config.zero_shot_policy_checkpoint),
        "zero_shot_value_checkpoint": str(config.zero_shot_value_checkpoint),
        "checkpoint_hashes": {
            "policy_sha256": ACTOR_SHA256,
            "value_sha256": VALUE_SHA256,
        },
        "base_encoder_source": "Policy-0809 / 0031 V5 GSB epoch-20; exact weights frozen",
        "option_encoder_source": "Policy-0809 / 0031 V5 GSB epoch-20",
        "action_decoder_source": "Policy-0809 / 0031 V5 GSB epoch-20; fully trainable in RL",
        "value_trunk_source": "paired 0809 0036 V9 latent-query Value epoch-9; pre-RL model-only",
        "value_semantics": "query_0 -> 2*sigmoid(value_logit)-1 terminal win/loss",
        "lora_structure": {
            "target": "Option TransformerDecoder block 1 self_attn+multihead_attn merged Q/V",
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "trainable_parameters": 10_240,
        },
        "lora_initialization": "fresh A=kaiming, B=zeros; exact zero-delta; no prior RL tensors",
        "layernorm_trainable_scope": (
            "option_encoder.cross_attention_transformer.norm.{weight,bias}"
            if config.enable_local_output_layernorm else "disabled"
        ),
        "optimizer_initialization": "fresh AdamW; no optimizer state loaded",
        "scheduler_initialization": "fresh/no scheduler in current 0037-compatible PPO interface",
        "rng_initialization": {
            "seed": config.rng_seed,
            "python_numpy_torch_cuda": "fresh per-run streams; no RNG state loaded",
        },
        "rollout_initialization": "empty on-policy buffer; fresh old-policy snapshot at update-0",
        "forbidden_initialization_sources": [
            "0038 RL-updated checkpoints", "any actor/value checkpoint/update-*.pt",
            "old PPO rollout/logprob",
        ],
        "new_module_missing_key_allowlist": list(NEW_MODULE_MISSING_PREFIX_ALLOWLIST),
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in asdict(config).items()},
    }


def build_preset_from_common_update0(
    deck: tuple[int, ...], integrated_flags: IntegratedFlags, *,
    device: str | torch.device = "cpu",
) -> tuple[SemanticActorCritic, SourceIdentity]:
    """Build paired-0809 core and overlay only the pre-PPO allocation BC head."""
    from .policy.actor_critic import _sha256
    if _sha256(COMMON_UPDATE0_CHECKPOINT) != COMMON_UPDATE0_SHA256:
        raise ValueError("0038 allocation BC source checkpoint hash mismatch")
    model, identity = build_update0_model(
        deck, device=device, integrated_flags=integrated_flags,
        config=InitializationConfig(
            enable_last_option_qv_lora=integrated_flags.enable_last_option_qv_lora,
            enable_local_output_layernorm=integrated_flags.enable_local_output_layernorm,
        ),
    )
    payload = torch.load(COMMON_UPDATE0_CHECKPOINT, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") or {}
    if payload.get("update") != 0 or metadata.get("ppo_updates") != 0:
        raise ValueError("common update-0 must not contain PPO-updated weights")
    current = model.state_dict()
    source_state = payload.get("state_dict") or {}
    state = {
        name: value for name, value in source_state.items()
        if name.startswith("allocation_head.")
    }
    expected = {name for name in current if name.startswith("allocation_head.")}
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(
            "allocation BC tensor inventory mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(current[name].device, current[name].dtype))
    assert_zero_delta_lora(model)
    return model, identity


__all__ = [
    "COMMON_UPDATE0_CHECKPOINT", "COMMON_UPDATE0_SHA256", "INITIALIZATION_SCHEMA",
    "InitializationConfig", "NEW_MODULE_MISSING_PREFIX_ALLOWLIST",
    "UPDATE0_RNG_SEED", "assert_no_rl_checkpoint_source", "assert_only_new_missing_keys",
    "assert_zero_delta_lora", "build_preset_from_common_update0", "build_update0_model",
    "initialization_manifest",
]

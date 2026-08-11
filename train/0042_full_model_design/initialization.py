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


INITIALIZATION_SCHEMA = "0042_strategy_conditioned_update0_initialization_v1"
ALLOCATION_HEAD_SCHEMA = "0042_allocation_head_bc_sidecar_v1"
UPDATE0_RNG_SEED = 420_042_001
ALLOCATION_HEAD_CHECKPOINT = (
    Path(__file__).resolve().parent / "assets/allocation_head_bc_v1.pt"
)
ALLOCATION_HEAD_SHA256 = "5d77f0ee9e47c15d9e896649daed71a00bd73e68c36c6eabbd30f3243d30e3c6"
ALLOCATION_HEAD_SOURCE_SHA256 = (
    "d572c8673fcf16c39fb17d8d261bec6578885a69c227e579bc4e6f616bb22820"
)
NEW_MODULE_MISSING_PREFIX_ALLOWLIST = (
    "allocation_head.", "prize_aux.", "prize_query_head.",
    "value_adapter.", "policy_strategy_adapter.", "tempo_aux_head.",
)


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    zero_shot_policy_checkpoint: Path = ACTOR_CHECKPOINT
    zero_shot_value_checkpoint: Path = VALUE_CHECKPOINT
    rng_seed: int = UPDATE0_RNG_SEED

    def adaptation(self) -> AdaptationConfig:
        return AdaptationConfig()


def assert_no_rl_checkpoint_source(path: Path, expected_sha256: str) -> None:
    resolved = path.resolve()
    if "versions" in resolved.parts or any(part.startswith("update-") for part in resolved.parts):
        raise ValueError(f"RL-updated checkpoint is forbidden for 0042 initialization: {path}")
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


def assert_zero_gate_base(model: SemanticActorCritic) -> None:
    model.assert_no_option_adaptation()
    gates = {"g_V": model.value_adapter.gate, "g_pi": model.policy_strategy_adapter.gate}
    violations = [name for name, value in gates.items() if torch.count_nonzero(value).item()]
    if violations:
        raise ValueError(f"0042 Base requires exact zero adapter gates: {violations}")


def build_update0_model(
    deck: tuple[int, ...],
    *,
    device: str | torch.device = "cpu",
    config: InitializationConfig = InitializationConfig(),
    integrated_flags: IntegratedFlags = IntegratedFlags(),
) -> tuple[SemanticActorCritic, SourceIdentity]:
    assert_no_rl_checkpoint_source(config.zero_shot_policy_checkpoint, ACTOR_SHA256)
    assert_no_rl_checkpoint_source(config.zero_shot_value_checkpoint, VALUE_SHA256)
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
    assert_zero_gate_base(model)
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
            "allocation_head_sha256": ALLOCATION_HEAD_SHA256,
            "allocation_head_source_sha256": ALLOCATION_HEAD_SOURCE_SHA256,
        },
        "base_encoder_source": "Policy-0809 / 0031 V5 GSB epoch-20; exact weights frozen",
        "option_encoder_source": "Policy-0809 / 0031 V5 GSB epoch-20",
        "action_decoder_source": "Policy-0809 / 0031 V5 GSB epoch-20; fully trainable in RL",
        "value_trunk_source": "paired 0809 0036 V9 latent-query Value epoch-9; pre-RL model-only",
        "value_semantics": "query_0 -> 2*sigmoid(value_logit)-1 terminal win/loss",
        "option_adaptation": "absent; OptionEncoder frozen and unparametrized",
        "adapter_initialization": "g_V=g_pi=0; residual MLPs normally initialized",
        "optimizer_initialization": "fresh AdamW; no optimizer state loaded",
        "scheduler_initialization": "fresh/no scheduler in current 0037-compatible PPO interface",
        "rng_initialization": {
            "seed": config.rng_seed,
            "python_numpy_torch_cuda": "fresh per-run streams; no RNG state loaded",
        },
        "rollout_initialization": "empty on-policy buffer; fresh old-policy snapshot at update-0",
        "forbidden_initialization_sources": [
            "0040 RL-updated checkpoints", "any actor/value checkpoint/update-*.pt",
            "any checkpoint containing Option LoRA keys",
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
    if _sha256(ALLOCATION_HEAD_CHECKPOINT) != ALLOCATION_HEAD_SHA256:
        raise ValueError("0042 allocation-head sidecar hash mismatch")
    model, identity = build_update0_model(
        deck, device=device, integrated_flags=integrated_flags,
        config=InitializationConfig(),
    )
    payload = torch.load(ALLOCATION_HEAD_CHECKPOINT, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "allocation_head_state_dict", "metadata"}:
        raise ValueError("allocation-head sidecar top-level inventory mismatch")
    if payload.get("schema_version") != ALLOCATION_HEAD_SCHEMA:
        raise ValueError("allocation-head sidecar schema mismatch")
    metadata = payload.get("metadata") or {}
    if (
        metadata.get("ppo_updates") != 0
        or metadata.get("source_checkpoint_sha256") != ALLOCATION_HEAD_SOURCE_SHA256
        or metadata.get("tensor_prefix_removed") != "allocation_head."
    ):
        raise ValueError("allocation-head sidecar provenance mismatch")
    state = payload.get("allocation_head_state_dict") or {}
    expected = model.allocation_head.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            "allocation BC tensor inventory mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    for name, value in state.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(
                f"allocation BC tensor contract mismatch for {name}: "
                f"got shape={tuple(value.shape)} dtype={value.dtype}; "
                f"expected shape={tuple(reference.shape)} dtype={reference.dtype}"
            )
    model.allocation_head.load_state_dict(state, strict=True, assign=False)
    assert_zero_gate_base(model)
    return model, identity


__all__ = [
    "ALLOCATION_HEAD_CHECKPOINT", "ALLOCATION_HEAD_SCHEMA", "ALLOCATION_HEAD_SHA256",
    "ALLOCATION_HEAD_SOURCE_SHA256", "INITIALIZATION_SCHEMA",
    "InitializationConfig", "NEW_MODULE_MISSING_PREFIX_ALLOWLIST",
    "UPDATE0_RNG_SEED", "assert_no_rl_checkpoint_source", "assert_only_new_missing_keys",
    "assert_zero_gate_base", "build_preset_from_common_update0", "build_update0_model",
    "initialization_manifest",
]

"""Regression guard for the approved PPO Protocol V2 configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CONFIG_SHA256 = "fe6e1862a9f8660b3b19505898dacf55a2955a07385958639a5cb1f80d94264c"
EXPECTED_TRAINABLE = (
    "actor.action_decoder.*", "policy_strategy_adapter.*", "value_head.queries",
    "value_head.blocks.*", "value_head.final_norm.*", "value_head.heads.value.*",
    "value_adapter.*", "allocation_head.*",
    "prize_aux.* when FULL_MODEL prize auxiliary is enabled",
)


@dataclass(frozen=True, slots=True)
class TrainingRegressionAudit:
    config_sha256: str
    games_per_update: int
    ppo_epochs: int
    ppo_minibatch_size: int
    trainable_contract: tuple[str, ...]
    allowed_v2_changes: tuple[str, ...]
    status: str = "PASS"
    schema_version: str = "0043_training_regression_audit_v1"

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def audit_training_config(path: Path) -> TrainingRegressionAudit:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"approved active training config SHA mismatch: {digest}")
    config = json.loads(raw)
    expected = {
        "games_per_update": 256,
        "rollout_batch_size": 256,
        "trajectory_games_per_update": 256,
        "ppo_epochs": 3,
        "ppo_minibatch_size": 2048,
    }
    mismatches = {key: config.get(key) for key, value in expected.items() if config.get(key) != value}
    ppo_expected = {
        "decoder_learning_rate": 2e-5,
        "policy_adapter_learning_rate": 4e-5,
        "allocation_learning_rate": 2e-5,
        "value_learning_rate": 1e-4,
        "prize_learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 2048,
        "gamma": 1,
        "gae_lambda": 0.95,
        "clip_ratio": 0.1,
        "reference_kl_coefficient": 0.02,
        "target_behavior_kl": 0.015,
        "hard_behavior_kl_guard": 0.025,
    }
    mismatches.update({
        f"ppo.{key}": config.get("ppo", {}).get(key)
        for key, value in ppo_expected.items() if config.get("ppo", {}).get(key) != value
    })
    trainable = tuple(config.get("trainable_contract", ()))
    if trainable != EXPECTED_TRAINABLE:
        mismatches["trainable_contract"] = trainable
    if mismatches:
        raise ValueError(f"0043 PPO regression mismatch: {mismatches}")
    return TrainingRegressionAudit(
        config_sha256=digest,
        games_per_update=256,
        ppo_epochs=3,
        ppo_minibatch_size=2048,
        trainable_contract=trainable,
        allowed_v2_changes=(
            "opponent_asset_routing", "opponent_sampler", "pfsp_curriculum",
            "rollout_telemetry", "promotion_governance",
        ),
    )


__all__ = ["TrainingRegressionAudit", "audit_training_config"]

"""Frozen project identity and model-view contracts for 0014."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import PROJECT_ID

REPOSITORY_ROOT = Path(__file__).parents[2]
RAW_DATASET = (
    REPOSITORY_ROOT
    / "rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1"
)
MODEL_READY_DATASET = (
    REPOSITORY_ROOT
    / "rl_runs/0014_faithful_board_causal_features/dataset/V1_full_feature_superset"
)


@dataclass(frozen=True, slots=True)
class FeatureSwitches:
    """Reader-visible families; materialization always stores the full superset."""

    card_capability: bool = False
    public_zone_inventory: bool = False
    own_resource_ledger: bool = False
    event_memory: bool = False
    registered_deck: bool = False
    relation_graph: bool = False
    goal_qkv: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    project_id: str = PROJECT_ID
    raw_dataset_content_sha256: str = (
        "146ac0799b0a23a5089615361e76c33722c08df45b276cfd2b387171c70dc0c7"
    )
    raw_record_schema: str = "decision_record_v3"
    source_first_date: str = "2026-07-18"
    source_last_date: str = "2026-07-25"
    train_decisions: int = 145_961
    validation_decisions: int = 16_167
    expert_name: str = "Yushin Ito"
    feature_schema_version: str = "faithful_board_causal_input_v1"
    model_ready_schema_version: str = "faithful_board_causal_cache_v1"


PROJECT_CONFIG = ExperimentConfig()
A0_SWITCHES = FeatureSwitches()
AC_SWITCHES = FeatureSwitches(
    card_capability=True,
    public_zone_inventory=True,
    own_resource_ledger=True,
    event_memory=True,
    registered_deck=True,
    relation_graph=True,
    goal_qkv=True,
)

__all__ = [
    "A0_SWITCHES",
    "AC_SWITCHES",
    "ExperimentConfig",
    "FeatureSwitches",
    "MODEL_READY_DATASET",
    "PROJECT_CONFIG",
    "RAW_DATASET",
]

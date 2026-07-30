"""Frozen project identity and universal R15 model-view contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import PROJECT_ID


REPOSITORY_ROOT = Path(__file__).parents[2]
MODEL_READY_DATASET = (
    REPOSITORY_ROOT / f"rl_runs/{PROJECT_ID}/dataset/V1_corrected_model_ready"
)
SOURCE_MODEL_PROJECT = "0016_alakazam_multideck_bc"
SOURCE_MODEL_VERSION = "V2_win_multideck_r15_gradual_option"


@dataclass(frozen=True, slots=True)
class FeatureSwitches:
    card_capability: bool = False
    public_zone_inventory: bool = False
    own_resource_ledger: bool = False
    event_memory: bool = False
    registered_deck: bool = False
    relation_graph: bool = False
    goal_qkv: bool = False


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
    "AC_SWITCHES",
    "FeatureSwitches",
    "MODEL_READY_DATASET",
    "SOURCE_MODEL_PROJECT",
    "SOURCE_MODEL_VERSION",
]

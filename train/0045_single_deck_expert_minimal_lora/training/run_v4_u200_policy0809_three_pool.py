"""0045 V4: deck-007 U200 continuation with targeted Meta rollout/evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..league.meta_balanced import balanced_meta_deck_schedule
from ..own_archetype import OwnArchetypeVocabulary
from .lr_profiles import EXPERT_COLD_START_LR_PROFILE
from .run_v1 import PROJECT_ROOT, ROOT, run


VERSION = "V4_dragapult_007_u200_policy0809_three_pool"
VERSION_ROOT = ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions" / VERSION
START_UPDATE = 200
PARENT_VERSION = "V3_dragapult_007_expert_continue_u45"
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions"
    / PARENT_VERSION / "checkpoint/update-000200.pt"
)
PARENT_CHECKPOINT_SHA256 = "d8a2cd2e8b2f696fee452951156be01cdf08200ee0bd502ca205d98d2d0a7306"
REFERENCE_CHECKPOINT = (
    ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions"
    / "V1_minimal_lora_dragapult_007/checkpoint/update-000000.pt"
)
REFERENCE_CHECKPOINT_SHA256 = "3c13e085b4587a8112148b393c4b80c4c875efbc13820dcf44acb6a0a1991bc7"
FOCAL_DECK_ID = "007"
ROLLOUT_GAMES = 512
ROLLOUT_OPPONENT_POLICY_IDS = ("Champion-G2",)
OPPONENT_SAMPLING_MODE = "meta_balanced_training_pool"
OPPONENT_META_WEIGHTS = {
    0: 3.0,
    1: 3.0,
    2: 3.0,
    3: 6.0,
    4: 3.0,
    5: 6.0,
    27: 1.0,
}
PERIODIC_EVALUATION_PROFILE = "policy0809_three_pool_cuda512"
WANDB_RUN_ID = "0045-v4-dragapult-007-u200-policy0809-three-pool"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V4 formal version path is already used: {version_root}")
    for path, expected_hash in (
        (PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256),
        (REFERENCE_CHECKPOINT, REFERENCE_CHECKPOINT_SHA256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"checkpoint identity changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    reference = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0045_minimal_lora_model_only_v1"
        or parent.get("update") != START_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V4 parent is not exact V3 U200")
    if (
        reference.get("schema_version") != "0045_minimal_lora_model_only_v1"
        or reference.get("update") != 0
    ):
        raise RuntimeError("V4 reference is not Frozen-0045-Init U0")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    training_ids = {
        row.deck_id for row in registry.decks if "training" in row.roles
    }
    mappings = tuple(row for row in vocabulary.mappings if row.deck_id in training_ids)
    schedule = balanced_meta_deck_schedule(
        quota_seed=430_044_200,
        shuffle_seed=430_045_200,
        mappings=mappings,
        lanes=ROLLOUT_GAMES,
        meta_weights=OPPONENT_META_WEIGHTS,
    )
    quotas = Counter(row.archetype_id for row in schedule)
    return {
        "schema_version": "0045_v4_u200_policy0809_three_pool_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "update": START_UPDATE,
            "checkpoint": str(PARENT_CHECKPOINT),
            "sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer": "fresh",
            "on_policy_data": "fresh",
        },
        "reference": {
            "policy_id": "Frozen-0045-Init",
            "update": 0,
            "checkpoint": str(REFERENCE_CHECKPOINT),
            "sha256": REFERENCE_CHECKPOINT_SHA256,
        },
        "rollout": {
            "games_per_update": ROLLOUT_GAMES,
            "opponent_policy_ids": list(ROLLOUT_OPPONENT_POLICY_IDS),
            "sampling_mode": OPPONENT_SAMPLING_MODE,
            "meta_weights": OPPONENT_META_WEIGHTS,
            "first_schedule_meta_quotas": dict(sorted(quotas.items())),
        },
        "periodic_evaluation": {
            "profile": PERIODIC_EVALUATION_PROFILE,
            "interval_updates": 5,
            "opponent_policy_id": "Policy-0809",
            "pools": {
                "low_score": [2, 3, 5],
                "priority": [0, 1, 4, 27],
                "remaining": "all other active Meta classes",
            },
            "games_per_pool": 512,
            "total_games_per_checkpoint": 1536,
            "candidate_materializations_per_checkpoint": 1,
        },
        "optimizer_profile": EXPERT_COLD_START_LR_PROFILE.metadata(),
        "update_limit": None,
        "checkpoint_retention": "all",
        "wandb": {
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    }


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    readiness()
    run(
        updates=updates,
        wandb_mode=wandb_mode,
        launch_formal=True,
        version=VERSION,
        start_update=START_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT,
        reference_checkpoint=REFERENCE_CHECKPOINT,
        baseline_evaluation_checkpoint=START_UPDATE,
        periodic_evaluation_enabled=True,
        periodic_evaluation_profile=PERIODIC_EVALUATION_PROFILE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0045 · V4 · deck 007 U200 · Policy-0809 three-pool eval",
        focal_deck_ids=(FOCAL_DECK_ID,),
        focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=START_UPDATE,
        reference_anchor_update=0,
        reference_anchor_identity="Frozen-0045-Init",
        opponent_sampling_mode=OPPONENT_SAMPLING_MODE,
        opponent_policy_ids=ROLLOUT_OPPONENT_POLICY_IDS,
        latest_champion_policy_id="Champion-G2",
        rollout_games=ROLLOUT_GAMES,
        opponent_meta_weights=OPPONENT_META_WEIGHTS,
        learning_rate_profile=EXPERT_COLD_START_LR_PROFILE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--readiness-output", type=Path)
    args = parser.parse_args()
    if args.launch_formal:
        launch(wandb_mode=args.wandb_mode, updates=args.updates)
        return 0
    result = readiness()
    if args.readiness_output:
        args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
        args.readiness_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V22: continue durable V21 U1 with 512 games and six 3x Meta classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..assets import sha256_file
from .run_v1 import ROOT, run
from .run_v19 import (
    FOCAL_DECK_ID,
    G3_PORTABLE,
    LEARNING_RATE_PROFILE,
    OPPONENT_POLICY_ID,
    OPPONENT_SCHEDULE,
    PPO_MINIBATCH_SIZE,
)
from .run_v21 import (
    BEHAVIOR_PROBE_BATCH_SIZE,
    FORWARD_MICROBATCH_SIZE,
    OFFLOAD_REFERENCE_AFTER_CACHE,
    VERSION as FAILED_VERSION,
    readiness as v21_readiness,
)


VERSION = "V22_v21_u1_rollout512_meta6_x3"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
PARENT_VERSION = FAILED_VERSION
SOURCE_PARENT_UPDATE = 1
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
    / "checkpoint/update-000001.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "140b02c3a89679b5b0feef556752c99a65d3bef0550d810cd1d9424cc03bb028"
)
ROLLOUT_GAMES = 512
WEIGHTED_META_IDS = (0, 1, 2, 3, 5, 27)
OPPONENT_META_WEIGHTS = {meta_id: 3.0 for meta_id in WEIGHTED_META_IDS}
WANDB_RUN_ID = "0044-v22-v21-u1-rollout512-meta6-x3"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v21_readiness(version_root=version_root)
    if not PARENT_CHECKPOINT.is_file():
        raise FileNotFoundError("V22 exact V21 U1 parent checkpoint is missing")
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V22 V21-U1 parent checkpoint identity changed")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V22 parent is not exact durable V21 U1")

    result["ppo"].update({
        "rollout_games": ROLLOUT_GAMES,
        "batch_size": PPO_MINIBATCH_SIZE,
        "forward_microbatch_size": FORWARD_MICROBATCH_SIZE,
        "behavior_probe_batch_size": BEHAVIOR_PROBE_BATCH_SIZE,
        "offload_reference_after_cache": OFFLOAD_REFERENCE_AFTER_CACHE,
        "learning_rate_profile": LEARNING_RATE_PROFILE.metadata(),
    })
    result.update({
        "schema_version": "0044_v22_v21_u1_rollout512_meta6_x3_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "retry_of": FAILED_VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "source_update": SOURCE_PARENT_UPDATE,
            "local_start_update": 0,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer": "fresh",
            "pfsp_state": None,
        },
        "continuation": {
            "v21_status": "failed_after_durable_u1",
            "discarded_partial_update": 2,
            "on_policy_data": "fresh",
            "semantic_changes": {
                "rollout_games": [1024, ROLLOUT_GAMES],
                "opponent_meta_weights": {
                    "before": {0: 2.0},
                    "after": OPPONENT_META_WEIGHTS,
                },
            },
            "unchanged": {
                "opponent_policy_id": OPPONENT_POLICY_ID,
                "learning_rate_profile": LEARNING_RATE_PROFILE.profile_id,
                "ppo_epochs": 3,
                "logical_minibatch": PPO_MINIBATCH_SIZE,
            },
        },
        "rollout_contract": {
            "games_per_update": ROLLOUT_GAMES,
            "opponent_policy_id": OPPONENT_POLICY_ID,
            "opponent_sampling": OPPONENT_SCHEDULE,
            "opponent_meta_weights": OPPONENT_META_WEIGHTS,
            "weighted_internal_meta_ids": list(WEIGHTED_META_IDS),
            "expected_weighted_meta_lanes_each": 38,
            "expected_other_meta_lanes_each": [12, 13],
            "exact_deck_pool": [f"{deck_id:03d}" for deck_id in range(1, 70)],
            "pfsp_enabled": False,
            "evaluation_enabled": False,
        },
        "wandb": {
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    })
    return result


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    readiness()
    run(
        updates=updates,
        wandb_mode=wandb_mode,
        launch_formal=True,
        version=VERSION,
        start_update=0,
        parent_checkpoint=PARENT_CHECKPOINT,
        parent_pfsp_state=None,
        baseline_evaluation_checkpoint=None,
        periodic_evaluation_enabled=False,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V22 · V21 U1 → rollout 512 · Meta 00/01/02/03/05/27 3x",
        focal_deck_ids=(FOCAL_DECK_ID,),
        focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        reference_anchor_update=SOURCE_PARENT_UPDATE,
        reference_anchor_identity=(
            f"{PARENT_VERSION}@update-{SOURCE_PARENT_UPDATE:06d}"
        ),
        opponent_sampling_mode=OPPONENT_SCHEDULE,
        opponent_policy_ids=(OPPONENT_POLICY_ID,),
        latest_champion_policy_id=OPPONENT_POLICY_ID,
        rollout_games=ROLLOUT_GAMES,
        opponent_meta_weights=OPPONENT_META_WEIGHTS,
        ppo_batch_size=PPO_MINIBATCH_SIZE,
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
        behavior_probe_batch_size=BEHAVIOR_PROBE_BATCH_SIZE,
        offload_reference_after_cache=OFFLOAD_REFERENCE_AFTER_CACHE,
        learning_rate_profile=LEARNING_RATE_PROFILE,
        focal_base_checkpoint=G3_PORTABLE,
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


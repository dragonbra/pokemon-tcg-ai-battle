"""V23: final entropy-only fine-tune from durable V22 U9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..assets import sha256_file
from .run_v1 import ROOT, run
from .run_v22 import (
    BEHAVIOR_PROBE_BATCH_SIZE,
    FOCAL_DECK_ID,
    FORWARD_MICROBATCH_SIZE,
    G3_PORTABLE,
    LEARNING_RATE_PROFILE,
    OFFLOAD_REFERENCE_AFTER_CACHE,
    OPPONENT_META_WEIGHTS,
    OPPONENT_POLICY_ID,
    OPPONENT_SCHEDULE,
    PPO_MINIBATCH_SIZE,
    ROLLOUT_GAMES,
    VERSION as PARENT_VERSION,
    readiness as v22_readiness,
)


VERSION = "V23_v22_u9_final_entropy_0015"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
SOURCE_PARENT_UPDATE = 9
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
    / "checkpoint/update-000009.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "5dad7923c636ca33f4f19dafd6de250d964ccb46f72da2d0161a765c1a319d98"
)
ENTROPY_COEFFICIENT_BEFORE = 0.003
ENTROPY_COEFFICIENT = 0.0015
WANDB_RUN_ID = "0044-v23-v22-u9-final-entropy-0015"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v22_readiness(version_root=version_root)
    if not PARENT_CHECKPOINT.is_file():
        raise FileNotFoundError("V23 exact V22 U9 parent checkpoint is missing")
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V23 V22-U9 parent checkpoint identity changed")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V23 parent is not exact durable V22 U9")

    result["ppo"].update({
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "rollout_games": ROLLOUT_GAMES,
        "batch_size": PPO_MINIBATCH_SIZE,
        "forward_microbatch_size": FORWARD_MICROBATCH_SIZE,
        "behavior_probe_batch_size": BEHAVIOR_PROBE_BATCH_SIZE,
        "offload_reference_after_cache": OFFLOAD_REFERENCE_AFTER_CACHE,
        "learning_rate_profile": LEARNING_RATE_PROFILE.metadata(),
    })
    result.update({
        "schema_version": "0044_v23_v22_u9_final_entropy_0015_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "source_update": SOURCE_PARENT_UPDATE,
            "local_start_update": 0,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer": "fresh",
            "pfsp_state": None,
        },
        "final_finetune": {
            "on_policy_data": "fresh",
            "entropy_coefficient_before": ENTROPY_COEFFICIENT_BEFORE,
            "entropy_coefficient_after": ENTROPY_COEFFICIENT,
            "single_semantic_change": "entropy_bonus_coefficient",
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
        wandb_name="0044 · V23 · V22 U9 → final entropy 0.0015",
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
        entropy_coefficient=ENTROPY_COEFFICIENT,
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

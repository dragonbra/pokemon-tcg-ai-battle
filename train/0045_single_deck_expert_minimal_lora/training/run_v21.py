"""V21: execution-only memory fix after two near-capacity CUDA resets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_v1 import ROOT, run
from .run_v19 import (
    FOCAL_DECK_ID,
    G3_PORTABLE,
    LEARNING_RATE_PROFILE,
    OPPONENT_META_WEIGHTS,
    OPPONENT_POLICY_ID,
    OPPONENT_SCHEDULE,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    PARENT_VERSION,
    PPO_MINIBATCH_SIZE,
    ROLLOUT_GAMES,
    SOURCE_PARENT_UPDATE,
)
from .run_v20 import VERSION as FAILED_VERSION, readiness as v20_readiness


VERSION = "V21_v20_retry_reference_offload_microbatch256"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
FORWARD_MICROBATCH_SIZE = 256
BEHAVIOR_PROBE_BATCH_SIZE = 256
OFFLOAD_REFERENCE_AFTER_CACHE = True
WANDB_RUN_ID = "0044-v21-v20-retry-reference-offload-microbatch256"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v20_readiness(version_root=version_root)
    result["ppo"].update({
        "forward_microbatch_size": FORWARD_MICROBATCH_SIZE,
        "behavior_probe_batch_size": BEHAVIOR_PROBE_BATCH_SIZE,
        "offload_reference_after_cache": OFFLOAD_REFERENCE_AFTER_CACHE,
    })
    result.update({
        "schema_version": "0044_v21_reference_offload_memory_fix_readiness_v1",
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
        "cuda_recovery": {
            "repeated_failure": "CUDA driver error: device not ready",
            "failed_versions": [
                "V19_v18_u14_deck069_half_standard_lr_1024_rollout",
                FAILED_VERSION,
            ],
            "last_durable_update": 0,
            "discarded_partial_update": 1,
            "root_cause": (
                "near-capacity physical execution: microbatch reduction alone left "
                "the frozen reference model resident and probe/PPO peaks at ~15.8 GiB"
            ),
            "physical_changes_only": {
                "forward_microbatch": [512, FORWARD_MICROBATCH_SIZE],
                "behavior_probe_batch": [512, BEHAVIOR_PROBE_BATCH_SIZE],
                "reference_policy": (
                    "compute full cached log-probs on CUDA once, then move immutable "
                    "reference weights to CPU before PPO backward"
                ),
            },
            "semantic_change": False,
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
        updates=updates, wandb_mode=wandb_mode, launch_formal=True,
        version=VERSION, start_update=0,
        parent_checkpoint=PARENT_CHECKPOINT, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=None, periodic_evaluation_enabled=False,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V21 · reference offload · physical microbatch 256",
        focal_deck_ids=(FOCAL_DECK_ID,), focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed", evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        reference_anchor_update=SOURCE_PARENT_UPDATE,
        reference_anchor_identity=f"{PARENT_VERSION}@update-{SOURCE_PARENT_UPDATE:06d}",
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

"""V20: memory-safe retry of V19 with physical PPO microbatch 512."""

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
    VERSION as FAILED_VERSION,
    readiness as v19_readiness,
)


VERSION = "V20_v19_retry_microbatch512"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
FORWARD_MICROBATCH_SIZE = 512
WANDB_RUN_ID = "0044-v20-v19-retry-microbatch512"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v19_readiness(version_root=version_root)
    result["ppo"]["forward_microbatch_size"] = FORWARD_MICROBATCH_SIZE
    result.update({
        "schema_version": "0044_v20_v19_memory_safe_retry_readiness_v1",
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
            "v19_failure": "CUDA driver error: device not ready",
            "v19_last_durable_update": 0,
            "discarded_partial_update": 1,
            "allocator_oom": False,
            "post_failure_cuda_smoke": "PASS",
            "logical_minibatch": PPO_MINIBATCH_SIZE,
            "old_physical_microbatch": 768,
            "new_physical_microbatch": FORWARD_MICROBATCH_SIZE,
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
        wandb_name="0044 · V20 · V19 retry · physical microbatch 512",
        focal_deck_ids=(FOCAL_DECK_ID,),
        focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        evaluation_focal_deck_id=FOCAL_DECK_ID,
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
        learning_rate_profile=LEARNING_RATE_PROFILE,
        focal_base_checkpoint=G3_PORTABLE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"), default="online"
    )
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


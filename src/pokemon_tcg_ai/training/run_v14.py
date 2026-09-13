"""V14/G4: Meta-balanced 67-deck long run with all evaluate disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_v1 import ROOT, run
from .run_v13 import readiness as v13_readiness
from .run_v12 import (
    ACTOR_LEARNING_RATE_SCALE,
    FOCAL_DECK_IDS,
    FOCAL_SCHEDULE,
    G3_PORTABLE,
    INITIALIZATION_DECK_ID,
    META_RESIDUAL_LEARNING_RATE,
    OPPONENT_POLICY_ID,
    OPPONENT_SCHEDULE,
    PARENT_CHECKPOINT,
    PARENT_VERSION,
    PPO_MINIBATCH_SIZE,
    ROLLOUT_GAMES,
    SOURCE_PARENT_UPDATE,
    VALUE_LEARNING_RATE,
)


VERSION = "V14_g4_g3_meta_balanced_512_rollout_only"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
WANDB_RUN_ID = "0044-v14-g4-g3-meta-balanced-512-rollout-only"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v13_readiness(version_root=version_root)
    result.pop("failure_boundary", None)
    result.update({
        "schema_version": "0044_v14_g3_to_g4_rollout_only_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "retry_of": "V13_g4_g3_meta_balanced_512_meta_residual_context41_fix",
        "evaluation": {
            "enabled": False,
            "baseline_u0": False,
            "periodic": False,
            "reason": "67-deck group intelligence is monitored through rollout telemetry",
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
        baseline_evaluation_checkpoint=None,
        periodic_evaluation_enabled=False,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V14 G4 · G3 → Meta-balanced 67-deck · rollout only",
        focal_deck_ids=FOCAL_DECK_IDS,
        focal_deck_id=INITIALIZATION_DECK_ID,
        focal_schedule_mode=FOCAL_SCHEDULE,
        evaluation_focal_deck_id=INITIALIZATION_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        opponent_sampling_mode=OPPONENT_SCHEDULE,
        opponent_policy_ids=(OPPONENT_POLICY_ID,),
        latest_champion_policy_id=OPPONENT_POLICY_ID,
        rollout_games=ROLLOUT_GAMES,
        ppo_batch_size=PPO_MINIBATCH_SIZE,
        forward_microbatch_size=1024,
        actor_learning_rate_scale=ACTOR_LEARNING_RATE_SCALE,
        value_learning_rate=VALUE_LEARNING_RATE,
        prize_learning_rate=VALUE_LEARNING_RATE,
        meta_actor_residual_learning_rate=META_RESIDUAL_LEARNING_RATE,
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

"""V18: clean V17 retry after admitting the live-pool mode to telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_v1 import ROOT, run
from .run_v17 import (
    ACTOR_LEARNING_RATE_SCALE,
    FOCAL_DECK_ID,
    FORWARD_MICROBATCH_SIZE,
    G3_PORTABLE,
    META_RESIDUAL_LEARNING_RATE,
    OPPONENT_POLICY_ID,
    OPPONENT_SCHEDULE,
    PARENT_CHECKPOINT,
    PARENT_VERSION,
    PPO_MINIBATCH_SIZE,
    REFERENCE_ANCHOR_UPDATE,
    ROLLOUT_GAMES,
    SOURCE_PARENT_UPDATE,
    VALUE_LEARNING_RATE,
    VERSION as FAILED_VERSION,
    readiness as v17_readiness,
)


VERSION = "V18_g4_u57_deck069_telemetry_fix"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
WANDB_RUN_ID = "0044-v18-g4-u57-deck069-telemetry-fix"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v17_readiness(version_root=version_root)
    result.update({
        "schema_version": "0044_v18_deck069_telemetry_fix_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "retry_of": FAILED_VERSION,
        "telemetry_fix": {
            "root_cause": "new sampling mode omitted from telemetry closed enum",
            "accepted_mode": OPPONENT_SCHEDULE,
            "v17_u1_policy_evidence": "checkpoint_only_not_canonical_metrics",
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
        wandb_name="0044 · V18 · V15 U57 → deck 069 · telemetry fix",
        focal_deck_ids=(FOCAL_DECK_ID,),
        focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        reference_anchor_update=REFERENCE_ANCHOR_UPDATE,
        reference_anchor_identity=f"{PARENT_VERSION}@update-{SOURCE_PARENT_UPDATE:06d}",
        opponent_sampling_mode=OPPONENT_SCHEDULE,
        opponent_policy_ids=(OPPONENT_POLICY_ID,),
        latest_champion_policy_id=OPPONENT_POLICY_ID,
        rollout_games=ROLLOUT_GAMES,
        ppo_batch_size=PPO_MINIBATCH_SIZE,
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
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

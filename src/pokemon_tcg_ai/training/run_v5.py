"""V5 pristine semantic retry of V4 after its pre-benchmark loader failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import run_v4 as contract
from .run_v1 import run


VERSION = "V5_g2_uniform_0042_v1_lr_benchmark_v2"
VERSION_ROOT = contract.ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
WANDB_RUN_ID = "0044-v5-g2-uniform-0042-v1-lr-benchmark-v2"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = dict(contract.readiness(version_root=version_root))
    result.update({
        "schema_version": "0044_v5_launch_readiness_v1",
        "version": VERSION,
        "retry_of": {
            "version": contract.VERSION,
            "boundary": "failed_before_u0_benchmark_games_and_before_first_rollout",
            "root_cause": "benchmark_v2_policy0809_module_inventory_adapter_missing",
            "training_updates_completed": 0,
        },
        "wandb": {
            "entity": "dragon_bra", "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--readiness-output", type=Path)
    args = parser.parse_args()
    result = readiness()
    if not args.launch_formal:
        if args.readiness_output:
            args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
            args.readiness_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    run(
        updates=args.updates, wandb_mode=args.wandb_mode, launch_formal=True,
        version=VERSION, start_update=contract.START_UPDATE,
        parent_checkpoint=None, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=contract.START_UPDATE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V5 G2 uniform · 0042 V1 LR · Benchmark V2",
        focal_deck_ids=contract.FOCAL_DECK_IDS,
        opponent_sampling_mode=contract.OPPONENT_SAMPLING_MODE,
        actor_learning_rate_scale=contract.ACTOR_LEARNING_RATE_SCALE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

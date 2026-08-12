"""V2 continuation after the U20 mixed-focal cohort optimization."""

from __future__ import annotations

import argparse

from .run_v1 import ROOT, run


VERSION = "V2_mixed_focal_cohorts"
START_UPDATE = 21
PARENT_ROOT = ROOT / "rl_runs/0043_champion_league_rl/versions/V1_focal_002_007"
PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint/update-000021.pt"
PARENT_PFSP_STATE = PARENT_ROOT / "artifact/pfsp_state.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument(
        "--updates", type=int, default=None,
        help="absolute checkpoint update to stop at; omitted means continuous training",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    run(
        updates=args.updates,
        wandb_mode=args.wandb_mode,
        launch_formal=args.launch_formal,
        version=VERSION,
        start_update=START_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT,
        parent_pfsp_state=PARENT_PFSP_STATE,
        wandb_run_id="0043-v2-mixed-focal-cohorts",
        wandb_name="0043 · V2 mixed focal cohorts",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

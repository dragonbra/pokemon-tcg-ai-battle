"""0045 V3: continue deck-007 cold-start training from V2 U45."""

from __future__ import annotations

import argparse
from pathlib import Path

from .lr_profiles import EXPERT_COLD_START_LR_PROFILE
from .run_v1 import run


VERSION = "V3_dragapult_007_expert_continue_u45"
WANDB_RUN_ID = "0045-v3-dragapult-007-expert-continue-u45"
START_UPDATE = 45


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--u0-reference-checkpoint", required=True, type=Path)
    args = parser.parse_args()
    run(
        updates=args.updates,
        wandb_mode=args.wandb_mode,
        launch_formal=args.launch_formal,
        version=VERSION,
        start_update=START_UPDATE,
        parent_checkpoint=args.parent_checkpoint.resolve(),
        reference_checkpoint=args.u0_reference_checkpoint.resolve(),
        baseline_evaluation_checkpoint=START_UPDATE,
        source_parent_version="V2_dragapult_007_expert_cold_start_lr",
        source_parent_update=START_UPDATE,
        reference_anchor_update=0,
        reference_anchor_identity="Frozen-0045-Init",
        learning_rate_profile=EXPERT_COLD_START_LR_PROFILE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0045 · V3 · deck 007 expert continue from V2 U45",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

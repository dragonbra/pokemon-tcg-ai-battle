"""0045 V2: deck-007 expert cold-start with the project-default 2x Actor LR."""

from __future__ import annotations

import argparse
from pathlib import Path

from .lr_profiles import EXPERT_COLD_START_LR_PROFILE
from .run_v1 import run


VERSION = "V2_dragapult_007_expert_cold_start_lr"
WANDB_RUN_ID = "0045-v2-dragapult-007-expert-cold-start-lr"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--u0-checkpoint", required=True, type=Path)
    args = parser.parse_args()
    version_root = (
        Path(__file__).resolve().parents[3]
        / "rl_runs/0047_meta_routed_moe_rl/versions"
        / VERSION
    )
    existing_metrics = version_root / "artifact/training_metrics.jsonl"
    u0_eval_only_restart = existing_metrics.is_file() and bool(
        existing_metrics.read_text(encoding="utf-8").strip()
    )
    run(
        updates=args.updates,
        wandb_mode=args.wandb_mode,
        launch_formal=args.launch_formal,
        version=VERSION,
        parent_checkpoint=args.u0_checkpoint.resolve(),
        baseline_evaluation_checkpoint=None if u0_eval_only_restart else 0,
        allow_pristine_restart=u0_eval_only_restart,
        source_parent_version="Frozen-0045-Init",
        source_parent_update=0,
        reference_anchor_update=0,
        reference_anchor_identity="Frozen-0045-Init",
        learning_rate_profile=EXPERT_COLD_START_LR_PROFILE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0045 · V2 · deck 007 expert cold-start LR",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

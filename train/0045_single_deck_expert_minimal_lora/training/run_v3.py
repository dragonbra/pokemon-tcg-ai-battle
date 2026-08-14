"""0044 V3: continue U1 with singleton-G2 telemetry health contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from .run_v1 import ROOT, PROJECT, run


VERSION = "V3_singleton_g2_telemetry_fix"
WANDB_RUN_ID = "0044-v3-singleton-g2-telemetry-fix"
PARENT_ROOT = ROOT / "rl_runs" / PROJECT / "versions/V2_cached_complete_g2_reference"
PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint/update-000001.pt"
PARENT_PFSP_STATE = PARENT_ROOT / "artifact/pfsp_state.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    if not PARENT_CHECKPOINT.is_file() or not PARENT_PFSP_STATE.is_file():
        raise FileNotFoundError("0044 V3 immutable U1/PFSP parent is unavailable")
    run(
        updates=args.updates, wandb_mode=args.wandb_mode,
        launch_formal=args.launch_formal, version=VERSION, start_update=1,
        parent_checkpoint=PARENT_CHECKPOINT,
        parent_pfsp_state=PARENT_PFSP_STATE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V3 G2 → 007 · singleton-G2 telemetry fix",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

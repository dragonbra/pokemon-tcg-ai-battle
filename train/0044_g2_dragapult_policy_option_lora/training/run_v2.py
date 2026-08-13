"""0044 V2: cache complete-G2 reference log-probs once per rollout batch."""

from __future__ import annotations

import argparse

from .run_v1 import run


VERSION = "V2_cached_complete_g2_reference"
WANDB_RUN_ID = "0044-v2-cached-complete-g2-reference"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    run(
        updates=args.updates, wandb_mode=args.wandb_mode,
        launch_formal=args.launch_formal, version=VERSION,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V2 G2 → 007 · cached complete-G2 reference",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI for 0026 smoke and versioned PPO training."""

from __future__ import annotations

import argparse
import json

from .smoke import run_smoke
from .training.run import RunConfig, run, run_smoke_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--games", type=int, default=4)
    smoke.add_argument("--workers", type=int, default=2)
    smoke.add_argument("--device", default="cuda:0")
    smoke_update = commands.add_parser("smoke-update")
    smoke_update.add_argument("--version", required=True)
    smoke_update.add_argument("--games", type=int, default=4)
    smoke_update.add_argument("--workers", type=int, default=2)
    smoke_update.add_argument("--device", default="cuda:0")
    train = commands.add_parser("train")
    train.add_argument("--version", required=True)
    train.add_argument("--updates", type=int, default=20)
    train.add_argument("--games-per-update", type=int, default=512)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=20260802)
    train.add_argument("--eval-every", type=int, default=5)
    train.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="online")
    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(games=args.games, workers=args.workers, device=args.device)
    elif args.command == "smoke-update":
        result = run_smoke_gate(
            version=args.version, games=args.games,
            workers=args.workers, device_name=args.device,
        )
    else:
        result = run(RunConfig(
            version=args.version, updates=args.updates,
            games_per_update=args.games_per_update, workers=args.workers,
            device=args.device, seed=args.seed, eval_every=args.eval_every,
            wandb_mode=args.wandb_mode,
        ))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["main"]

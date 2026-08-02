"""Bind a formal training child and watchdog to one foreground terminal."""

from __future__ import annotations

import argparse
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from . import PROJECT_ID
from .monitor_training import MONITOR_ROOT, monitor
from .training.run import assert_fresh_version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--games-per-update", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--interval-seconds", type=float, default=60)
    args = parser.parse_args()
    assert_fresh_version(args.version)
    monitor_root = MONITOR_ROOT / args.version
    monitor_root.mkdir(parents=True, exist_ok=False)
    log_path = monitor_root / "training.log"
    command = [
        sys.executable, "-m", f"train.{PROJECT_ID}", "train",
        "--version", args.version,
        "--updates", str(args.updates),
        "--games-per-update", str(args.games_per_update),
        "--workers", str(args.workers),
        "--device", args.device,
        "--seed", str(args.seed),
        "--eval-every", str(args.eval_every),
        "--wandb-mode", "online",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        watchdog = Namespace(
            version=args.version,
            pid=child.pid,
            log=log_path,
            interval_seconds=args.interval_seconds,
            first_metrics_minutes=30.0,
            metrics_stale_minutes=75.0,
            min_disk_gib=10.0,
            min_ram_gib=1.0,
        )
        try:
            result = monitor(watchdog)
        except KeyboardInterrupt:
            child.terminate()
            child.wait(timeout=30)
            raise
        if result != 0:
            return result
        return child.wait(timeout=30)


if __name__ == "__main__":
    raise SystemExit(main())

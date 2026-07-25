"""Generate a tiny synthetic training curve to validate W&B plumbing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rl_environment.logging import TrainingLogger


def run_fake_training(output: Path, *, steps: int = 5) -> dict[str, object]:
    """Write deterministic fake BC metrics through the production logger."""
    if steps < 1:
        raise ValueError("steps must be positive")
    metrics_path = output / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"smoke output already exists: {metrics_path}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "task": "ptcg_wandb_fake_bc_smoke_v1",
                "seed": 7,
                "synthetic": True,
                "steps": steps,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with TrainingLogger(metrics_path, output / "tensorboard") as logger:
        for epoch in range(1, steps + 1):
            progress = epoch / steps
            logger.log(
                epoch,
                {
                    "train/bc_loss": round(1.0 - 0.7 * progress, 6),
                    "train/action_accuracy": round(0.35 + 0.5 * progress, 6),
                    "validation/bc_loss": round(1.05 - 0.62 * progress, 6),
                    "validation/action_accuracy": round(0.3 + 0.46 * progress, 6),
                    "validation/legal_action_rate": 1.0,
                },
            )
    return {
        "mode": os.environ.get("WANDB_MODE", "disabled"),
        "metrics": str(metrics_path.resolve()),
        "tensorboard": str((output / "tensorboard").resolve()),
        "wandb": str((output / "wandb").resolve()),
        "steps": steps,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("disabled", "offline", "online"),
        default="offline",
    )
    args = parser.parse_args()
    os.environ["WANDB_MODE"] = args.mode
    os.environ["WANDB_ALLOW_NONCANONICAL"] = "1"
    os.environ.setdefault("WANDB_JOB_TYPE", "bc_train")
    os.environ.setdefault("WANDB_TAGS", "smoke,synthetic")
    print(json.dumps(run_fake_training(args.output, steps=args.steps), indent=2))


if __name__ == "__main__":
    _main()

"""Small official-engine rollout smoke for the heterogeneous collector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from . import FOCAL_DECK_ID
from .league import load_frozen_catalog
from .policy import load_actor_critic
from .rollout import HeterogeneousRolloutCollector, RolloutJob


def runtime_root() -> Path:
    roots = sorted(Path("evaluation/arena/opponents").glob("*/cg/game.py"))
    if not roots:
        raise FileNotFoundError("Frozen official-engine runtime is unavailable")
    return roots[0].parents[1].resolve()


def smoke_jobs(count: int, *, seed: int = 22022) -> list[RolloutJob]:
    if count < 2:
        raise ValueError("smoke count must be at least two")
    catalog = load_frozen_catalog()
    focal = next(item for item in catalog if item.deck_id == FOCAL_DECK_ID)
    opponents = [item for item in catalog if item.deck_id != FOCAL_DECK_ID]
    root = runtime_root()
    return [
        RolloutJob(
            game_id=f"smoke-{index:04d}",
            opponent_id=opponents[(index // 2) % len(opponents)].deck_id,
            focal_first=index % 2 == 0,
            seed=seed + index,
            source_policy_update=0,
            focal_deck=focal.deck,
            opponent_deck=opponents[(index // 2) % len(opponents)].deck,
            runtime_root=root,
        )
        for index in range(count)
    ]


def run_smoke(*, games: int = 4, workers: int = 2, device: str = "cuda:0") -> dict[str, object]:
    model, _, identity = load_actor_critic(device)
    collector = HeterogeneousRolloutCollector(
        model, device=torch.device(device), workers=workers, mode="sample"
    )
    started = time.perf_counter()
    episodes = collector.collect(smoke_jobs(games))
    wall = time.perf_counter() - started
    errors = [episode.error for episode in episodes if not episode.valid]
    result = {
        "games": len(episodes),
        "valid": sum(episode.valid for episode in episodes),
        "errors": errors,
        "wins": sum(episode.reward == 1 for episode in episodes),
        "losses": sum(episode.reward == -1 for episode in episodes),
        "draws": sum(episode.reward == 0 for episode in episodes),
        "decisions": sum(len(episode.decisions) for episode in episodes),
        "wall_seconds": wall,
        "initial_checkpoint_sha256": identity.checkpoint_sha256,
        "representation_sha256": model.representation_sha256(),
        **collector.metrics(),
    }
    if errors or result["valid"] != games:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run_smoke(games=args.games, workers=args.workers, device=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

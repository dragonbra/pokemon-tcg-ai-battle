from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.cli import _load_official_card_ids
from evaluation.packages.loader import load_submission_package

from .constants import REPOSITORY_ROOT
from .policy.actor_critic import load_source_actor_critic
from .rollout.collector import RolloutCollector
from .rollout.protocol import RolloutJob


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0017 CPU official-engine rollout smoke")
    parser.add_argument("--opponent", default="dragapult_ex_01")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.games < 1 or args.workers < 1:
        raise ValueError("games and workers must be positive")

    torch.set_num_threads(1)
    evaluation_root = REPOSITORY_ROOT / "evaluation"
    root = evaluation_root / "arena/opponents" / args.opponent
    opponent = load_submission_package(
        root,
        _load_official_card_ids(evaluation_root),
        name=args.opponent,
    )
    model, metadata = load_source_actor_critic()
    device = torch.device("cpu")
    model.to(device).eval()
    collector = RolloutCollector(
        model,
        device=device,
        workers=args.workers,
        mode="sample" if args.sample else "greedy",
    )
    jobs = [
        RolloutJob(
            episode_id=f"smoke-{index:04d}",
            opponent=opponent,
            candidate_first=index % 2 == 0,
            seed=20260728 + index,
        )
        for index in range(args.games)
    ]
    episodes = collector.collect(jobs)
    print(
        json.dumps(
            {
                "source_version": metadata["version"],
                "device": str(device),
                "episodes": [
                    {
                        "episode_id": episode.episode_id,
                        "valid": episode.valid,
                        "reward": episode.reward,
                        "status": episode.status,
                        "error": episode.error,
                        "candidate_first": episode.candidate_first,
                        "engine_selections": episode.engine_selections,
                        "candidate_decisions": len(episode.decisions),
                        "complete_rounds": episode.complete_rounds,
                        "diagnostics": episode.diagnostics,
                    }
                    for episode in episodes
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if episodes and all(item.valid for item in episodes) else 1


if __name__ == "__main__":
    raise SystemExit(main())

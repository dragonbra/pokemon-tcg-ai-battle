"""Repeatable non-training official-engine gate for League rollout."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from . import FOCAL_DECK_ID
from .decks import load_deck_plugins
from .league import DEFAULT_DECK_ROOT
from .policy import load_league_actor_critic
from .rollout import LeaguePolicyView, LeagueRolloutCollector, RolloutJob


def run_rollout_smoke(*, device: str = "cuda:0", workers: int = 4,
                      deck_root: Path = DEFAULT_DECK_ROOT) -> dict[str, object]:
    plugins = {item.deck_id: item for item in load_deck_plugins(deck_root)}
    focal = plugins[FOCAL_DECK_ID]
    opponent_ids = ("festival_lead_dipplin_002", "mega_kangaskhan_ex_crustle_004")
    missing = [item for item in opponent_ids if item not in plugins]
    if missing:
        raise ValueError(f"smoke opponents are absent: {missing}")
    runtime_candidates = sorted(Path("evaluation/arena/opponents").glob("*/cg/game.py"))
    if not runtime_candidates:
        raise FileNotFoundError("no validated official engine runtime package exists")
    runtime_root = runtime_candidates[0].parents[1].resolve()
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
        # PyTorch 2.11 + CUDA 12.8 can reject peak-stat reset before the first context exists.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch_device)
    model, identity = load_league_actor_critic(device)
    jobs = []
    for index, opponent_id in enumerate(opponent_ids):
        opponent = plugins[opponent_id]
        for focal_first in (True, False):
            jobs.append(RolloutJob(
                game_id=f"smoke-{index}-{int(focal_first)}", focal_deck_id=focal.deck_id,
                opponent_deck_id=opponent.deck_id, opponent_view=LeaguePolicyView.FROZEN,
                focal_first=focal_first, seed=22000 + index * 2 + int(focal_first),
                source_policy_update=0, focal_deck=focal.deck, opponent_deck=opponent.deck,
                runtime_root=runtime_root,
            ))
    collector = LeagueRolloutCollector(model, device=torch_device, workers=workers, mode="sample")
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    wall = time.perf_counter() - started
    result = {
        "gate": "0024_official_engine_rollout_smoke_v1",
        "foundation_sha256": identity.weights_sha256,
        "runtime_root": str(runtime_root), "device": str(torch_device),
        "games": len(episodes), "finished": sum(item.valid for item in episodes),
        "errors": sum(not item.valid for item in episodes), "wall_seconds": wall,
        "episodes_per_second": len(episodes) / wall,
        "episodes": [
            {"game_id": item.job.game_id, "opponent": item.job.opponent_deck_id,
             "focal_first": item.job.focal_first, "reward": item.reward,
             "decisions": len(item.decisions), "valid": item.valid, "error": item.error}
            for item in episodes
        ],
        **collector.metrics(),
    }
    if torch_device.type == "cuda":
        result["cuda/allocated_bytes"] = torch.cuda.memory_allocated(torch_device)
        result["cuda/reserved_bytes"] = torch.cuda.memory_reserved(torch_device)
        result["cuda/peak_allocated_bytes"] = torch.cuda.max_memory_allocated(torch_device)
    if result["finished"] != 4 or result["errors"] != 0:
        raise RuntimeError(f"official-engine smoke failed: {result}")
    return result


__all__ = ["run_rollout_smoke"]

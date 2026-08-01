"""Official-engine rollout-supply scaling benchmark for formal League PPO."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import torch

from .decks import load_deck_plugins
from .league import DEFAULT_DECK_ROOT
from .policy import load_league_actor_critic
from .rollout import LeagueRolloutCollector
from .storage import runtime_storage
from .training.run import schedule_jobs


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_worker_benchmark(*, workers: int, games: int = 256, coalesce_ms: float = 5.0,
                         device: str = "cuda:0", output: Path,
                         deck_root: Path = DEFAULT_DECK_ROOT) -> dict[str, object]:
    if workers < 1 or games < 1 or coalesce_ms < 0:
        raise ValueError("benchmark concurrency values are invalid")
    plugins = load_deck_plugins(deck_root)
    runtime = sorted(Path("evaluation/arena/opponents").glob("*/cg/game.py"))[0].parents[1].resolve()
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch_device)
    model, identity = load_league_actor_critic(device)
    collector = LeagueRolloutCollector(
        model, device=torch_device, workers=workers, mode="sample", coalesce_ms=coalesce_ms,
    )
    jobs = schedule_jobs(
        plugins, count=games, update=0, seed=22022, runtime_root=runtime, evaluation=False,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    wall_seconds = time.perf_counter() - started
    metrics = collector.metrics()
    requests = metrics["rollout/inference_requests"]
    batches = metrics["rollout/inference_batches"]
    storage = runtime_storage(output.parent)
    result: dict[str, object] = {
        "schema_version": "0024_worker_scaling_v1",
        "foundation_sha256": identity.weights_sha256,
        "official_engine_runtime": str(runtime),
        "workers": workers, "games": games, "coalesce_ms": coalesce_ms,
        "device": str(torch_device), "finished": sum(item.valid for item in episodes),
        "errors": sum(not item.valid for item in episodes),
        "wall_seconds": wall_seconds, "episodes_per_second": games / wall_seconds,
        "mean_batch_size": requests / batches if batches else 0.0,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **metrics, **storage.metrics(),
    }
    if torch_device.type == "cuda":
        result.update({
            "cuda/allocated_bytes": torch.cuda.memory_allocated(torch_device),
            "cuda/reserved_bytes": torch.cuda.memory_reserved(torch_device),
            "cuda/peak_allocated_bytes": torch.cuda.max_memory_allocated(torch_device),
        })
    if result["finished"] != games or result["errors"] != 0:
        raise RuntimeError(f"worker benchmark failed completion gate: {result}")
    _atomic_json(output, result)
    return result


__all__ = ["run_worker_benchmark"]

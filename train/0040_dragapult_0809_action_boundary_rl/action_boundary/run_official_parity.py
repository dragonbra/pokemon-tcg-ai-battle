"""Find reproducible n=1..5 Phantom states, then run exhaustive official parity."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from evaluation.runtime.seeded import build_seeded_runtime

from ..initialization import build_update0_model
from ..rollout import FullSemanticRolloutCollector
from ..training.run_full_semantic import build_jobs, focal_deck
from .official_parity import exhaustive_scenario_parity, scenario_from_sample

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experiments/0040_dragapult_0809_action_boundary_rl/OFFICIAL_PARITY.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-games", type=int, default=512)
    parser.add_argument("--chunk-games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--engines-per-worker", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    deck = focal_deck()
    model, _ = build_update0_model(deck, device=device)
    found = {}
    searched = errors = 0
    for offset in range(0, args.search_games, args.chunk_games):
        jobs = build_jobs(source_policy_update=0, seed=380_200_000 + offset,
                          count=min(args.chunk_games, args.search_games - offset))
        jobs = [replace(job, game_id=f"parity-search-{offset + index:06d}",
                        action_boundary_mode="shadow", trace_policy="compact",
                        include_action_prefix=True)
                for index, job in enumerate(jobs)]
        collector = FullSemanticRolloutCollector(
            model, model.actor, device=device, worker_processes=min(args.workers, len(jobs)),
            engines_per_worker=min(args.engines_per_worker, len(jobs)),
            inference_channels_per_role=min(args.engines_per_worker, len(jobs)),
            mode="sample", coalesce_ms=5.0, timeout_seconds=240.0,
        )
        episodes = collector.collect(jobs)
        searched += len(episodes)
        errors += sum(not item.valid for item in episodes)
        for episode in episodes:
            for sample in episode.diagnostics.get("allocation_bc_samples", []):
                found.setdefault(len(sample.counters), {
                    key: value for key, value in sample.__dict__.items()
                } if hasattr(sample, "__dict__") else {
                    name: getattr(sample, name) for name in sample.__slots__
                })
        if set(found) == {1, 2, 3, 4, 5}:
            break
    if set(found) != {1, 2, 3, 4, 5}:
        raise RuntimeError(f"parity search missed target counts: {sorted(set(range(1, 6)) - set(found))}")
    runtime = build_seeded_runtime()
    results = []
    for n in range(1, 6):
        scenario = scenario_from_sample(found[n], deck)
        results.append(exhaustive_scenario_parity(scenario, runtime.library_path))
    report = {
        "schema_version": "0038_official_phantom_parity_v1",
        "search_games": searched, "search_errors": errors,
        "expected_allocation_counts": [1, 7, 28, 84, 210],
        "results": results,
        "total_allocations": sum(item["allocations"] for item in results),
        "parity_failures": sum(item["parity_failures"] for item in results),
        "comparisons": [
            "macro primitive sequence == legacy canonical primitive sequence",
            "final stable official observation semantics exact equality",
            "reversed-order alias final official state exact equality",
            "actor/turn/phase/priority/legal/reward/terminal/winner/deck-visible state",
            "opaque serialized state differences audited against identical replay instability",
            "engine RNG parity follows identical engine seed and identical primitive Select calls",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if report["parity_failures"]:
        raise RuntimeError(f"official parity failures: {report['parity_failures']}")


if __name__ == "__main__":
    main()

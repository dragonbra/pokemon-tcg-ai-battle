"""Run the official CPU-256 Policy-0809 Frozen preflight for 0042."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from ..candidate_deployment import materialize_kaggle_evaluation_candidate
from ..evaluation.frozen_jobs import CANONICAL_CONTRACT_ID, build_frozen_jobs
from ..training import run_full_semantic as runner


SCHEMA = "0042_cpu256_policy0809_preflight_v1"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(*, checkpoint: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    device = torch.device("cuda:0")
    model, candidate_audit = materialize_kaggle_evaluation_candidate(
        source=runner.CANDIDATE_ROOT,
        checkpoint=checkpoint,
        deck=runner.focal_deck(),
        device=device,
        temporary_root=runner.ROOT / ".tmp/evaluation/0042_cpu256_candidate",
    )
    opponent = runner.load_frozen_opponent(device)
    jobs, schedule_sha256 = build_frozen_jobs(
        focal_deck_id=runner.FOCAL_DECK_ID,
        focal_deck=runner.focal_deck(),
        runtime_root=runner.runtime_root(),
        source_policy_update=0,
        focal_deployment_identity=candidate_audit.effective_candidate_sha256,
        opponent_effective_policy_sha256=(
            opponent._policy_identity_audit.effective_policy_sha256
        ),
        evaluation_units=1,
    )
    config = runner.RunConfig(
        version="V999_cpu256_preflight",
        updates=1,
        worker_processes=16,
        engines_per_worker=8,
        inference_channels_per_role=8,
        engine_backend="official",
        launch_formal=False,
        wandb_mode="offline",
    )
    collector = runner.build_collector(
        model, opponent, config, mode="greedy", record_trajectory=False
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    elapsed = time.perf_counter() - started
    metrics = collector.metrics()
    invalid = [episode for episode in episodes if not episode.valid or episode.error]
    semantic_fallbacks = [
        episode for episode in episodes
        if episode.diagnostics.get("macro_fallback_reason")
        not in {None, runner.CHANCE_BOUNDARY_FALLBACK}
    ]
    if len(episodes) != 256 or invalid or semantic_fallbacks:
        raise RuntimeError(
            "CPU-256 health gate failed: "
            f"episodes={len(episodes)} invalid={len(invalid)} "
            f"semantic_fallbacks={len(semantic_fallbacks)}"
        )
    results_path = output.with_name(output.stem + "-per-game.json")
    runner._persist_frozen_results(
        results_path,
        episodes,
        checkpoint_update=0,
        panel_version=CANONICAL_CONTRACT_ID,
        schedule_sha256=schedule_sha256,
        opponent_identity_audit=opponent._policy_identity_audit,
        candidate_deployment_audit=candidate_audit,
        expected_games=256,
        benchmark_kind="cpu_256",
    )
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "games": len(episodes),
        "valid": len(episodes) - len(invalid),
        "errors": len(invalid),
        "semantic_fallbacks": len(semantic_fallbacks),
        "elapsed_seconds": elapsed,
        "games_per_second": len(episodes) / max(elapsed, 1e-9),
        "schedule_sha256": schedule_sha256,
        "candidate_deployment_identity_audit": candidate_audit.to_manifest(),
        "opponent_policy_identity_audit": (
            opponent._policy_identity_audit.to_manifest()
        ),
        "collector_metrics": metrics,
        "per_game_results": str(results_path),
    }
    _atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        checkpoint=args.checkpoint.resolve(), output=args.output.resolve()
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

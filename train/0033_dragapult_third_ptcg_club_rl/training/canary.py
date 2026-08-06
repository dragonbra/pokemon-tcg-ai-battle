from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rollout import ResidentRolloutCollector
from .runtime import build_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies-per-opponent", type=int, default=4)
    parser.add_argument("--opponents-per-cohort", type=int, default=5)
    parser.add_argument("--cohort-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--actor-execution", choices=("eager", "cuda_graph"), default="cuda_graph")
    args = parser.parse_args()
    runtime = build_runtime(
        rules=args.rules,
        extension_dir=args.extension_dir,
        source_checkpoint=args.source_checkpoint,
        support_report=args.support_report,
        frozen_root=args.frozen_root,
        checkpoint_root=args.checkpoint_root,
        copies_per_opponent=args.copies_per_opponent,
        opponents_per_cohort=args.opponents_per_cohort,
        cohort_index=args.cohort_index,
    )
    collector = ResidentRolloutCollector(
        engine=runtime.engine,
        policy=runtime.policy,
        decks=runtime.decks,
        seeds=runtime.seeds,
        opponent_head_indices=runtime.opponent_head_indices,
        steps=args.steps,
        actor_execution=args.actor_execution,
    )
    result = collector.collect(source_policy_update=0)
    payload = {
        "schema": "0033_resident_rollout_canary_v1",
        "passed": True,
        "focal_deck_id": "dragapult_third_ptcg_club",
        "base_opponents": list(runtime.base_opponent_deck_ids),
        "full_opponent_pool": list(runtime.full_opponent_deck_ids),
        "lanes": len(runtime.opponent_deck_ids),
        "metrics": result.metrics,
        "prepared_decisions": result.batch.decisions,
        "prepared_complete_episodes": result.batch.completed_episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

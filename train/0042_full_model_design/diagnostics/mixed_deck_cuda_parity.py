"""Compare 256 heterogeneous CUDA lanes with an exact-deck grouped reference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from engine_cuda.tools.run_official_semantic0031_v2_parity_scaffold import (
    FAMILY_KEYS,
    GLOBAL_KEYS,
)

from ..candidate_deployment import materialize_kaggle_evaluation_candidate
from ..evaluation.frozen_jobs import build_frozen_jobs
from ..rollout.cuda_collector import ChunkedCudaRolloutCollector
from ..rollout.deck_routing import exact_deck_sha256
from ..training import run_full_semantic as runner


SCHEMA = "0042_mixed_deck_cuda_grouped_parity_v1"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _trace_sink(
    jobs: Sequence[Any], traces: dict[str, list[dict[str, Any]]]
):
    def semantic_hash(mapping: dict[str, torch.Tensor], row: int) -> str:
        digest = hashlib.sha256()
        for name in GLOBAL_KEYS:
            tensor = mapping[name][row].contiguous()
            digest.update(name.encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        for family, names in FAMILY_KEYS.items():
            mask = mapping[f"{family}_mask"][row].bool()
            live = int(mask.sum())
            digest.update(f"{family}:{live}".encode("ascii"))
            for name in names:
                tensor = mapping[name][row, :live].contiguous()
                digest.update(name.encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def sink(**payload: Any) -> None:
        ready_rows = payload["ready"].nonzero(as_tuple=False).flatten()
        if not ready_rows.numel():
            return
        job_indices = payload["lane_job"].index_select(0, ready_rows).cpu().tolist()
        actors = payload["actor"].index_select(0, ready_rows).cpu().tolist()
        turns = payload["turns"].index_select(0, ready_rows).cpu().tolist()
        decisions = payload["lane_decisions"].index_select(
            0, ready_rows
        ).cpu().tolist()
        lengths = payload["lengths"].index_select(0, ready_rows).cpu().tolist()
        actions = payload["actions"].index_select(0, ready_rows).cpu()
        contexts = payload["semantic"]["global_cat"].index_select(
            0, ready_rows
        )[:, 1].cpu().tolist()
        semantic_names = set(GLOBAL_KEYS)
        for family, names in FAMILY_KEYS.items():
            semantic_names.add(f"{family}_mask")
            semantic_names.update(names)
        semantic_rows = {
            name: payload["semantic"][name].index_select(0, ready_rows).cpu()
            for name in semantic_names
        }
        for offset, job_index in enumerate(job_indices):
            length = int(lengths[offset])
            traces[jobs[int(job_index)].game_id].append({
                "decision": int(decisions[offset]),
                "turn": int(turns[offset]),
                "actor": int(actors[offset]),
                "context": int(contexts[offset]) - 1,
                "action": actions[offset, :length].tolist(),
                "semantic_sha256": semantic_hash(semantic_rows, offset),
            })

    return sink


def _collector(
    model: Any,
    opponent: Any,
    jobs: Sequence[Any],
    traces: dict[str, list[dict[str, Any]]],
    *,
    role_compacted: bool,
) -> ChunkedCudaRolloutCollector:
    return ChunkedCudaRolloutCollector(
        model,
        opponent,
        rollout_batch_size=len(jobs),
        trajectory_games_per_update=None,
        device=torch.device("cuda:0"),
        rules_path=runner.CUDA_RULES,
        extension_dir=runner.CUDA_EXTENSION,
        lane_count=len(jobs),
        mode="greedy",
        check_interval=8,
        record_trajectory=False,
        agent_selects_first_player=True,
        decision_trace_sink=_trace_sink(jobs, traces),
        opponent_policy_id=runner.TRAINING_OPPONENT_POLICY_ID,
        opponent_identity_audit=opponent._policy_identity_audit,
        role_compacted=role_compacted,
    )


def _episode_signature(episode: Any) -> dict[str, Any]:
    return {
        "valid": bool(episode.valid),
        "reward": episode.reward,
        "turns": int(episode.turns),
        "termination": episode.diagnostics.get("termination"),
        "focal_first": bool(episode.job.focal_first),
        "actual_focal_first": episode.diagnostics.get("actual_focal_first"),
        "first_player_chooser": episode.diagnostics.get("first_player_chooser"),
        "first_player_action": episode.diagnostics.get("first_player_action"),
        "opponent_exact_deck_sha256": episode.diagnostics.get(
            "opponent_exact_deck_sha256"
        ),
        "opponent_effective_policy_sha256": episode.diagnostics.get(
            "opponent_effective_policy_sha256"
        ),
        "lane_routing_audit_status": episode.diagnostics.get(
            "lane_routing_audit_status"
        ),
    }


def run(*, checkpoint: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    device = torch.device("cuda:0")
    model, candidate_audit = materialize_kaggle_evaluation_candidate(
        source=runner.CANDIDATE_ROOT,
        checkpoint=checkpoint,
        deck=runner.focal_deck(),
        device=device,
        temporary_root=runner.ROOT / ".tmp/evaluation/0042_mixed_deck_parity_candidate",
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
    if len(jobs) != 256:
        raise RuntimeError("mixed-deck parity requires one complete 256-game unit")

    mixed_traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()
    mixed_collector = _collector(
        model, opponent, jobs, mixed_traces, role_compacted=True
    )
    mixed = mixed_collector.collect(list(jobs))
    mixed_seconds = time.perf_counter() - started
    mixed_metrics = mixed_collector.metrics()

    groups: dict[str, list[Any]] = defaultdict(list)
    for job in jobs:
        groups[exact_deck_sha256(job.opponent_deck)].append(job)
    grouped_traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_jobs = [
        job
        for deck_sha256, group in sorted(groups.items())
        for job in group
        if exact_deck_sha256(job.opponent_deck) == deck_sha256
    ]
    if len(grouped_jobs) != 256:
        raise RuntimeError("grouped reference changed job cardinality")
    started = time.perf_counter()
    grouped_collector = _collector(
        model, opponent, grouped_jobs, grouped_traces, role_compacted=False
    )
    grouped_episodes = grouped_collector.collect(grouped_jobs)
    grouped_metrics = grouped_collector.metrics()
    grouped_seconds = time.perf_counter() - started

    mixed_by_id = {episode.job.game_id: episode for episode in mixed}
    grouped_by_id = {episode.job.game_id: episode for episode in grouped_episodes}
    if set(mixed_by_id) != set(grouped_by_id) or len(mixed_by_id) != 256:
        raise RuntimeError("mixed/grouped result game identities disagree")
    first_divergence = None
    for game_id in sorted(mixed_by_id):
        mixed_signature = _episode_signature(mixed_by_id[game_id])
        grouped_signature = _episode_signature(grouped_by_id[game_id])
        if mixed_signature != grouped_signature:
            first_divergence = {
                "game_id": game_id,
                "category": "terminal_or_identity",
                "mixed": mixed_signature,
                "grouped": grouped_signature,
            }
            break
        mixed_trace = mixed_traces[game_id]
        grouped_trace = grouped_traces[game_id]
        if mixed_trace != grouped_trace:
            index = next(
                (
                    offset
                    for offset, pair in enumerate(zip(mixed_trace, grouped_trace))
                    if pair[0] != pair[1]
                ),
                min(len(mixed_trace), len(grouped_trace)),
            )
            first_divergence = {
                "game_id": game_id,
                "category": "decision_trace",
                "decision_offset": index,
                "mixed": mixed_trace[index] if index < len(mixed_trace) else None,
                "grouped": grouped_trace[index] if index < len(grouped_trace) else None,
                "mixed_decisions": len(mixed_trace),
                "grouped_decisions": len(grouped_trace),
            }
            break

    deck_frequency = Counter(
        exact_deck_sha256(job.opponent_deck) for job in jobs
    )
    passed = (
        first_divergence is None
        and len(groups) == 55
        and all(episode.valid for episode in mixed)
        and all(episode.valid for episode in grouped_episodes)
        and mixed_metrics.get("rollout/lane_routing_audit_pass") == 1.0
        and mixed_metrics.get("rollout/lane_routing_audit_failures") == 0.0
        and grouped_metrics.get("rollout/lane_routing_audit_pass") == 1.0
        and grouped_metrics.get("rollout/lane_routing_audit_failures") == 0.0
    )
    report = {
        "schema_version": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "games": 256,
        "unique_exact_decks": len(groups),
        "schedule_sha256": schedule_sha256,
        "focal_deck_id": runner.FOCAL_DECK_ID,
        "focal_exact_deck_sha256": runner.FOCAL_EXACT_DECK_SHA256,
        "candidate_deployment_identity_audit": candidate_audit.to_manifest(),
        "opponent_policy_identity_audit": (
            opponent._policy_identity_audit.to_manifest()
        ),
        "deck_frequency": dict(sorted(deck_frequency.items())),
        "mixed": {
            "seconds": mixed_seconds,
            "games_per_second": 256 / max(mixed_seconds, 1e-9),
            "metrics": mixed_metrics,
        },
        "grouped_reference": {
            "seconds": grouped_seconds,
            "games_per_second": 256 / max(grouped_seconds, 1e-9),
            "groups": len(groups),
            "execution": "single_256_lane_batch_sorted_by_exact_deck",
            "metrics": grouped_metrics,
        },
        "trace_decisions": sum(len(trace) for trace in mixed_traces.values()),
        "first_divergence": first_divergence,
    }
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError(f"mixed-deck CUDA parity failed: {first_divergence}")
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

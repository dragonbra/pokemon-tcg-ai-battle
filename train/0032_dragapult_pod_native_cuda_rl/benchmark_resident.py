"""Benchmark the 0032 actor inside the fully resident official CUDA loop."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
CUDA_ROOT = ROOT / "engine_cuda"
for path in (CUDA_ROOT / "python", CUDA_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from run_official_seeded_reset_paired import read_deck  # noqa: E402


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[position]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--supported-decks", type=Path, required=True)
    parser.add_argument("--focal-deck", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-action-steps", type=int, default=64)
    parser.add_argument("--copies-per-opponent", type=int, default=1)
    parser.add_argument(
        "--actor-execution",
        choices=("eager", "cuda_graph"),
        default="cuda_graph",
    )
    parser.add_argument("--seed-start", type=int, default=2026080601)
    parser.add_argument("--model-seed", type=int, default=32032)
    args = parser.parse_args()
    if (
        min(args.steps, args.max_action_steps, args.copies_per_opponent) < 1
        or args.warmup < 0
    ):
        raise ValueError("steps/max-action-steps must be positive and warmup non-negative")
    sys.path.insert(0, str(args.extension_dir.resolve()))
    import _ptcg_cuda

    model_module = importlib.import_module("train.0032_dragapult_pod_native_cuda_rl.model")
    transfer = importlib.import_module("train.0032_dragapult_pod_native_cuda_rl.transfer")
    device = torch.device("cuda")
    opponent_snapshot = [
        Path(line.strip()).resolve()
        for line in args.supported_decks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(opponent_snapshot) != 38:
        raise ValueError(
            f"expected the admitted 38-deck snapshot, got {len(opponent_snapshot)}"
        )
    deck_paths = opponent_snapshot * args.copies_per_opponent
    focal = read_deck(args.focal_deck)
    opponents = [read_deck(path) for path in deck_paths]
    batch_size = len(opponents)
    deck_pairs = [[focal, opponent] for opponent in opponents]
    decks = torch.tensor(deck_pairs, dtype=torch.int32, device=device)
    seeds = torch.arange(
        args.seed_start, args.seed_start + batch_size, dtype=torch.int64, device=device
    )
    lane_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    engine = create_official_engine(args.rules.read_bytes(), batch_size=batch_size)
    engine.reset_seeded_interactive_masked(decks, seeds, lane_mask)
    config = model_module.ModelConfig(max_action_steps=args.max_action_steps)
    torch.manual_seed(args.model_seed)
    model = model_module.PodNativeActorCritic(config).to(device).eval()
    transfer_report = transfer.load_0031_initialization(model, args.source_checkpoint)
    model.requires_grad_(False)

    graph = None
    graph_actions = None
    graph_capture_equal = None
    if args.actor_execution == "cuda_graph":
        engine.advance_to_decision()
        graph_raw_batch = dict(engine.encode_policy_v1())

        def graph_actor() -> Any:
            batch = dict(graph_raw_batch)
            batch["entity_mask"] = graph_raw_batch["entity_mask"].bool()
            batch["option_mask"] = graph_raw_batch["option_mask"].bool()
            return model.greedy(batch)

        with torch.inference_mode():
            eager_actions = graph_actor()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_actions = graph_actor()
            graph.replay()
            torch.cuda.synchronize(device)
        graph_capture_equal = bool(
            torch.equal(eager_actions.sequences, graph_actions.sequences)
            and torch.equal(eager_actions.lengths, graph_actions.lengths)
            and torch.equal(eager_actions.legal, graph_actions.legal)
        )
        if not graph_capture_equal:
            raise RuntimeError("CUDA Graph actor output differs from eager output")

    decisions = torch.zeros((), dtype=torch.int64, device=device)
    episodes = torch.zeros((), dtype=torch.int64, device=device)
    illegal = torch.zeros((), dtype=torch.int64, device=device)
    overflows = torch.zeros((), dtype=torch.int64, device=device)

    def step(*, measure: bool) -> None:
        terminal = engine.statuses().eq(TERMINAL)
        seeds.add_(terminal.long() * batch_size)
        engine.reset_seeded_interactive_masked(decks, seeds, terminal)
        if measure:
            episodes.add_(terminal.long().sum())
        engine.advance_to_decision()
        ready = engine.statuses().eq(NEEDS_ACTION)
        raw_batch = dict(engine.encode_policy_v1())
        if graph is None:
            batch = raw_batch
            batch["entity_mask"] = batch["entity_mask"].bool()
            batch["option_mask"] = batch["option_mask"].bool()
            actions = model.greedy(batch)
        else:
            graph.replay()
            actions = graph_actions
            batch = raw_batch
        if measure:
            # Codec tensors are borrowed engine views and may be refreshed by apply.
            decisions.add_(ready.long().sum())
            illegal.add_((ready & ~actions.legal).long().sum())
            overflows.add_(
                (ready & batch["max_count"].gt(args.max_action_steps)).long().sum()
            )
        engine.pack_actions(actions.sequences, actions.lengths)
        engine.apply_packed_actions()

    with torch.inference_mode():
        for _ in range(args.warmup):
            step(measure=False)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push("0032_pod_native_resident_rollout")
        total_start.record()
        for index in range(args.steps):
            starts[index].record()
            step(measure=True)
            ends[index].record()
        total_end.record()
        torch.cuda.nvtx.range_pop()
        total_end.synchronize()

    elapsed_seconds = total_start.elapsed_time(total_end) / 1000.0
    step_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    decision_count = int(decisions.item())
    episode_count = int(episodes.item())
    illegal_count = int(illegal.item())
    overflow_count = int(overflows.item())
    statuses = torch.bincount(engine.statuses().long(), minlength=4).cpu().tolist()
    resident_decisions_per_second = decision_count / elapsed_seconds
    resident_episodes_per_second = episode_count / elapsed_seconds
    cpu_reference = {
        "source": "evaluation/arena/combat_mat/0031_latest_frozen_test/THROUGHPUT.md",
        "official_cpu_engine_cpu_candidate_selections_per_second": 168.449,
        "official_cpu_engine_cpu_candidate_games_per_second": 1.209,
        "comparison_limit": (
            "contextual same-machine reference only; candidate model, lane pool, and episode "
            "completion window differ, so this is not an apples-to-apples strength comparison"
        ),
    }
    result = {
        "schema": "0032_pod_native_resident_throughput_v1",
        "passed": (
            decision_count > 0
            and illegal_count == 0
            and overflow_count == 0
            and statuses[ERROR] == 0
        ),
        "scope": "official CUDA engine + PolicyCodecV1 + 0032 actor fully resident rollout",
        "policy_strength_claimed": False,
        "focal_deck": str(args.focal_deck),
        "admitted_opponents": len(opponent_snapshot),
        "copies_per_opponent": args.copies_per_opponent,
        "batch_lanes": batch_size,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "max_action_steps": args.max_action_steps,
        "actor_execution": args.actor_execution,
        "cuda_graph_capture_equal_to_eager": graph_capture_equal,
        "decisions": decision_count,
        "completed_episodes": episode_count,
        "elapsed_seconds": elapsed_seconds,
        "decisions_per_second": resident_decisions_per_second,
        "episodes_per_second": resident_episodes_per_second,
        "step_latency_ms": {
            "mean": statistics.fmean(step_ms),
            "p50": percentile(step_ms, 0.50),
            "p95": percentile(step_ms, 0.95),
            "max": max(step_ms),
        },
        "legality": {
            "illegal_actor_rows": illegal_count,
            "max_action_step_overflows": overflow_count,
            "engine_error_lanes": int(statuses[ERROR]),
            "final_status_counts": {
                "idle": int(statuses[0]),
                "needs_action": int(statuses[1]),
                "terminal": int(statuses[2]),
                "error": int(statuses[3]),
            },
        },
        "memory": {
            "engine_allocated_bytes": int(engine.allocated_bytes),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "hot_path": {
            "cpu_engine_calls": 0,
            "cpu_policy_calls": 0,
            "h2d_copies": 0,
            "d2h_copies": 0,
            "host_value_synchronizations": 0,
            "final_measurement_synchronizations": 1,
            "actor_cuda_graph_replay": args.actor_execution == "cuda_graph",
        },
        "contextual_cpu_reference": cpu_reference,
        "contextual_decision_throughput_ratio": (
            resident_decisions_per_second
            / cpu_reference["official_cpu_engine_cpu_candidate_selections_per_second"]
        ),
        "contextual_episode_throughput_ratio": (
            resident_episodes_per_second
            / cpu_reference["official_cpu_engine_cpu_candidate_games_per_second"]
        ),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        "rules_sha256": sha256_file(args.rules),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "model_seed": args.model_seed,
        "transferred_parameter_fraction": transfer_report["transferred_parameter_fraction"],
        "opponent_deck_sha256": [sha256_file(path) for path in opponent_snapshot],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

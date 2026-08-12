from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from run_official_seeded_reset_paired import read_deck, read_fixture  # noqa: E402


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description=(
            "Measure a GPU-only semantic0031 v2 encode/pack/apply/advance decision loop."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=private / "0022_deck40_focal_s1_1_n2" / "manifest.json",
    )
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=CUDA_ENGINE_ROOT / "build" / "torch_0031_linux",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch", type=int, default=152)
    parser.add_argument("--decisions", type=int, default=64)
    parser.add_argument(
        "--until-terminal",
        action="store_true",
        help="Run full games and report games/s instead of a fixed decision window.",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=2048,
        help="Per-lane safety cap when --until-terminal is set.",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=32,
        help="GPU decision count between terminal-status checks in full-game mode.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    args = parse_args()
    if min(args.batch, args.decisions, args.trials, args.max_decisions, args.check_interval) <= 0 or args.warmup < 0:
        raise SystemExit("batch, decisions, trials, max-decisions, and check-interval must be positive; warmup is nonnegative")
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "encode_semantic0031_v2_lanes"):
        raise RuntimeError("_ptcg_cuda lacks encode_semantic0031_v2_lanes")

    manifest: dict[str, Any] = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("semantic0031 v2 manifest contains no cases")
    case = cases[0]
    fixture = read_fixture(workspace_path(str(case["fixture"])))
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    deck_rows = [
        read_deck(workspace_path(str(case["deck0"]))),
        read_deck(workspace_path(str(case["deck1"]))),
    ]
    decks = torch.tensor([deck_rows] * args.batch, dtype=torch.int32, device=device)
    seeds = torch.arange(
        int(fixture.seeds[0]),
        int(fixture.seeds[0]) + args.batch,
        dtype=torch.int64,
        device=device,
    )
    lanes = torch.arange(args.batch, dtype=torch.int32, device=device)
    # The cardinality tensor stays on device.  Choosing the first min_count
    # options is deterministic and valid because codec option rows are legal.
    option_indices = torch.arange(128, dtype=torch.int64, device=device).expand(
        args.batch, 128
    ).contiguous()
    engine = create_official_engine(
        workspace_path(str(manifest["rules"])).read_bytes(),
        batch_size=args.batch,
        device_index=args.device_index,
    )

    def reset() -> None:
        engine.reset_seeded_first_min_semantic(decks, seeds)

    def run_decisions() -> None:
        for _ in range(args.decisions):
            encoded = engine.encode_semantic0031_v2_lanes(lanes)
            engine.pack_actions(option_indices, encoded["min_count"])
            engine.apply_packed_actions()
            engine.advance_to_decision()

    def run_to_terminal() -> tuple[int, int, int]:
        executed = 0
        while executed < args.max_decisions:
            chunk = min(args.check_interval, args.max_decisions - executed)
            for _ in range(chunk):
                encoded = engine.encode_semantic0031_v2_lanes(lanes)
                engine.pack_actions(option_indices, encoded["min_count"])
                engine.apply_packed_actions()
                engine.advance_to_decision()
            executed += chunk
            torch.cuda.synchronize(device)
            statuses = engine.statuses()
            errors = int(statuses.eq(3).sum().item())
            terminals = int(statuses.eq(2).sum().item())
            if errors or terminals == args.batch:
                return executed, terminals, errors
        statuses = engine.statuses()
        return (
            executed,
            int(statuses.eq(2).sum().item()),
            int(statuses.eq(3).sum().item()),
        )

    for _ in range(args.warmup):
        reset()
        if args.until_terminal:
            executed, terminals, errors = run_to_terminal()
            if errors or terminals != args.batch:
                raise RuntimeError(
                    f"warmup did not finish cleanly: decisions={executed} "
                    f"terminal={terminals}/{args.batch} errors={errors}"
                )
        else:
            run_decisions()
    torch.cuda.synchronize(device)

    per_decision_ms: list[float] = []
    game_gpu_ms: list[float] = []
    game_wall_ms: list[float] = []
    decisions_to_terminal: list[int] = []
    error_counts: list[int] = []
    terminal_counts: list[int] = []
    for _ in range(args.trials):
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        reset()
        if args.until_terminal:
            executed, terminals, errors = run_to_terminal()
        else:
            run_decisions()
            statuses = engine.statuses()
            errors = int(statuses.eq(3).sum().item())
            terminals = int(statuses.eq(2).sum().item())
            executed = args.decisions
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start.elapsed_time(end)
        per_decision_ms.append(elapsed_ms / executed)
        if args.until_terminal:
            game_gpu_ms.append(elapsed_ms)
            game_wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
            decisions_to_terminal.append(executed)
        error_counts.append(errors)
        terminal_counts.append(terminals)
        if errors or (args.until_terminal and terminals != args.batch):
            raise RuntimeError(
                f"trial did not finish cleanly: decisions={executed} "
                f"terminal={terminals}/{args.batch} errors={errors}"
            )

    result = {
        "extension": str(Path(_ptcg_cuda.__file__).resolve()),
        "gpu": torch.cuda.get_device_name(args.device_index),
        "batch": args.batch,
        "decisions": args.decisions,
        "warmup": args.warmup,
        "trials": args.trials,
        "until_terminal": args.until_terminal,
        "decision_ms": {
            "median": statistics.median(per_decision_ms),
            "p10": percentile(per_decision_ms, 0.10),
            "p90": percentile(per_decision_ms, 0.90),
            "values": per_decision_ms,
        },
        "error_counts": error_counts,
        "terminal_counts": terminal_counts,
    }
    if args.until_terminal:
        gpu_median_ms = statistics.median(game_gpu_ms)
        wall_median_ms = statistics.median(game_wall_ms)
        result["game_gpu_ms"] = {
            "median": gpu_median_ms,
            "p10": percentile(game_gpu_ms, 0.10),
            "p90": percentile(game_gpu_ms, 0.90),
            "values": game_gpu_ms,
        }
        result["game_wall_ms"] = {
            "median": wall_median_ms,
            "p10": percentile(game_wall_ms, 0.10),
            "p90": percentile(game_wall_ms, 0.90),
            "values": game_wall_ms,
        }
        result["decisions_to_terminal"] = decisions_to_terminal
        result["games_per_second"] = {
            "gpu_only_median": args.batch * 1000.0 / gpu_median_ms,
            "wall_median": args.batch * 1000.0 / wall_median_ms,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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
        description="Measure the GPU-resident semantic0031 v2 encoder only."
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
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
        help="ready-lane counts to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return path


def tensor_bytes(encoded: dict[str, Any]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in encoded.values())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def benchmark_batch(
    *,
    batch: int,
    manifest: dict[str, Any],
    device_index: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, Any]:
    import torch

    case = manifest["cases"][0]
    fixture = read_fixture(workspace_path(str(case["fixture"])))
    deck_rows = [
        read_deck(workspace_path(str(case["deck0"]))),
        read_deck(workspace_path(str(case["deck1"]))),
    ]
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    decks_i32 = torch.tensor([deck_rows] * batch, dtype=torch.int32, device=device)
    seeds = torch.tensor(
        [fixture.seeds[index % len(fixture.seeds)] for index in range(batch)],
        dtype=torch.int64,
        device=device,
    )
    engine = create_official_engine(
        workspace_path(str(manifest["rules"])).read_bytes(),
        batch_size=batch,
        device_index=device_index,
    )
    engine.reset_seeded_first_min_semantic(decks_i32, seeds)
    lanes = torch.arange(batch, dtype=torch.int32, device=device)

    for _ in range(warmup):
        encoded = engine.encode_semantic0031_v2_lanes(lanes)
    torch.cuda.synchronize(device)
    output_bytes = tensor_bytes(encoded)
    del encoded

    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        before = time.perf_counter()
        start.record()
        for _ in range(iterations):
            encoded = engine.encode_semantic0031_v2_lanes(lanes)
        end.record()
        torch.cuda.synchronize(device)
        wall_ms.append((time.perf_counter() - before) * 1000.0 / iterations)
        gpu_ms.append(start.elapsed_time(end) / iterations)
        del encoded

    return {
        "batch": batch,
        "output_bytes": output_bytes,
        "gpu_ms_per_encode": {
            "median": statistics.median(gpu_ms),
            "p10": percentile(gpu_ms, 0.10),
            "p90": percentile(gpu_ms, 0.90),
            "trials": gpu_ms,
        },
        "wall_ms_per_encode": {
            "median": statistics.median(wall_ms),
            "p10": percentile(wall_ms, 0.10),
            "p90": percentile(wall_ms, 0.90),
            "trials": wall_ms,
        },
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.trials <= 0:
        raise SystemExit("warmup must be nonnegative; iterations and trials must be positive")
    if not args.batches or any(batch <= 0 for batch in args.batches):
        raise SystemExit("all batch sizes must be positive")
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "encode_semantic0031_v2_lanes"):
        raise RuntimeError("_ptcg_cuda lacks encode_semantic0031_v2_lanes")

    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    if not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise SystemExit("semantic0031 v2 benchmark manifest contains no cases")
    result = {
        "extension": str(Path(_ptcg_cuda.__file__).resolve()),
        "gpu": torch.cuda.get_device_name(args.device_index),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "trials": args.trials,
        "results": [
            benchmark_batch(
                batch=batch,
                manifest=manifest,
                device_index=args.device_index,
                warmup=args.warmup,
                iterations=args.iterations,
                trials=args.trials,
            )
            for batch in args.batches
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

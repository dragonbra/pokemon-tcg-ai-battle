from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(WORKSPACE_ROOT / "tools"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    EntityPointerPolicyV1DeviceAdapter,
)
from pure_policy_model_v1 import load_policy_checkpoint  # noqa: E402


STATE_ABI_VERSION = 6
STATE_BYTES = 119_936
NEEDS_ACTION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the official resident CUDA state -> PolicyCodecV1 -> "
            "GPU BC -> packed device action loop without CPU-engine fallback."
        )
    )
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument("--fixture", type=Path, default=private / "official_main_fixture.bin")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "bc_models"
            / "archive"
            / "agent_pure_lucario_v1_bc512_e4_probe"
            / "policy.pt"
        ),
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-select", type=int, default=80)
    parser.add_argument("--auto-advance", type=int, default=8)
    parser.add_argument("--mode", choices=("reset", "continuous"), default="reset")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_resident_policy_loop.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_main_expected_state(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 64 or data[:8] != b"PTCGMAIN":
        raise ValueError("official Main fixture header is invalid")
    state_abi, state_bytes, record_bytes, record_count = struct.unpack_from(
        "<IIII", data, 12
    )
    if (
        state_abi != STATE_ABI_VERSION
        or state_bytes != STATE_BYTES
        or record_count <= 0
        or record_bytes < 2 * STATE_BYTES
    ):
        raise ValueError("official Main fixture ABI is incompatible")
    expected_offset = 64 + 320 + STATE_BYTES
    if expected_offset + STATE_BYTES > len(data):
        raise ValueError("official Main fixture is truncated")
    return data[expected_offset : expected_offset + STATE_BYTES]


def codec_batch(engine: Any) -> dict[str, Any]:
    encoded = dict(engine.encode_policy_v1())
    encoded["entity_mask"] = encoded["entity_mask"].bool()
    encoded["option_mask"] = encoded["option_mask"].bool()
    return encoded


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("batch/steps must be positive and warmup must be non-negative")
    if not 1 <= args.max_select <= 80:
        raise ValueError("max-select must be in [1, 80]")
    if args.auto_advance < 0:
        raise ValueError("auto-advance must be non-negative")

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "apply_packed_actions"):
        raise RuntimeError("_ptcg_cuda lacks the resident apply_packed_actions API")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)

    rules = args.rules.resolve()
    fixture = args.fixture.resolve()
    checkpoint = args.checkpoint.resolve()
    for path in (rules, fixture, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    engine = create_official_engine(
        rules.read_bytes(),
        batch_size=args.batch,
        device_index=args.device_index,
    )
    model, _payload = load_policy_checkpoint(checkpoint, device)
    model.eval()
    model.requires_grad_(False)
    adapter = EntityPointerPolicyV1DeviceAdapter(model, max_select=args.max_select)

    fixture_state = first_main_expected_state(fixture)
    initial_cpu = torch.frombuffer(
        bytearray(fixture_state * args.batch), dtype=torch.uint8
    ).view(args.batch, STATE_BYTES)
    initial_device = initial_cpu.to(device=device, non_blocking=False)
    engine.reset_states(initial_device)

    decision_counter = torch.zeros((), dtype=torch.int64, device=device)

    def one_resident_step(*, count_decisions: bool) -> None:
        nonlocal decision_counter
        if args.mode == "reset":
            # Device-to-device reset keeps the benchmark repeatable without a
            # host transfer. Continuous mode instead advances the same states.
            engine.reset_states(initial_device)
            engine.classify()
        else:
            for _ in range(args.auto_advance):
                engine.advance_to_decision()
        batch = codec_batch(engine)
        ready = engine.statuses().eq(NEEDS_ACTION)
        batch["_route_mask"] = ready
        action_indices, action_lengths = adapter.act_device(batch)
        engine.pack_actions(action_indices, action_lengths)
        engine.apply_packed_actions()
        if count_decisions:
            decision_counter = decision_counter + ready.long().sum()

    for _ in range(args.warmup):
        one_resident_step(count_decisions=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nvtx_range = "official_resident_state_codec_policy_action"
    torch.cuda.nvtx.range_push(nvtx_range)
    start.record()
    for _ in range(args.steps):
        one_resident_step(count_decisions=True)
    end.record()
    torch.cuda.nvtx.range_pop()
    # The only timed-loop synchronization/host read is the final measurement.
    end.synchronize()
    elapsed_sec = start.elapsed_time(end) / 1000.0

    decisions = int(decision_counter.item())
    status_counts = torch.bincount(engine.statuses().long(), minlength=4).cpu().tolist()
    result = {
        "schema_version": 1,
        "passed": decisions > 0 and status_counts[3] == 0,
        "scope": "official resident CUDA single-policy loop; fixture-backed state reset",
        "mode": args.mode,
        "batch": args.batch,
        "warmup": args.warmup,
        "steps": args.steps,
        "max_select": args.max_select,
        "decisions": decisions,
        "elapsed_sec": elapsed_sec,
        "decisions_per_sec": decisions / elapsed_sec if elapsed_sec > 0 else 0.0,
        "status_counts": {
            "idle": int(status_counts[0]),
            "needs_action": int(status_counts[1]),
            "terminal": int(status_counts[2]),
            "error": int(status_counts[3]),
        },
        "hot_path": {
            "cpu_engine_calls": 0,
            "cpu_policy_calls": 0,
            "h2d_copies": 0,
            "d2h_copies": 0,
            "host_synchronizations": 0,
            "final_measurement_synchronizations": 1,
            "reset_copy_kind": "device_to_device" if args.mode == "reset" else "none",
        },
        "memory": {
            "official_arena_bytes": int(engine.allocated_bytes),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "checkpoint_sha256": sha256_file(checkpoint),
        "rule_pack_sha256": sha256_file(rules),
        "private_fixture_sha256": sha256_file(fixture),
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        "nvtx_hot_range": nvtx_range,
        "limitations": [
            "Fixture-backed reset is not a complete seeded game setup path.",
            "This gate covers one resident PolicyCodecV1 policy, not the full opponent pool.",
            "Full official card semantic coverage remains a separate mandatory gate.",
            "Transfer/synchronization counters are contract assertions; Nsight is still required.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

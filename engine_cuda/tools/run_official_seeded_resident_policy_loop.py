from __future__ import annotations

import argparse
import hashlib
import json
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
from run_official_seeded_reset_paired import read_deck  # noqa: E402


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description=(
            "Run a seeded, GPU-resident Official setup/Main/Battle loop with "
            "device terminal reset and a frozen BC policy."
        )
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--deck0",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "bc_models"
            / "archive"
            / "agent_pure_lucario_v1_bc512_e4_probe"
            / "deck.csv"
        ),
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "bc_models"
            / "agent_cynthia_core_meanpool_epochmix_v1_c7b3253f_20260727"
            / "deck.csv"
        ),
    )
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
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-select", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=2026080101)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "artifacts"
        / "official_seeded_resident_policy_loop.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def codec_batch(engine: Any) -> dict[str, Any]:
    encoded = dict(engine.encode_policy_v1())
    encoded["entity_mask"] = encoded["entity_mask"].bool()
    encoded["option_mask"] = encoded["option_mask"].bool()
    return encoded


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("batch/steps must be positive and warmup non-negative")
    if not 1 <= args.max_select <= 80:
        raise ValueError("max-select must be in [1, 80]")

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "reset_seeded_interactive_masked"):
        raise RuntimeError("_ptcg_cuda lacks masked interactive seeded reset")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)

    rules = args.rules.resolve()
    deck0 = args.deck0.resolve()
    deck1 = args.deck1.resolve()
    checkpoint = args.checkpoint.resolve()
    for path in (rules, deck0, deck1, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    deck_rows = [read_deck(deck0), read_deck(deck1)]
    decks = torch.tensor(
        [deck_rows for _ in range(args.batch)], dtype=torch.int32, device=device
    )
    seeds = torch.arange(
        args.seed_start,
        args.seed_start + args.batch,
        dtype=torch.int64,
        device=device,
    )
    lane_mask = torch.ones(args.batch, dtype=torch.bool, device=device)

    engine = create_official_engine(
        rules.read_bytes(), batch_size=args.batch, device_index=args.device_index
    )
    model, _payload = load_policy_checkpoint(checkpoint, device)
    model.eval()
    model.requires_grad_(False)
    adapter = EntityPointerPolicyV1DeviceAdapter(model, max_select=args.max_select)

    # Startup H2D/reset work is outside the measured loop. Every object used
    # below is a borrowed device tensor or a GPU-resident model parameter.
    engine.reset_seeded_interactive_masked(decks, seeds, lane_mask)
    decision_counter = torch.zeros((), dtype=torch.int64, device=device)
    episode_counter = torch.zeros((), dtype=torch.int64, device=device)
    overflow_counter = torch.zeros((), dtype=torch.int64, device=device)

    def one_resident_step(*, count_metrics: bool) -> None:
        terminal = engine.statuses().eq(TERMINAL)
        # Invoke the masked reset unconditionally.  Branching on
        # ``terminal.any()`` would materialize a CUDA boolean on the host and
        # silently synchronize every rollout step that reaches this line.
        seeds.add_(terminal.to(torch.int64) * args.batch)
        engine.reset_seeded_interactive_masked(decks, seeds, terminal)
        if count_metrics:
            episode_counter.add_(terminal.long().sum())

        # Advance only idle lanes; ready and terminal lanes are no-ops in the
        # device kernel. This also completes interactive setup automatically.
        engine.advance_to_decision()
        batch = codec_batch(engine)
        ready = engine.statuses().eq(NEEDS_ACTION)
        max_count = batch["max_count"].long()
        if count_metrics:
            overflow_counter.add_((ready & max_count.gt(args.max_select)).long().sum())
        batch["_route_mask"] = ready
        action_indices, action_lengths = adapter.act_device(batch)
        engine.pack_actions(action_indices, action_lengths)
        engine.apply_packed_actions()
        if count_metrics:
            decision_counter.add_(ready.long().sum())

    for _ in range(args.warmup):
        one_resident_step(count_metrics=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nvtx_range = "official_seeded_resident_setup_codec_bc_action_reset"
    torch.cuda.nvtx.range_push(nvtx_range)
    start.record()
    for _ in range(args.steps):
        one_resident_step(count_metrics=True)
    end.record()
    torch.cuda.nvtx.range_pop()
    end.synchronize()
    elapsed_sec = start.elapsed_time(end) / 1000.0

    decisions = int(decision_counter.item())
    episodes = int(episode_counter.item())
    overflows = int(overflow_counter.item())
    status_counts = torch.bincount(engine.statuses().long(), minlength=4).cpu().tolist()
    result = {
        "schema_version": 1,
        "passed": (
            decisions > 0
            and overflows == 0
            and status_counts[ERROR] == 0
        ),
        "scope": (
            "official seeded resident CUDA setup/Main/Battle loop with device "
            "terminal masked reset and frozen PolicyCodecV1 BC"
        ),
        "batch": args.batch,
        "warmup": args.warmup,
        "steps": args.steps,
        "max_select": args.max_select,
        "seed_start": args.seed_start,
        "decisions": decisions,
        "episodes_reset_on_device": episodes,
        "max_select_overflow_lanes": overflows,
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
            "device_seeded_masked_reset": True,
            "device_setup_codec": True,
            "device_action_apply": True,
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
        "deck_sha256": [sha256_file(deck0), sha256_file(deck1)],
        "rule_pack_sha256": sha256_file(rules),
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        "nvtx_hot_range": nvtx_range,
        "limitations": [
            "This local gate uses one frozen BC; the 10+ pool/learner router remains to be integrated.",
            "max_select overflow is fail-closed; the tested historical corpus has max action length 23.",
            "Compute Sanitizer/Nsight and 5090 performance gates require Linux/TCC server validation.",
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

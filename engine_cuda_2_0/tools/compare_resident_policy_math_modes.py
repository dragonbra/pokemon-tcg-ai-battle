from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))
sys.path.insert(0, str(WORKSPACE_ROOT / "tools"))

from ptcg_cuda_engine.legacy_codecs import policy_codec_v1_to_idonly_codec_v1  # noqa: E402
from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.policy_pool import GPUResidentPolicyPool  # noqa: E402
from run_official_seeded_multi_policy_loop import (  # noqa: E402
    ERROR,
    MIXED_PRECISION_PRESETS,
    NEEDS_ACTION,
    TERMINAL,
    balanced_seat_schedule,
    codec_batch,
    error_diagnostics,
    load_adapters,
    load_pool_rows,
    parse_policy_math_overrides,
)


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description=(
            "Compare two resident policy math modes on the exact same Official "
            "CUDA states. The baseline action is applied to advance the engine; "
            "candidate actions are diagnostic only."
        )
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--pool",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "configs"
        / "strongest_bc_opponent_pool_v1.json",
    )
    parser.add_argument("--batch", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--max-select", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=2026080101)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--baseline-math-mode",
        choices=("fp32", "tf32", "bf16", "fp16"),
        default="tf32",
    )
    parser.add_argument(
        "--candidate-math-mode",
        choices=("fp32", "tf32", "bf16", "fp16"),
        default="fp16",
    )
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument(
        "--baseline-policy-math-override",
        action="append",
        default=[],
        metavar="POLICY=MODE",
    )
    parser.add_argument(
        "--baseline-mixed-precision-preset",
        choices=tuple(sorted(MIXED_PRECISION_PRESETS)),
        default="none",
    )
    parser.add_argument(
        "--candidate-policy-math-override",
        action="append",
        default=[],
        metavar="POLICY=MODE",
    )
    parser.add_argument(
        "--candidate-mixed-precision-preset",
        choices=tuple(sorted(MIXED_PRECISION_PRESETS)),
        default="none",
    )
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "resident_policy_math_compare.json",
    )
    return parser.parse_args()


def set_math_mode(torch: Any, math_mode: str) -> None:
    if math_mode == "tf32":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def build_encoded_by_codec(
    policy_codec: dict[str, Any],
    *,
    needs_idonly: bool,
    idonly_entities: int,
    idonly_options: int,
) -> dict[str, dict[str, Any]]:
    encoded_by_codec: dict[str, dict[str, Any]] = {"policy_codec_v1": policy_codec}
    if needs_idonly:
        idonly_codec = policy_codec_v1_to_idonly_codec_v1(
            policy_codec,
            max_card_id=2048,
            max_action_steps=16,
            target_entity_capacity=idonly_entities,
            target_option_capacity=idonly_options,
        )
        encoded_by_codec["idonly_codec_v1"] = idonly_codec
        encoded_by_codec["marnie_prize_codec_v4"] = idonly_codec
    return encoded_by_codec


def policy_structure(policy: Any) -> tuple[Any, ...]:
    """Fields that must stay identical across math-mode comparisons.

    The dtype is intentionally excluded because it is the treatment variable
    being compared by this diagnostic.
    """

    return (
        policy.policy_id,
        policy.name,
        policy.deck,
        policy.checkpoint,
        policy.adapter,
        policy.codec,
        policy.frozen,
    )


def action_sequence(indices: Any, length: int) -> list[int]:
    return [int(value) for value in indices[:length].tolist()]


def equiv_sequence(option_equiv: Any, lane: int, sequence: list[int]) -> list[int]:
    width = int(option_equiv.shape[1])
    result: list[int] = []
    for option_index in sequence:
        if 0 <= option_index < width:
            result.append(int(option_equiv[lane, option_index].item()))
        else:
            result.append(-999999)
    return result


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("batch/steps must be positive and warmup non-negative")
    if not 1 <= args.max_select <= 80:
        raise ValueError("max-select must be in [1, 80]")

    import torch
    import _ptcg_cuda  # noqa: F401

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)

    rules = args.rules.resolve()
    pool_path = args.pool.resolve()
    for required in (rules, pool_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    rows, unsupported_rows = load_pool_rows(pool_path, args.policy)
    baseline_overrides = parse_policy_math_overrides(
        args.baseline_policy_math_override,
        preset=args.baseline_mixed_precision_preset,
    )
    candidate_overrides = parse_policy_math_overrides(
        args.candidate_policy_math_override,
        preset=args.candidate_mixed_precision_preset,
    )

    set_math_mode(torch, args.baseline_math_mode)
    baseline_manifest, baseline_adapters, registered_decks, baseline_report = load_adapters(
        rows,
        device,
        args.max_select,
        args.baseline_math_mode,
        policy_math_overrides=baseline_overrides,
    )
    policy_count = len(baseline_manifest.policies)
    schedule = balanced_seat_schedule(args.batch, policy_count)
    seat_counts = Counter(policy for pair in schedule for policy in pair)
    route_capacity = max(seat_counts.values())
    baseline_pool = GPUResidentPolicyPool(
        baseline_manifest,
        baseline_adapters,
        capacity=route_capacity,
        max_select=args.max_select,
    )

    set_math_mode(torch, args.candidate_math_mode)
    candidate_manifest, candidate_adapters, candidate_decks, candidate_report = load_adapters(
        rows,
        device,
        args.max_select,
        args.candidate_math_mode,
        policy_math_overrides=candidate_overrides,
    )
    if tuple(policy_structure(policy) for policy in candidate_manifest.policies) != tuple(
        policy_structure(policy) for policy in baseline_manifest.policies
    ):
        raise RuntimeError("baseline and candidate manifests differ")
    if candidate_decks != registered_decks:
        raise RuntimeError("baseline and candidate registered decks differ")
    candidate_pool = GPUResidentPolicyPool(
        candidate_manifest,
        candidate_adapters,
        capacity=route_capacity,
        max_select=args.max_select,
    )

    set_math_mode(torch, args.baseline_math_mode)
    seat_policy_ids = torch.tensor(schedule, dtype=torch.long, device=device)
    decks = torch.tensor(
        [[registered_decks[left], registered_decks[right]] for left, right in schedule],
        dtype=torch.int32,
        device=device,
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
    engine.reset_seeded_interactive_masked(decks, seeds, lane_mask)

    needs_idonly = bool(
        {"idonly_codec_v1", "marnie_prize_codec_v4"} & set(baseline_manifest.codecs)
    )
    idonly_entities = max(
        (
            row["max_entities"]
            for row in baseline_report
            if row["codec"] in {"idonly_codec_v1", "marnie_prize_codec_v4"}
        ),
        default=128,
    )
    idonly_options = max(
        (
            row["max_options"]
            for row in baseline_report
            if row["codec"] in {"idonly_codec_v1", "marnie_prize_codec_v4"}
        ),
        default=80,
    )

    decisions = 0
    compared_steps = 0
    raw_mismatches = 0
    equiv_mismatches = 0
    raw_mismatch_equiv_same = 0
    length_mismatches = 0
    episodes_reset = 0
    by_policy: dict[str, Counter[str]] = {
        policy.name: Counter() for policy in baseline_manifest.policies
    }
    examples: list[dict[str, Any]] = []

    total_steps = args.warmup + args.steps
    stopped_on_error = False
    with torch.inference_mode():
        for step in range(total_steps):
            terminal = engine.statuses().eq(TERMINAL)
            seeds.add_(terminal.to(torch.int64) * args.batch)
            engine.reset_seeded_interactive_masked(decks, seeds, terminal)
            if step >= args.warmup:
                episodes_reset += int(terminal.long().sum().item())
            engine.advance_to_decision()
            policy_codec = codec_batch(engine)
            ready = engine.statuses().eq(NEEDS_ACTION)
            actor = (policy_codec["global_cat"][:, 3].long() - 1).clamp(min=0, max=1)
            acting_policy_ids = seat_policy_ids.gather(1, actor.view(-1, 1)).view(-1)
            encoded_by_codec = build_encoded_by_codec(
                policy_codec,
                needs_idonly=needs_idonly,
                idonly_entities=idonly_entities,
                idonly_options=idonly_options,
            )

            set_math_mode(torch, args.baseline_math_mode)
            baseline_actions = baseline_pool.act(encoded_by_codec, acting_policy_ids, ready)
            set_math_mode(torch, args.candidate_math_mode)
            candidate_actions = candidate_pool.act(encoded_by_codec, acting_policy_ids, ready)
            set_math_mode(torch, args.baseline_math_mode)

            if step >= args.warmup:
                ready_cpu = ready.detach().cpu()
                acting_cpu = acting_policy_ids.detach().cpu()
                baseline_indices = baseline_actions.indices.detach().cpu()
                baseline_lengths = baseline_actions.lengths.detach().cpu()
                candidate_indices = candidate_actions.indices.detach().cpu()
                candidate_lengths = candidate_actions.lengths.detach().cpu()
                option_equiv = policy_codec.get("option_equiv")
                option_equiv_cpu = (
                    option_equiv.detach().cpu() if option_equiv is not None else None
                )
                for lane in ready_cpu.nonzero(as_tuple=False).flatten().tolist():
                    policy_id = int(acting_cpu[lane].item())
                    policy_name = baseline_manifest.policies[policy_id].name
                    stats = by_policy[policy_name]
                    stats["decisions"] += 1
                    decisions += 1
                    base_len = int(baseline_lengths[lane].item())
                    cand_len = int(candidate_lengths[lane].item())
                    base_seq = action_sequence(baseline_indices[lane], base_len)
                    cand_seq = action_sequence(candidate_indices[lane], cand_len)
                    raw_equal = base_len == cand_len and base_seq == cand_seq
                    if raw_equal:
                        stats["raw_equal"] += 1
                        stats["equiv_equal"] += 1
                        continue
                    raw_mismatches += 1
                    stats["raw_mismatch"] += 1
                    if base_len != cand_len:
                        length_mismatches += 1
                        stats["length_mismatch"] += 1
                    if option_equiv_cpu is not None and base_len == cand_len:
                        base_equiv = equiv_sequence(option_equiv_cpu, lane, base_seq)
                        cand_equiv = equiv_sequence(option_equiv_cpu, lane, cand_seq)
                        equiv_equal = base_equiv == cand_equiv
                    else:
                        base_equiv = []
                        cand_equiv = []
                        equiv_equal = False
                    if equiv_equal:
                        raw_mismatch_equiv_same += 1
                        stats["equiv_equal"] += 1
                        stats["raw_mismatch_equiv_same"] += 1
                    else:
                        equiv_mismatches += 1
                        stats["equiv_mismatch"] += 1
                    if len(examples) < args.max_examples:
                        examples.append(
                            {
                                "step": step,
                                "lane": int(lane),
                                "policy_id": policy_id,
                                "policy_name": policy_name,
                                "baseline_length": base_len,
                                "candidate_length": cand_len,
                                "baseline_indices": base_seq,
                                "candidate_indices": cand_seq,
                                "baseline_option_equiv": base_equiv,
                                "candidate_option_equiv": cand_equiv,
                                "raw_equal": raw_equal,
                                "option_equiv_equal": equiv_equal,
                            }
                        )
                compared_steps += 1

            engine.pack_actions(baseline_actions.indices, baseline_actions.lengths)
            engine.apply_packed_actions()
            if bool(engine.statuses().eq(ERROR).any().item()):
                stopped_on_error = True
                break

    status_counts = torch.bincount(engine.statuses().long(), minlength=4).cpu().tolist()
    diagnostics = error_diagnostics(engine, schedule, baseline_manifest.policies)
    result = {
        "schema_version": 1,
        "passed": (
            decisions > 0
            and equiv_mismatches == 0
            and int(status_counts[ERROR]) == 0
        ),
        "scope": "same-state resident policy math-mode action comparison",
        "baseline_math_mode": args.baseline_math_mode,
        "candidate_math_mode": args.candidate_math_mode,
        "baseline_mixed_precision_preset": args.baseline_mixed_precision_preset,
        "candidate_mixed_precision_preset": args.candidate_mixed_precision_preset,
        "baseline_policy_math_overrides": baseline_overrides,
        "candidate_policy_math_overrides": candidate_overrides,
        "batch": args.batch,
        "warmup": args.warmup,
        "steps": args.steps,
        "compared_steps": compared_steps,
        "policies_loaded": policy_count,
        "route_capacity": route_capacity,
        "decisions": decisions,
        "episodes_reset_on_device": episodes_reset,
        "raw_mismatches": raw_mismatches,
        "raw_mismatch_rate": raw_mismatches / decisions if decisions else 0.0,
        "raw_mismatch_option_equiv_same": raw_mismatch_equiv_same,
        "equiv_mismatches": equiv_mismatches,
        "equiv_mismatch_rate": equiv_mismatches / decisions if decisions else 0.0,
        "length_mismatches": length_mismatches,
        "by_policy": {
            name: {key: int(value) for key, value in stats.items()}
            for name, stats in by_policy.items()
        },
        "examples": examples,
        "status_counts": {
            "idle": int(status_counts[0]),
            "needs_action": int(status_counts[1]),
            "terminal": int(status_counts[2]),
            "error": int(status_counts[3]),
        },
        "diagnostics": diagnostics,
        "stopped_on_error": stopped_on_error,
        "policies": baseline_report,
        "candidate_policies": candidate_report,
        "unsupported_pool_entries": [
            {
                "name": str(row.get("name")),
                "codec": str(row.get("codec")),
                "reason": "resident device codec is not implemented yet",
            }
            for row in unsupported_rows
        ],
        "hot_path_note": (
            "This is a diagnostic semantic gate and intentionally performs "
            "host reads/synchronization for comparison. It is not a rollout "
            "throughput benchmark."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**result, "output": str(args.output)}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

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
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from run_official_seeded_reset_paired import (  # noqa: E402
    STATE_BYTES,
    first_mismatch,
    read_deck,
    read_fixture,
    tensor_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every seeded-reset matrix case in one CUDA/PyTorch process."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("seeded-first-min", "interactive-device"),
        default="seeded-first-min",
        help=(
            "Use the fused seeded reset or replay every setup choice through "
            "the CUDA codec/action path."
        ),
    )
    parser.add_argument(
        "--setup-check-interval",
        type=int,
        default=16,
        help="Test-only host completion check interval for interactive setup.",
    )
    parser.add_argument(
        "--max-setup-actions",
        type=int,
        default=128,
        help="Maximum device action steps before interactive setup fails.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"matrix path escapes workspace: {relative}")
    return path


def main() -> None:
    args = parse_args()
    if args.setup_check_interval <= 0:
        raise SystemExit("--setup-check-interval must be positive")
    if args.max_setup_actions <= 0:
        raise SystemExit("--max-setup-actions must be positive")
    manifest_path = args.manifest.resolve()
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("seeded-reset matrix manifest contains no cases")
    rules = workspace_path(str(manifest["rules"]))

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if str(_ptcg_cuda.OFFICIAL_SEEDED_SETUP_POLICY) != "first_min_v1":
        raise RuntimeError("official CUDA extension seeded setup policy mismatch")
    if (
        args.mode == "interactive-device"
        and str(_ptcg_cuda.OFFICIAL_INTERACTIVE_SETUP_POLICY) != "device_action_v1"
    ):
        raise RuntimeError("official CUDA extension interactive setup policy mismatch")

    first_fixture = read_fixture(workspace_path(str(cases[0]["fixture"])))
    batch = len(first_fixture.seeds)
    device = torch.device("cuda", args.device_index)
    engine = create_official_engine(
        rules.read_bytes(), batch_size=batch, device_index=args.device_index
    )
    case_results: list[dict[str, Any]] = []
    total_state_mismatch_lanes = 0
    total_status_mismatch_lanes = 0
    total_dtype_mismatch_lanes = 0
    total_setup_codec_decisions = 0
    total_setup_kernel_steps = 0
    total_completion_checks = 0
    max_observed_setup_actions = 0
    select_context_offset = int(
        getattr(_ptcg_cuda, "OFFICIAL_SELECT_CONTEXT_OFFSET", 2594)
    )
    main_context = 1
    option_indices = torch.arange(
        80, dtype=torch.int64, device=device
    ).expand(batch, -1).contiguous()

    def replay_interactive_setup(decks: Any, seeds: Any) -> dict[str, Any]:
        engine.reset_seeded_interactive(decks, seeds)
        action_counts = torch.zeros(batch, dtype=torch.int64, device=device)
        invalid_option_lanes = torch.zeros(batch, dtype=torch.bool, device=device)
        kernel_steps = 0
        completion_checks = 0
        remaining_setup_lanes = batch
        while kernel_steps < args.max_setup_actions:
            setup_mask = engine.state_bytes()[:, select_context_offset].ne(main_context)
            action_counts.add_(setup_mask)
            codec = engine.encode_policy_v1()
            option_counts = codec["option_mask"].sum(dim=1)
            invalid_option_lanes.logical_or_(setup_mask.logical_and(option_counts.le(0)))
            engine.pack_actions(option_indices, codec["min_count"])
            engine.apply_packed_setup_actions()
            kernel_steps += 1
            if (
                kernel_steps % args.setup_check_interval == 0
                or kernel_steps == args.max_setup_actions
            ):
                completion_checks += 1
                remaining_setup_lanes = int(
                    engine.state_bytes()[:, select_context_offset]
                    .ne(main_context)
                    .sum()
                    .item()
                )
                if remaining_setup_lanes == 0:
                    break
        return {
            "state": tensor_bytes(engine.state_bytes()),
            "statuses": [
                int(value) for value in engine.statuses().detach().cpu().tolist()
            ],
            "action_counts": [
                int(value) for value in action_counts.detach().cpu().tolist()
            ],
            "invalid_option_lanes": int(invalid_option_lanes.sum().item()),
            "remaining_setup_lanes": remaining_setup_lanes,
            "kernel_steps": kernel_steps,
            "completion_checks": completion_checks,
        }

    for case in cases:
        fixture_path = workspace_path(str(case["fixture"]))
        deck_paths = [workspace_path(str(case["deck0"])), workspace_path(str(case["deck1"]))]
        fixture = read_fixture(fixture_path)
        if len(fixture.seeds) != batch:
            raise ValueError("all matrix fixtures must have the same record count")
        deck_rows = [read_deck(path) for path in deck_paths]
        decks_i32 = torch.tensor(
            [deck_rows for _ in range(batch)], dtype=torch.int32, device=device
        )
        seeds = torch.tensor(fixture.seeds, dtype=torch.int64, device=device)

        interactive_i32: dict[str, Any] | None = None
        interactive_i64: dict[str, Any] | None = None
        if args.mode == "interactive-device":
            interactive_i32 = replay_interactive_setup(decks_i32, seeds)
            interactive_i64 = replay_interactive_setup(decks_i32.to(torch.int64), seeds)
            state_i32 = interactive_i32["state"]
            status_i32 = interactive_i32["statuses"]
            state_i64 = interactive_i64["state"]
            status_i64 = interactive_i64["statuses"]
        else:
            engine.reset_seeded_first_min(decks_i32, seeds)
            state_i32 = tensor_bytes(engine.state_bytes())
            status_i32 = [
                int(value) for value in engine.statuses().detach().cpu().tolist()
            ]
            engine.reset_seeded_first_min(decks_i32.to(torch.int64), seeds)
            state_i64 = tensor_bytes(engine.state_bytes())
            status_i64 = [
                int(value) for value in engine.statuses().detach().cpu().tolist()
            ]

        state_mismatch_lanes = sum(
            fixture.states[index * STATE_BYTES : (index + 1) * STATE_BYTES]
            != state_i32[index * STATE_BYTES : (index + 1) * STATE_BYTES]
            for index in range(batch)
        )
        status_mismatch_lanes = sum(
            expected != actual
            for expected, actual in zip(fixture.statuses, status_i32, strict=True)
        )
        dtype_mismatch_lanes = sum(
            state_i32[index * STATE_BYTES : (index + 1) * STATE_BYTES]
            != state_i64[index * STATE_BYTES : (index + 1) * STATE_BYTES]
            for index in range(batch)
        )
        dtype_status_mismatch_lanes = sum(
            left != right for left, right in zip(status_i32, status_i64, strict=True)
        )
        action_count_mismatch_lanes = 0
        invalid_option_lanes = 0
        remaining_setup_lanes = 0
        setup_codec_decisions = 0
        setup_kernel_steps = 0
        completion_checks = 0
        min_setup_actions = 0
        max_setup_actions = 0
        if interactive_i32 is not None and interactive_i64 is not None:
            action_count_mismatch_lanes = sum(
                left != right
                for left, right in zip(
                    interactive_i32["action_counts"],
                    interactive_i64["action_counts"],
                    strict=True,
                )
            )
            invalid_option_lanes = (
                interactive_i32["invalid_option_lanes"]
                + interactive_i64["invalid_option_lanes"]
            )
            remaining_setup_lanes = (
                interactive_i32["remaining_setup_lanes"]
                + interactive_i64["remaining_setup_lanes"]
            )
            setup_codec_decisions = sum(interactive_i32["action_counts"])
            setup_kernel_steps = (
                interactive_i32["kernel_steps"] + interactive_i64["kernel_steps"]
            )
            completion_checks = (
                interactive_i32["completion_checks"]
                + interactive_i64["completion_checks"]
            )
            min_setup_actions = min(interactive_i32["action_counts"])
            max_setup_actions = max(interactive_i32["action_counts"])
            total_setup_codec_decisions += setup_codec_decisions
            total_setup_kernel_steps += setup_kernel_steps
            total_completion_checks += completion_checks
            max_observed_setup_actions = max(
                max_observed_setup_actions, max_setup_actions
            )
        total_state_mismatch_lanes += state_mismatch_lanes
        total_status_mismatch_lanes += status_mismatch_lanes
        total_dtype_mismatch_lanes += (
            dtype_mismatch_lanes
            + dtype_status_mismatch_lanes
            + action_count_mismatch_lanes
        )
        case_passed = (
            state_mismatch_lanes == 0
            and status_mismatch_lanes == 0
            and dtype_mismatch_lanes == 0
            and dtype_status_mismatch_lanes == 0
            and action_count_mismatch_lanes == 0
            and invalid_option_lanes == 0
            and remaining_setup_lanes == 0
        )
        case_results.append(
            {
                "name": str(case["name"]),
                "deck0_name": str(case["deck0_name"]),
                "deck1_name": str(case["deck1_name"]),
                "passed": case_passed,
                "records": batch,
                "state_mismatch_lanes": state_mismatch_lanes,
                "status_mismatch_lanes": status_mismatch_lanes,
                "int32_int64_state_mismatch_lanes": dtype_mismatch_lanes,
                "int32_int64_status_mismatch_lanes": dtype_status_mismatch_lanes,
                "int32_int64_action_count_mismatch_lanes": (
                    action_count_mismatch_lanes
                ),
                "invalid_setup_option_lanes": invalid_option_lanes,
                "remaining_setup_lanes": remaining_setup_lanes,
                "setup_codec_decisions": setup_codec_decisions,
                "setup_kernel_steps_both_dtypes": setup_kernel_steps,
                "test_only_completion_checks_both_dtypes": completion_checks,
                "min_setup_actions": min_setup_actions,
                "max_setup_actions": max_setup_actions,
                "first_fixture_state_mismatch": first_mismatch(
                    fixture.states, state_i32
                ),
                "first_dtype_state_mismatch": first_mismatch(state_i32, state_i64),
                "deck0_sha256": sha256_file(deck_paths[0]),
                "deck1_sha256": sha256_file(deck_paths[1]),
                "private_fixture_sha256": sha256_file(fixture_path),
            }
        )

    passed = (
        total_state_mismatch_lanes == 0
        and total_status_mismatch_lanes == 0
        and total_dtype_mismatch_lanes == 0
        and all(case["passed"] for case in case_results)
    )
    result = {
        "passed": passed,
        "contract": (
            "official_interactive_setup_cuda_ordered_matrix_v1"
            if args.mode == "interactive-device"
            else "official_seeded_reset_cuda_ordered_matrix_v1"
        ),
        "mode": args.mode,
        "setup_policy": (
            "first_min_v1_via_device_actions"
            if args.mode == "interactive-device"
            else str(_ptcg_cuda.OFFICIAL_SEEDED_SETUP_POLICY)
        ),
        "case_count": len(case_results),
        "records_per_case": batch,
        "records_compared": len(case_results) * batch,
        "official_cuda_state_bytes_compared": len(case_results) * batch * STATE_BYTES,
        "int32_int64_state_bytes_compared": len(case_results) * batch * STATE_BYTES,
        "state_mismatch_lanes": total_state_mismatch_lanes,
        "status_mismatch_lanes": total_status_mismatch_lanes,
        "dtype_mismatch_lanes": total_dtype_mismatch_lanes,
        "setup_codec_decisions": total_setup_codec_decisions,
        "setup_kernel_steps_both_dtypes": total_setup_kernel_steps,
        "test_only_completion_checks_both_dtypes": total_completion_checks,
        "max_observed_setup_actions": max_observed_setup_actions,
        "select_context_offset": select_context_offset,
        "state_bytes": STATE_BYTES,
        "arena_allocated_bytes": int(engine.allocated_bytes),
        "device": torch.cuda.get_device_name(args.device_index),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "rule_pack_sha256": sha256_file(rules),
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        "cases": case_results,
    }
    if args.mode == "interactive-device":
        result["device_action_contract"] = (
            "CUDA codec, first-min option indices, counts, action packing, and "
            "setup state transitions remain device-resident; completion/status/"
            "state reads are test-only. Main lanes are protected by the "
            "setup-only device apply gate."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

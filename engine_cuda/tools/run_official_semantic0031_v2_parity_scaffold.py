from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
DEFAULT_MODEL_PACKAGE = Path(
    "/bc_models/0031_pt0805_buneary_lopunny_froslass_compact_fp16"
)
DEFAULT_CHECKPOINT = Path("/bc_models/semantic0031_0806_shared_prototype_fp32.pt")
NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3

sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.semantic0031_bridge import (  # noqa: E402
    Semantic0031DeviceAdapter,
    load_semantic0031_package,
    semantic0031_v2_ready_batch,
)
from run_official_seeded_reset_paired import read_deck  # noqa: E402


GLOBAL_KEYS = ("global_cat", "global_num", "global_state", "min_count", "max_count")
FAMILY_KEYS = {
    "card": ("card_cat", "card_num", "card_state", "card_parent"),
    "resource": ("resource_cat", "resource_num", "resource_state"),
    "event": (
        "event_cat",
        "event_num",
        "event_state",
        "event_source",
        "event_target",
        "event_before",
        "event_after",
    ),
    "option": (
        "option_cat",
        "option_num",
        "option_state",
        "option_source",
        "option_target",
        "option_context",
        "option_effect_card",
    ),
}


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-seed/action official CPU CausalKnowledge vs CUDA semantic0031 v2 "
            "parity, then compare shared-model logits and greedy actions."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=WORKSPACE_ROOT / "engine" / "source" / "ptcgProgram 22",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=private / "0022_deck40_focal_s1_1_n2" / "manifest.json",
    )
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--package", type=Path, default=DEFAULT_MODEL_PACKAGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=CUDA_ENGINE_ROOT / "build" / "torch_0031_linux",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=2026080601)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--decision-limit", type=int, default=256)
    parser.add_argument("--compare-decisions", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=("coverage-first-legal", "coverage-random-legal"),
        default="coverage-random-legal",
    )
    parser.add_argument("--skip-trace-build", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--model-chunk", type=int, default=2)
    parser.add_argument("--logits-atol", type=float, default=2e-4)
    parser.add_argument("--logits-rtol", type=float, default=2e-4)
    parser.add_argument("--max-mismatch-details", type=int, default=32)
    parser.add_argument(
        "--require-history-wrap",
        action="store_true",
        help="Fail the strict gate unless a compared decision observes more than 64 events.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_semantic0031_v2_trace.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_semantic0031_v2_parity.json",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return path


def build_and_run_trace(
    args: argparse.Namespace,
    rules: Path,
    deck0: Path,
    deck1: Path,
) -> dict[str, Any]:
    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    build_dir = CUDA_ENGINE_ROOT / "build" / "official_semantic0031_trace_linux"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_semantic0031_trace"
    if not args.skip_trace_build:
        subprocess.run(
            [
                "g++",
                "-std=c++20",
                "-O2",
                f"-I{source}",
                f"-I{CUDA_ENGINE_ROOT / 'include'}",
                f"-I{CUDA_ENGINE_ROOT / 'extractor'}",
                str(CUDA_ENGINE_ROOT / "extractor" / "official_battle_end_turn_paired.cpp"),
                "-o",
                str(executable),
            ],
            check=True,
        )
    if not executable.is_file():
        raise FileNotFoundError(executable)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(executable),
            "--rules",
            str(rules),
            "--deck0",
            str(deck0),
            "--deck1",
            str(deck1),
            "--seed-start",
            str(args.seed_start),
            "--seed-count",
            str(args.seed_count),
            "--decision-limit",
            str(args.decision_limit),
            "--policy",
            args.policy,
            "--trace-jsonl",
            str(args.trace),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def load_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"trace line {line_number} is not an object")
            observation = value.get("actor_observation")
            action = value.get("ordered_action")
            if not isinstance(observation, dict) or not isinstance(action, list):
                raise ValueError(f"trace line {line_number} lacks observation/action")
            records.append(value)
    if not records:
        raise ValueError("official semantic trace is empty")
    return records


def import_cpu_semantic(package_root: Path) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(package_root.resolve()))
    compiler = importlib.import_module("semantic0031.features.compiler")
    collate = importlib.import_module("semantic0031.features.collate")
    knowledge = importlib.import_module("semantic0031.knowledge.state")
    prototypes = importlib.import_module("semantic0031.domain.prototypes")
    prototype_index = prototypes.PrototypeIndex.load(
        package_root / "semantic0031/assets/official_public_prototypes_v1.json",
        package_root / "semantic0031/assets/official_full_engine_prototypes_v2.json",
    )
    return (
        compiler.compile_canonical_row,
        collate.collate_canonical_records,
        knowledge.CausalKnowledge,
        prototype_index,
    )


def deck_manifest(deck: Sequence[int]) -> dict[str, Any]:
    counts = Counter(int(card) for card in deck)
    return {"counts": [[card, count] for card, count in sorted(counts.items())]}


def mismatch_detail(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    import torch

    expected_tensor = torch.as_tensor(expected).detach().cpu()
    actual_tensor = torch.as_tensor(actual).detach().cpu()
    detail: dict[str, Any] = {
        "field": name,
        "expected_shape": list(expected_tensor.shape),
        "actual_shape": list(actual_tensor.shape),
    }
    if expected_tensor.shape != actual_tensor.shape:
        return detail
    unequal = expected_tensor.ne(actual_tensor)
    if bool(unequal.any()):
        flat_index = int(unequal.reshape(-1).nonzero(as_tuple=False)[0].item())
        detail["flat_index"] = flat_index
        detail["expected"] = expected_tensor.reshape(-1)[flat_index].item()
        detail["actual"] = actual_tensor.reshape(-1)[flat_index].item()
        if expected_tensor.ndim == 2 and expected_tensor.shape[1] > 0:
            row = flat_index // int(expected_tensor.shape[1])
            detail["row"] = row
            detail["expected_row"] = expected_tensor[row].tolist()
            detail["actual_row"] = actual_tensor[row].tolist()
    return detail


def compare_compiled_to_cuda(
    cpu_batch: Mapping[str, Any],
    cuda_batch: Mapping[str, Any],
    *,
    seed: int,
    decision: int,
    actor: int,
    mismatch_counts: Counter[str],
    mismatch_details: list[dict[str, Any]],
    max_details: int,
) -> None:
    import torch

    def record(name: str, expected: Any, actual: Any) -> None:
        mismatch_counts[name] += 1
        if len(mismatch_details) < max_details:
            mismatch_details.append(
                {
                    "seed": seed,
                    "decision": decision,
                    "actor": actor,
                    **mismatch_detail(name, expected, actual),
                }
            )

    for name in GLOBAL_KEYS:
        expected = cpu_batch[name][0]
        actual = cuda_batch[name][0]
        if expected.dtype == torch.bool:
            actual = actual.bool()
        else:
            actual = actual.to(expected.dtype)
        if expected.shape != actual.shape or not torch.equal(expected, actual):
            record(name, expected, actual)

    for family, names in FAMILY_KEYS.items():
        mask_name = f"{family}_mask"
        expected_mask = cpu_batch[mask_name][0].bool()
        actual_mask = cuda_batch[mask_name][0].bool()
        live = int(expected_mask.sum().item())
        canonical_actual_mask = torch.zeros_like(actual_mask)
        canonical_actual_mask[:live] = True
        if actual_mask.shape != canonical_actual_mask.shape or not torch.equal(
            actual_mask, canonical_actual_mask
        ):
            record(mask_name, canonical_actual_mask, actual_mask)
        if live > actual_mask.numel():
            continue
        for name in names:
            expected = cpu_batch[name][0, :live]
            actual = cuda_batch[name][0, :live].to(expected.dtype)
            if expected.shape != actual.shape or not torch.equal(expected, actual):
                record(name, expected, actual)
            padding = cuda_batch[name][0, live:]
            if padding.numel() and bool(padding.ne(0).any()):
                record(f"{name}.padding", torch.zeros_like(padding), padding)


def extract_greedy(result: Any, row: int) -> list[int]:
    length = int(result.lengths[row].detach().cpu().item())
    return [
        int(value)
        for value in result.sequences[row, :length].detach().cpu().tolist()
    ]


def compare_model_chunks(
    package: Any,
    adapter: Semantic0031DeviceAdapter,
    collate_records: Any,
    cpu_records: Sequence[Mapping[str, Any]],
    cuda_records: Sequence[Mapping[str, Any]],
    option_counts: Sequence[int],
    *,
    chunk_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import torch

    model = package.model
    device = adapter.device
    greedy_divergences = 0
    logits_failures = 0
    maximum_absolute_error = 0.0
    decisions = 0
    first_failure: dict[str, Any] | None = None
    with torch.inference_mode():
        for start in range(0, len(cpu_records), chunk_size):
            stop = min(len(cpu_records), start + chunk_size)
            cpu_mapping = collate_records(cpu_records[start:stop])
            cpu_mapping = {
                name: tensor.to(device)
                for name, tensor in cpu_mapping.items()
            }
            cuda_mapping = {
                name: torch.cat(
                    [record[name] for record in cuda_records[start:stop]], dim=0
                ).to(device)
                for name in cuda_records[start]
            }
            cuda_mapping = semantic0031_v2_ready_batch(cuda_mapping)

            cpu_batch = model.validate_batch(cpu_mapping)
            cuda_batch = model.validate_batch(cuda_mapping)
            cpu_state = model.state_encoder(cpu_batch, adapter.prototype_memory)
            cuda_state = model.state_encoder(cuda_batch, adapter.prototype_memory)
            cpu_options = model.option_encoder(
                cpu_batch, cpu_state, adapter.prototype_memory
            )
            cuda_options = adapter._encode_options(cuda_batch, cuda_state)
            cpu_decoder = model.action_decoder.initialize(cpu_batch, cpu_state.summary)
            cuda_decoder = model.action_decoder.initialize(cuda_batch, cuda_state.summary)
            cpu_logits = model.action_decoder.logits(cpu_batch, cpu_options, cpu_decoder)
            cuda_logits = model.action_decoder.logits(cuda_batch, cuda_options, cuda_decoder)
            cpu_greedy = model.action_decoder.greedy(
                cpu_batch, cpu_options, cpu_state.summary
            )
            cuda_greedy = model.action_decoder.greedy(
                cuda_batch, cuda_options, cuda_state.summary
            )
            for row, option_count in enumerate(option_counts[start:stop]):
                expected_action = extract_greedy(cpu_greedy, row)
                actual_action = extract_greedy(cuda_greedy, row)
                if expected_action != actual_action:
                    greedy_divergences += 1
                    if first_failure is None:
                        first_failure = {
                            "decision": start + row,
                            "cpu_greedy": expected_action,
                            "cuda_greedy": actual_action,
                        }
                expected_logits = torch.cat(
                    (cpu_logits[row, :option_count], cpu_logits[row, -1:])
                )
                actual_logits = torch.cat(
                    (cuda_logits[row, :option_count], cuda_logits[row, -1:])
                )
                finite = torch.isfinite(expected_logits) & torch.isfinite(actual_logits)
                if bool(finite.any()):
                    error = float(
                        (expected_logits[finite] - actual_logits[finite])
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                        .item()
                    )
                    maximum_absolute_error = max(maximum_absolute_error, error)
                close = torch.isclose(
                    expected_logits,
                    actual_logits,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=False,
                )
                if not bool(close.all()):
                    logits_failures += 1
                    if first_failure is None:
                        first_failure = {
                            "decision": start + row,
                            "logits_max_abs": maximum_absolute_error,
                        }
                decisions += 1
    return {
        "decisions": decisions,
        "logits_atol": atol,
        "logits_rtol": rtol,
        "max_logits_absolute_error": maximum_absolute_error,
        "logits_tolerance_failures": logits_failures,
        "greedy_action_divergences": greedy_divergences,
        "first_failure": first_failure,
        "passed": logits_failures == 0 and greedy_divergences == 0,
    }


def main() -> int:
    args = parse_args()
    if args.seed_count <= 0 or args.decision_limit <= 0:
        raise ValueError("seed-count and decision-limit must be positive")
    if args.compare_decisions < 0 or args.model_chunk <= 0:
        raise ValueError("compare-decisions cannot be negative and model-chunk must be positive")
    if args.max_mismatch_details <= 0:
        raise ValueError("max-mismatch-details must be positive")

    sys.path.insert(0, str(args.extension_dir.resolve()))
    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "encode_semantic0031_v2_lanes"):
        raise RuntimeError("_ptcg_cuda lacks encode_semantic0031_v2_lanes")
    if not args.package.is_dir():
        raise FileNotFoundError(args.package)
    if not args.skip_model and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    manifest: dict[str, Any] = json.loads(
        args.manifest.resolve().read_text(encoding="utf-8")
    )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 0 <= args.case_index < len(cases):
        raise ValueError("case-index is outside the manifest")
    case = cases[args.case_index]
    rules = workspace_path(str(manifest["rules"]))
    deck_paths = (
        workspace_path(str(case["deck0"])),
        workspace_path(str(case["deck1"])),
    )
    deck_rows = [read_deck(path) for path in deck_paths]

    trace_summary = build_and_run_trace(args, rules, *deck_paths)
    trace = load_trace(args.trace)
    if trace_summary.get("semantic_trace_records") != len(trace):
        raise RuntimeError("official trace record count disagrees with paired replay")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in trace:
        grouped[int(record["seed"])].append(record)

    compile_row, collate_records, causal_knowledge, cpu_prototypes = import_cpu_semantic(
        args.package
    )
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    engine = create_official_engine(
        rules.read_bytes(), batch_size=1, device_index=args.device_index
    )
    lanes = torch.zeros(1, dtype=torch.int32, device=device)
    decks_tensor = torch.tensor([deck_rows], dtype=torch.int32, device=device)
    manifests = [deck_manifest(deck) for deck in deck_rows]
    mismatch_counts: Counter[str] = Counter()
    mismatch_details: list[dict[str, Any]] = []
    compiled_records: list[Mapping[str, Any]] = []
    encoded_records: list[Mapping[str, Any]] = []
    option_counts: list[int] = []
    replayed = 0
    state_errors = 0
    max_history_total_count = 0
    wrapped_history_decisions = 0
    first_wrapped_history_decision: dict[str, int] | None = None
    requested = args.compare_decisions if args.compare_decisions > 0 else len(trace)

    for seed in sorted(grouped):
        if replayed >= requested:
            break
        records = sorted(grouped[seed], key=lambda item: int(item["decision"]))
        engine.reset_seeded_first_min_semantic(
            decks_tensor, torch.tensor([seed], dtype=torch.int64, device=device)
        )
        knowledge_by_actor = {
            actor: causal_knowledge(actor, deck_rows[actor]) for actor in (0, 1)
        }
        for record in records:
            if replayed >= requested:
                break
            status = int(engine.statuses()[0].detach().cpu().item())
            if status != NEEDS_ACTION:
                state_errors += 1
                break
            observation = record["actor_observation"]
            actor = int(observation["current"]["yourIndex"])
            action = [int(value) for value in record["ordered_action"]]
            snapshot = knowledge_by_actor[actor].consume(observation)
            canonical = compile_row(
                {
                    "actor_observation": observation,
                    "deck_manifest": manifests[actor],
                    "ordered_action": action,
                    "action_termination": None,
                },
                snapshot,
                cpu_prototypes,
            )
            encoded = engine.encode_semantic0031_v2_lanes(lanes)
            if canonical is None:
                raise AssertionError("canonical compiler returned no record")
            cpu_batch = collate_records([canonical])
            encoded_cpu = {
                name: tensor.detach().cpu() for name, tensor in encoded.items()
            }
            raw_history_device = engine.semantic_history_raw()
            history_total_count = int(raw_history_device["total_count"][0].item())
            max_history_total_count = max(max_history_total_count, history_total_count)
            if history_total_count > 64:
                wrapped_history_decisions += 1
                if first_wrapped_history_decision is None:
                    first_wrapped_history_decision = {
                        "seed": seed,
                        "decision": int(record["decision"]),
                        "total_count": history_total_count,
                    }
            detail_start = len(mismatch_details)
            compare_compiled_to_cuda(
                cpu_batch,
                encoded_cpu,
                seed=seed,
                decision=int(record["decision"]),
                actor=actor,
                mismatch_counts=mismatch_counts,
                mismatch_details=mismatch_details,
                max_details=args.max_mismatch_details,
            )
            if len(mismatch_details) > detail_start:
                raw_history = {
                    name: tensor.detach().cpu()
                    for name, tensor in raw_history_device.items()
                }
                history_debug = {
                    "known_serials": raw_history["known_opponent_hand"][
                        0, actor
                    ].nonzero(as_tuple=False).flatten().tolist(),
                    "possible_serials": raw_history["possible_opponent_hand"][
                        0, actor
                    ].nonzero(as_tuple=False).flatten().tolist(),
                    "remembered_serials": raw_history["remembered_opponent_cards"][
                        0, actor
                    ].nonzero(as_tuple=False).flatten().tolist(),
                    "unknown_count": int(
                        raw_history["unknown_opponent_hand"][0, actor].item()
                    ),
                    "possible_lower": int(
                        raw_history["possible_hand_lower"][0, actor].item()
                    ),
                    "possible_upper": int(
                        raw_history["possible_hand_upper"][0, actor].item()
                    ),
                    "recent_raw_events": [
                        {
                            "type": int(raw_history["log_type"][0, slot % 64].item()),
                            "params": raw_history["params"][0, slot % 64].tolist(),
                        }
                        for slot in range(
                            int(raw_history["total_count"][0].item())
                            - min(12, int(raw_history["total_count"][0].item())),
                            int(raw_history["total_count"][0].item()),
                        )
                    ],
                }
                for detail in mismatch_details[detail_start:]:
                    detail["cuda_semantic_history"] = history_debug
            compiled_records.append(canonical)
            encoded_records.append(encoded_cpu)
            option_counts.append(len(canonical["actor"]["option_cat"]))
            width = max(1, len(action))
            action_tensor = torch.full(
                (1, width), -1, dtype=torch.int64, device=device
            )
            if action:
                action_tensor[0, : len(action)] = torch.tensor(
                    action, dtype=torch.int64, device=device
                )
            counts = torch.tensor([len(action)], dtype=torch.int64, device=device)
            engine.pack_actions(action_tensor, counts)
            engine.apply_packed_actions()
            engine.advance_to_decision()
            replayed += 1
            if int(engine.statuses()[0].detach().cpu().item()) == ERROR:
                state_errors += 1
                break

    package = None
    model_report: dict[str, Any] = {"skipped": True, "passed": False}
    if not args.skip_model and not mismatch_counts and state_errors == 0:
        package = load_semantic0031_package(
            args.package,
            device=device,
            trusted_directory=True,
            checkpoint_override=args.checkpoint,
        )
        try:
            adapter = Semantic0031DeviceAdapter(package.model, package.deck)
            model_report = compare_model_chunks(
                package,
                adapter,
                collate_records,
                compiled_records,
                encoded_records,
                option_counts,
                chunk_size=args.model_chunk,
                atol=args.logits_atol,
                rtol=args.logits_rtol,
            )
        finally:
            package.close()

    tensor_passed = not mismatch_counts and state_errors == 0 and replayed == requested
    coverage_complete = requested == len(trace) and replayed == len(trace)
    history_wrap_passed = not args.require_history_wrap or wrapped_history_decisions > 0
    full_passed = (
        tensor_passed
        and coverage_complete
        and history_wrap_passed
        and not args.skip_model
        and bool(model_report.get("passed"))
    )
    report = {
        "schema_version": 2,
        "case": case.get("name"),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "trace_records": len(trace),
        "decisions_requested": requested,
        "decisions_replayed": replayed,
        "full_trace_coverage": coverage_complete,
        "fixed_action_cuda_state_errors": state_errors,
        "history_wrap": {
            "required": args.require_history_wrap,
            "capacity": 64,
            "max_total_count": max_history_total_count,
            "wrapped_decisions_compared": wrapped_history_decisions,
            "first_wrapped_decision": first_wrapped_history_decision,
            "passed": history_wrap_passed,
        },
        "cpu_pod_trace_summary": trace_summary,
        "exact_tensor_fields": list(GLOBAL_KEYS)
        + [name for names in FAMILY_KEYS.values() for name in names]
        + [f"{family}_mask" for family in FAMILY_KEYS],
        "static_prototype_fields": (
            "option skill/effect prototype relations are injected by the shared adapter "
            "and are gated by logits/greedy parity rather than duplicated per lane"
        ),
        "tensor_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "tensor_mismatch_details": mismatch_details,
        "integer_mask_relation_exact": tensor_passed,
        "model": model_report,
        "full_cpu_causalknowledge_parity": full_passed,
        "extension": str(Path(_ptcg_cuda.__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and not full_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())

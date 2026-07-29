from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CUDA_ENGINE_ROOT.parent
for module_root in (REPO_ROOT / "tools", CUDA_ENGINE_ROOT / "python"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from ptcg_cuda_engine.corpus import (  # noqa: E402
    iter_policy_codec_v1_shards,
    load_policy_codec_v1_manifest,
    pad_policy_codec_v1_rows,
    sha256_file,
)
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    EntityPointerPolicyV1DeviceAdapter,
)
from pure_policy_model_v1 import load_policy_checkpoint  # noqa: E402


MODEL_FIELDS = (
    "global_cat",
    "global_num",
    "entity_cat",
    "entity_num",
    "entity_parent",
    "entity_mask",
    "option_cat",
    "option_num",
    "option_equiv",
    "option_mask",
    "min_count",
    "max_count",
)


def batches_from_corpus(
    corpus_dir: Path,
    *,
    batch_size: int,
    limit: int,
) -> Iterator[dict[str, torch.Tensor]]:
    manifest = load_policy_codec_v1_manifest(corpus_dir)
    entity_capacity = int(manifest["capacity"]["entities"])
    option_capacity = int(manifest["capacity"]["options"])
    emitted = 0
    for shard in iter_policy_codec_v1_shards(corpus_dir):
        decisions = shard.decisions
        for begin in range(0, decisions, batch_size):
            if limit > 0 and emitted >= limit:
                return
            end = min(decisions, begin + batch_size)
            if limit > 0:
                end = min(end, begin + limit - emitted)
            padded = pad_policy_codec_v1_rows(
                shard.arrays,
                begin,
                end,
                entity_capacity=entity_capacity,
                option_capacity=option_capacity,
            )
            yield {name: torch.from_numpy(padded[name]) for name in MODEL_FIELDS}
            emitted += end - begin


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: value.to(device=device) for name, value in batch.items()}


def tensor_actions(actions: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    action_rows = actions.detach().cpu().numpy()
    length_rows = lengths.detach().cpu().numpy()
    return [
        [int(value) for value in action_rows[index, : int(length_rows[index])]]
        for index in range(len(length_rows))
    ]


def compare_actions(
    expected: list[list[int]],
    actual: list[list[int]],
    *,
    offset: int,
    examples: list[dict[str, Any]],
    max_examples: int,
) -> int:
    mismatches = 0
    for index, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        if expected_row != actual_row:
            mismatches += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "decision": offset + index,
                        "expected": expected_row,
                        "actual": actual_row,
                    }
                )
    return mismatches


def compare_action_semantics(
    expected: list[list[int]],
    actual: list[list[int]],
    option_equiv: torch.Tensor,
    *,
    offset: int,
    examples: list[dict[str, Any]],
    max_examples: int,
) -> int:
    equivalence = option_equiv.detach().cpu().numpy()
    mismatches = 0
    for index, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        same = len(expected_row) == len(actual_row)
        expected_groups: list[int] = []
        actual_groups: list[int] = []
        if same:
            for expected_option, actual_option in zip(expected_row, actual_row):
                expected_group = int(equivalence[index, expected_option])
                actual_group = int(equivalence[index, actual_option])
                expected_groups.append(expected_group)
                actual_groups.append(actual_group)
                if expected_option != actual_option and not (
                    expected_group >= 0 and expected_group == actual_group
                ):
                    same = False
        if not same:
            mismatches += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "decision": offset + index,
                        "expected": expected_row,
                        "actual": actual_row,
                        "expected_equivalence": expected_groups,
                        "actual_equivalence": actual_groups,
                    }
                )
    return mismatches


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark the PolicyCodecV1 device adapter."
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parity-start", type=int, default=0)
    parser.add_argument("--parity-decisions", type=int, default=4096)
    parser.add_argument(
        "--cpu-adapter-parity-decisions",
        type=int,
        default=4096,
        help=(
            "Packaged-CPU versus CPU-device-adapter comparison budget; "
            "GPU parity still covers every --parity-decisions row."
        ),
    )
    parser.add_argument("--max-select", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-repeats", type=int, default=None)
    parser.add_argument("--gpu-repeats", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=15)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-mismatch-examples", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cpu_repeats = args.repeats if args.cpu_repeats is None else args.cpu_repeats
    gpu_repeats = args.repeats if args.gpu_repeats is None else args.gpu_repeats
    if min(
        args.batch_size,
        args.parity_decisions,
        args.max_select,
        args.warmup,
        args.repeats,
        cpu_repeats,
        gpu_repeats,
        args.cpu_threads,
    ) <= 0:
        raise ValueError("batch, parity, loop, and thread arguments must be positive")
    if args.cpu_adapter_parity_decisions < 0:
        raise ValueError("CPU adapter parity decisions must be non-negative")
    if args.parity_start < 0 or args.max_mismatch_examples <= 0:
        raise ValueError("parity start must be non-negative and mismatch examples positive")

    corpus_dir = args.corpus_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires an available CUDA device")
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    cpu_model, _ = load_policy_checkpoint(checkpoint, "cpu")
    gpu_model, _ = load_policy_checkpoint(checkpoint, device)
    cpu_model.requires_grad_(False).eval()
    gpu_model.requires_grad_(False).eval()
    cpu_adapter = EntityPointerPolicyV1DeviceAdapter(cpu_model, max_select=args.max_select)
    gpu_adapter = EntityPointerPolicyV1DeviceAdapter(gpu_model, max_select=args.max_select)

    parity_started = time.perf_counter()
    checked = 0
    cpu_adapter_checked = 0
    cpu_adapter_mismatches = 0
    gpu_mismatches = 0
    gpu_semantic_mismatches = 0
    cpu_adapter_examples: list[dict[str, Any]] = []
    gpu_examples: list[dict[str, Any]] = []
    gpu_semantic_examples: list[dict[str, Any]] = []
    benchmark_cpu_batch: dict[str, torch.Tensor] | None = None
    benchmark_gpu_batch: dict[str, torch.Tensor] | None = None
    corpus_offset = 0
    parity_end = args.parity_start + args.parity_decisions
    with torch.inference_mode():
        for cpu_batch in batches_from_corpus(
            corpus_dir,
            batch_size=args.batch_size,
            limit=parity_end,
        ):
            source_rows = int(cpu_batch["option_mask"].shape[0])
            batch_end = corpus_offset + source_rows
            if batch_end <= args.parity_start:
                corpus_offset = batch_end
                continue
            begin = max(0, args.parity_start - corpus_offset)
            end = min(source_rows, parity_end - corpus_offset)
            if begin or end < source_rows:
                cpu_batch = {
                    name: value[begin:end]
                    for name, value in cpu_batch.items()
                }
            corpus_offset = batch_end
            gpu_batch = move_batch(cpu_batch, device)
            expected, _logprob, _entropy, _value = cpu_model.sample_decode_batch(
                cpu_batch,
                mode="greedy",
                compute_entropy=False,
            )
            cpu_adapter_rows = min(
                len(expected),
                max(
                    0,
                    args.cpu_adapter_parity_decisions - cpu_adapter_checked,
                ),
            )
            if cpu_adapter_rows:
                cpu_adapter_batch = {
                    name: value[:cpu_adapter_rows]
                    for name, value in cpu_batch.items()
                }
                cpu_actions, cpu_lengths = cpu_adapter.act_device(cpu_adapter_batch)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp,
            ):
                gpu_actions, gpu_lengths = gpu_adapter.act_device(gpu_batch)
            synchronize(device)
            if cpu_adapter_rows:
                cpu_adapter_mismatches += compare_actions(
                    expected[:cpu_adapter_rows],
                    tensor_actions(cpu_actions, cpu_lengths),
                    offset=args.parity_start + checked,
                    examples=cpu_adapter_examples,
                    max_examples=args.max_mismatch_examples,
                )
                cpu_adapter_checked += cpu_adapter_rows
            gpu_action_rows = tensor_actions(gpu_actions, gpu_lengths)
            gpu_mismatches += compare_actions(
                expected,
                gpu_action_rows,
                offset=args.parity_start + checked,
                examples=gpu_examples,
                max_examples=args.max_mismatch_examples,
            )
            gpu_semantic_mismatches += compare_action_semantics(
                expected,
                gpu_action_rows,
                cpu_batch["option_equiv"],
                offset=args.parity_start + checked,
                examples=gpu_semantic_examples,
                max_examples=args.max_mismatch_examples,
            )
            checked += len(expected)
            if benchmark_cpu_batch is None and len(expected) == args.batch_size:
                benchmark_cpu_batch = cpu_batch
                benchmark_gpu_batch = gpu_batch

    if checked != args.parity_decisions:
        raise RuntimeError(
            f"requested {args.parity_decisions} parity decisions but corpus yielded {checked}"
        )
    if benchmark_cpu_batch is None or benchmark_gpu_batch is None:
        raise RuntimeError("corpus did not yield a full benchmark batch")
    parity_sec = time.perf_counter() - parity_started

    with torch.inference_mode():
        for _ in range(args.warmup):
            cpu_model.sample_decode_batch(
                benchmark_cpu_batch,
                mode="greedy",
                compute_entropy=False,
            )
        cpu_started = time.perf_counter()
        for _ in range(cpu_repeats):
            cpu_model.sample_decode_batch(
                benchmark_cpu_batch,
                mode="greedy",
                compute_entropy=False,
            )
        cpu_sec = time.perf_counter() - cpu_started

        for _ in range(args.warmup):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp,
            ):
                gpu_adapter.act_device(benchmark_gpu_batch)
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push("policy_adapter_hot")
        start_event.record()
        for _ in range(gpu_repeats):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp,
            ):
                gpu_adapter.act_device(benchmark_gpu_batch)
        end_event.record()
        torch.cuda.nvtx.range_pop()
        synchronize(device)
        gpu_sec = start_event.elapsed_time(end_event) / 1000.0

    cpu_workload = args.batch_size * cpu_repeats
    gpu_workload = args.batch_size * gpu_repeats
    cpu_rate = cpu_workload / max(cpu_sec, 1.0e-9)
    gpu_rate = gpu_workload / max(gpu_sec, 1.0e-9)
    manifest = load_policy_codec_v1_manifest(corpus_dir)
    report = {
        "status": "pass"
        if cpu_adapter_mismatches == 0 and gpu_mismatches == 0
        else "fail",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "corpus": str(corpus_dir),
        "corpus_content_sha256": manifest["content_sha256"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "amp": args.amp,
        "cpu_threads": args.cpu_threads,
        "batch_size": args.batch_size,
        "max_select": args.max_select,
        "parity": {
            "start_decision": args.parity_start,
            "decisions": checked,
            "cpu_adapter_decisions": cpu_adapter_checked,
            "cpu_adapter_mismatches": cpu_adapter_mismatches,
            "gpu_mismatches": gpu_mismatches,
            "gpu_raw_index_mismatches": gpu_mismatches,
            "gpu_option_semantic_mismatches": gpu_semantic_mismatches,
            "cpu_adapter_examples": cpu_adapter_examples,
            "gpu_examples": gpu_examples,
            "gpu_option_semantic_examples": gpu_semantic_examples,
            "elapsed_sec": round(parity_sec, 6),
        },
        "policy_only_benchmark": {
            "warmup": args.warmup,
            "cpu_repeats": cpu_repeats,
            "gpu_repeats": gpu_repeats,
            "cpu_workload_decisions": cpu_workload,
            "gpu_workload_decisions": gpu_workload,
            "cpu_sec": round(cpu_sec, 6),
            "gpu_sec": round(gpu_sec, 6),
            "cpu_decisions_per_sec": round(cpu_rate, 3),
            "gpu_decisions_per_sec": round(gpu_rate, 3),
            "gpu_over_cpu_speedup": round(gpu_rate / max(cpu_rate, 1.0e-9), 3),
            "nvtx_range": "policy_adapter_hot",
            "timed_range_includes_transfers": False,
            "timed_range_includes_host_synchronize": False,
        },
        "gpu_memory": {
            "allocated_mib": round(torch.cuda.memory_allocated(device) / (1024**2), 3),
            "reserved_mib": round(torch.cuda.memory_reserved(device) / (1024**2), 3),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / (1024**2),
                3,
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(device) / (1024**2),
                3,
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out is not None:
        output_path = args.out.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

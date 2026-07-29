from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CUDA_ENGINE_ROOT.parent
for module_root in (REPO_ROOT / "tools", CUDA_ENGINE_ROOT / "python", CUDA_ENGINE_ROOT / "tools"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from benchmark_policy_codec_v1_device import (  # noqa: E402
    batches_from_corpus,
    move_batch,
    tensor_actions,
)
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    EntityPointerPolicyV1DeviceAdapter,
)
from pure_policy_model_v1 import load_policy_checkpoint  # noqa: E402


def trace_decode(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    target_rows: list[int],
    *,
    max_select: int,
) -> tuple[list[list[int]], dict[int, list[dict[str, Any]]]]:
    outputs = model.encode(dict(batch))
    option_repr = outputs["option_repr"]
    option_mask = batch["option_mask"].bool()
    batch_size, option_count = option_mask.shape
    hidden = model.initial_decoder_state(outputs["state_repr"])
    selected_mask = torch.zeros_like(option_mask)
    actions = torch.full(
        (batch_size, max_select),
        -1,
        dtype=torch.long,
        device=option_mask.device,
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=option_mask.device)
    min_count = batch["min_count"].long().view(batch_size)
    max_count = batch["max_count"].long().view(batch_size).clamp(
        min=0,
        max=max_select,
    )
    active = max_count.gt(0)
    traces: dict[int, list[dict[str, Any]]] = {row: [] for row in target_rows}

    for step in range(max_select):
        active_before = active
        valid_options = option_mask & ~selected_mask & active_before.unsqueeze(1)
        stop_valid = active_before & min_count.le(step)
        logits = model.pointer_logits(
            option_repr,
            hidden,
            valid_options,
            stop_valid,
        )
        choice = logits.argmax(dim=1)

        selected_logits = logits[target_rows].detach().float().cpu()
        selected_choices = choice[target_rows].detach().cpu()
        selected_active = active_before[target_rows].detach().cpu()
        top_values, top_indices = selected_logits.topk(k=2, dim=1)
        for index, row in enumerate(target_rows):
            if not bool(selected_active[index]):
                continue
            traces[row].append(
                {
                    "step": step,
                    "choice": int(selected_choices[index]),
                    "top_indices": [int(value) for value in top_indices[index].tolist()],
                    "top_logits": [float(value) for value in top_values[index].tolist()],
                    "top_margin": float(top_values[index, 0] - top_values[index, 1]),
                }
            )

        chosen_valid = active_before & choice.lt(option_count)
        safe_choice = choice.clamp(min=0, max=option_count - 1)
        actions[:, step] = torch.where(chosen_valid, safe_choice, -1)
        previously_selected = selected_mask.gather(1, safe_choice.unsqueeze(1))
        selected_mask.scatter_(
            1,
            safe_choice.unsqueeze(1),
            previously_selected | chosen_valid.unsqueeze(1),
        )
        lengths = lengths + chosen_valid.long()
        active = chosen_valid & lengths.lt(max_count)
        selected_repr = option_repr.gather(
            1,
            safe_choice.view(-1, 1, 1).expand(-1, 1, option_repr.size(-1)),
        ).squeeze(1)
        hidden = model.advance_decoder(selected_repr, hidden, chosen_valid)

    all_actions = tensor_actions(actions, lengths)
    return [all_actions[row] for row in target_rows], traces


def first_trace_difference(
    cpu_trace: list[dict[str, Any]],
    gpu_trace: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for step in range(max(len(cpu_trace), len(gpu_trace))):
        cpu_row = cpu_trace[step] if step < len(cpu_trace) else None
        gpu_row = gpu_trace[step] if step < len(gpu_trace) else None
        if cpu_row is None or gpu_row is None or cpu_row["choice"] != gpu_row["choice"]:
            return {"step": step, "cpu": cpu_row, "gpu": gpu_row}
    return None


def cast_floating_batch(
    batch: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(dtype=dtype) if value.is_floating_point() else value
        for name, value in batch.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose rare CPU/GPU greedy-action differences at fixed corpus rows."
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-select", type=int, default=80)
    parser.add_argument("--cpu-threads", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    decisions = sorted({int(value) for value in args.decisions.split(",") if value.strip()})
    if not decisions or decisions[0] < 0:
        raise ValueError("decisions must contain non-negative corpus row indexes")
    if min(args.batch_size, args.max_select, args.cpu_threads) <= 0:
        raise ValueError("batch size, max select, and CPU threads must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cpu_model, _ = load_policy_checkpoint(args.checkpoint.resolve(), "cpu")
    gpu_model, _ = load_policy_checkpoint(args.checkpoint.resolve(), device)
    gpu_dtype = torch.float64 if args.gpu_dtype == "float64" else torch.float32
    gpu_model.to(dtype=gpu_dtype)
    cpu_model.requires_grad_(False).eval()
    gpu_model.requires_grad_(False).eval()
    gpu_adapter = EntityPointerPolicyV1DeviceAdapter(
        gpu_model,
        max_select=args.max_select,
    )

    remaining = set(decisions)
    offset = 0
    batch_reports: list[dict[str, Any]] = []
    with torch.inference_mode():
        for cpu_batch in batches_from_corpus(
            args.corpus_dir.resolve(),
            batch_size=args.batch_size,
            limit=decisions[-1] + 1,
        ):
            rows = int(cpu_batch["option_mask"].shape[0])
            batch_targets = sorted(value for value in remaining if offset <= value < offset + rows)
            if batch_targets:
                local_rows = [value - offset for value in batch_targets]
                gpu_batch = cast_floating_batch(
                    move_batch(cpu_batch, device),
                    gpu_dtype,
                )
                packaged_cpu, _logprob, _entropy, _value = cpu_model.sample_decode_batch(
                    cpu_batch,
                    mode="greedy",
                    compute_entropy=False,
                )
                gpu_actions, gpu_lengths = gpu_adapter.act_device(gpu_batch)
                torch.cuda.synchronize(device)
                gpu_all = tensor_actions(gpu_actions, gpu_lengths)
                cpu_trace_actions, cpu_traces = trace_decode(
                    cpu_model,
                    cpu_batch,
                    local_rows,
                    max_select=args.max_select,
                )
                gpu_trace_actions, gpu_traces = trace_decode(
                    gpu_model,
                    gpu_batch,
                    local_rows,
                    max_select=args.max_select,
                )
                row_reports = []
                for item_index, (global_row, local_row) in enumerate(
                    zip(batch_targets, local_rows)
                ):
                    expected = packaged_cpu[local_row]
                    actual = gpu_all[local_row]
                    singleton_cpu_batch = {
                        name: value[local_row : local_row + 1]
                        for name, value in cpu_batch.items()
                    }
                    singleton_gpu_batch = {
                        name: value[local_row : local_row + 1]
                        for name, value in gpu_batch.items()
                    }
                    singleton_cpu, _logprob, _entropy, _value = (
                        cpu_model.sample_decode_batch(
                            singleton_cpu_batch,
                            mode="greedy",
                            compute_entropy=False,
                        )
                    )
                    singleton_gpu_actions, singleton_gpu_lengths = (
                        gpu_adapter.act_device(singleton_gpu_batch)
                    )
                    torch.cuda.synchronize(device)
                    singleton_gpu = tensor_actions(
                        singleton_gpu_actions,
                        singleton_gpu_lengths,
                    )
                    first_difference = first_trace_difference(
                        cpu_traces[local_row],
                        gpu_traces[local_row],
                    )
                    difference_equivalence = None
                    if first_difference is not None:
                        cpu_difference = first_difference["cpu"]
                        gpu_difference = first_difference["gpu"]
                        if cpu_difference is not None and gpu_difference is not None:
                            cpu_choice = int(cpu_difference["choice"])
                            gpu_choice = int(gpu_difference["choice"])
                            option_count = int(cpu_batch["option_mask"].shape[1])
                            if cpu_choice < option_count and gpu_choice < option_count:
                                cpu_group = int(
                                    cpu_batch["option_equiv"][local_row, cpu_choice]
                                )
                                gpu_group = int(
                                    cpu_batch["option_equiv"][local_row, gpu_choice]
                                )
                                difference_equivalence = {
                                    "cpu_group": cpu_group,
                                    "gpu_group": gpu_group,
                                    "same_nonnegative_group": (
                                        cpu_group >= 0 and cpu_group == gpu_group
                                    ),
                                }
                    row_reports.append(
                        {
                            "decision": global_row,
                            "local_row": local_row,
                            "packaged_cpu_action": expected,
                            "gpu_adapter_action": actual,
                            "singleton_cpu_action": singleton_cpu[0],
                            "singleton_gpu_action": singleton_gpu[0],
                            "cpu_trace_action": cpu_trace_actions[item_index],
                            "gpu_trace_action": gpu_trace_actions[item_index],
                            "reproduced_mismatch": expected != actual,
                            "cpu_trace_matches_packaged": (
                                cpu_trace_actions[item_index] == expected
                            ),
                            "gpu_trace_matches_adapter": (
                                gpu_trace_actions[item_index] == actual
                            ),
                            "first_trace_difference": first_difference,
                            "first_difference_equivalence": difference_equivalence,
                        }
                    )
                batch_reports.append(
                    {
                        "batch_offset": offset,
                        "batch_rows": rows,
                        "targets": batch_targets,
                        "rows": row_reports,
                    }
                )
                remaining.difference_update(batch_targets)
                if not remaining:
                    break
            offset += rows

    if remaining:
        raise RuntimeError(f"corpus did not contain decisions: {sorted(remaining)}")
    flat_rows = [row for batch in batch_reports for row in batch["rows"]]
    reproduced = sum(bool(row["reproduced_mismatch"]) for row in flat_rows)
    report = {
        "status": "pass" if reproduced == len(decisions) else "incomplete_reproduction",
        "device": torch.cuda.get_device_name(device),
        "gpu_dtype": args.gpu_dtype,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "requested_decisions": decisions,
        "reproduced_mismatches": reproduced,
        "batches": batch_reports,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(payload, end="")
    if args.out:
        args.out.resolve().write_text(payload, encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

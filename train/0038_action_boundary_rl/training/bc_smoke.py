"""Short, non-formal BC diagnostic over the bounded POD corpus."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CUDA_PYTHON = REPOSITORY_ROOT / "engine_cuda" / "python"
if str(CUDA_PYTHON) not in sys.path:
    sys.path.insert(0, str(CUDA_PYTHON))

from ptcg_cuda_engine.corpus import iter_policy_codec_v1_shards, pad_policy_codec_v1_rows  # noqa: E402


project = importlib.import_module("train.0038_action_boundary_rl")
model_module = importlib.import_module("train.0038_action_boundary_rl.model")
transfer_module = importlib.import_module("train.0038_action_boundary_rl.transfer")


ACTOR_FIELDS = tuple(project.contract.FIELD_SHAPES)


def ordered_bc_objective(
    model: Any,
    batch: Mapping[str, Tensor],
    actions: Tensor,
    action_len: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    validated = model.validate_batch(batch)
    encoded = model.encode(validated)
    state = model.action_decoder.initialize(validated, encoded.state_summary)
    option_count = validated.option_count
    total_loss = encoded.value.sum() * 0.0
    correct = torch.zeros((), dtype=torch.long, device=encoded.value.device)
    tokens = torch.zeros((), dtype=torch.long, device=encoded.value.device)
    exact = torch.ones(validated.batch_size, dtype=torch.bool, device=encoded.value.device)

    for step in range(actions.shape[1] + 1):
        logits = model.action_decoder.logits(validated, encoded.option_tokens, state)
        has_action = action_len.gt(step)
        stop_here = action_len.eq(step) & action_len.lt(validated.max_count)
        supervised = has_action | stop_here
        if step < actions.shape[1]:
            raw_target = actions[:, step]
        else:
            raw_target = torch.full_like(action_len, -1)
        target = torch.where(has_action, raw_target, torch.full_like(action_len, option_count))
        row_loss = F.cross_entropy(logits, target, reduction="none")
        total_loss = total_loss + (row_loss * supervised).sum()
        prediction = logits.argmax(dim=1)
        row_correct = prediction.eq(target)
        correct = correct + (row_correct & supervised).sum()
        tokens = tokens + supervised.sum()
        exact = exact & (~supervised | row_correct)
        state = model.action_decoder.consume(
            encoded.option_tokens,
            state,
            torch.where(has_action, raw_target, torch.full_like(raw_target, -1)),
        )
    loss = total_loss / tokens.clamp_min(1)
    return loss, {
        "loss": loss.detach(),
        "token_accuracy": correct.float() / tokens.clamp_min(1),
        "teacher_exact": exact.float().mean(),
        "tokens": tokens,
    }


def iter_batches(
    corpus: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[dict[str, Tensor], Tensor, Tensor]]:
    for shard in iter_policy_codec_v1_shards(corpus):
        decisions = shard.decisions
        for begin in range(0, decisions, batch_size):
            end = min(begin + batch_size, decisions)
            padded = pad_policy_codec_v1_rows(
                shard.arrays,
                begin,
                end,
                entity_capacity=128,
                option_capacity=128,
            )
            actor = {
                name: torch.from_numpy(np.ascontiguousarray(padded[name])).to(device)
                for name in ACTOR_FIELDS
            }
            actions = torch.from_numpy(np.ascontiguousarray(padded["actions"])).to(device)
            action_len = torch.from_numpy(np.ascontiguousarray(padded["action_len"])).to(device)
            yield actor, actions, action_len


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("no metric rows")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def evaluate(
    model: Any,
    corpus: Path,
    *,
    batch_size: int,
    batches: int,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    context = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if amp
        else nullcontext()
    )
    started = time.perf_counter()
    examples = 0
    with torch.no_grad():
        for index, (batch, actions, lengths) in enumerate(
            iter_batches(corpus, batch_size=batch_size, device=device)
        ):
            if index >= batches:
                break
            with context():
                _, metrics = ordered_bc_objective(model, batch, actions, lengths)
            rows.append(
                {
                    "loss": float(metrics["loss"].float().item()),
                    "token_accuracy": float(metrics["token_accuracy"].float().item()),
                    "teacher_exact": float(metrics["teacher_exact"].float().item()),
                }
            )
            examples += int(lengths.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = aggregate(rows)
    result.update({"examples": examples, "seconds": elapsed, "examples_per_second": examples / elapsed})
    return result


def run_arm(
    *,
    initialization: str,
    train_corpus: Path,
    validation_corpus: Path,
    source_checkpoint: Path,
    batch_size: int,
    train_batches: int,
    validation_batches: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(32032)
    model = model_module.PodNativeActorCritic().to(device)
    transfer_report = None
    if initialization == "0031_explicit_transfer":
        transfer_report = transfer_module.load_0031_initialization(model, source_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    amp = device.type == "cuda"
    initial = evaluate(
        model,
        validation_corpus,
        batch_size=batch_size,
        batches=validation_batches,
        device=device,
        amp=amp,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    examples = 0
    for index, (batch, actions, lengths) in enumerate(
        iter_batches(train_corpus, batch_size=batch_size, device=device)
    ):
        if index >= train_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
            loss, metrics = ordered_bc_objective(model, batch, actions, lengths)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows.append(
            {
                "loss": float(metrics["loss"].float().item()),
                "token_accuracy": float(metrics["token_accuracy"].float().item()),
                "teacher_exact": float(metrics["teacher_exact"].float().item()),
                "grad_norm": float(grad_norm.float().item()),
            }
        )
        examples += int(lengths.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = evaluate(
        model,
        validation_corpus,
        batch_size=batch_size,
        batches=validation_batches,
        device=device,
        amp=amp,
    )
    return {
        "initialization": initialization,
        "initial_validation": initial,
        "optimization": {
            **aggregate(rows),
            "batches": len(rows),
            "examples": examples,
            "seconds": elapsed,
            "examples_per_second": examples / elapsed,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "final_validation": final,
        "transfer_fraction": (
            transfer_report["transferred_parameter_fraction"] if transfer_report else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = []
    for initialization in ("random", "0031_explicit_transfer"):
        arms.append(
            run_arm(
                initialization=initialization,
                train_corpus=args.corpus / "train",
                validation_corpus=args.corpus / "validation",
                source_checkpoint=args.source_checkpoint,
                batch_size=args.batch_size,
                train_batches=args.train_batches,
                validation_batches=args.validation_batches,
                learning_rate=args.learning_rate,
                device=device,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema": "0034_pod_bc_smoke_v1",
        "purpose": "gradient_and_label_contract_diagnostic_not_policy_strength",
        "device": str(device),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "train_batches": args.train_batches,
        "validation_batches": args.validation_batches,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

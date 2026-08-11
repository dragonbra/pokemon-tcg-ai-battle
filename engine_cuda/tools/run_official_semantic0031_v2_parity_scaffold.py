from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
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
    "option_skill": (
        "option_skill_id",
        "option_skill_role",
        "option_skill_parent",
    ),
    "option_effect": (
        "option_effect_id",
        "option_effect_role",
        "option_effect_parent",
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
        "--training-checkpoint",
        type=Path,
        help="Optional numbered-project model-only checkpoint used for Value parity.",
    )
    parser.add_argument(
        "--training-project",
        default="train.0038_action_boundary_rl",
        help="Numbered project package that strictly reconstructs --training-checkpoint.",
    )
    parser.add_argument(
        "--training-deck",
        type=Path,
        help=(
            "Optional exact-60 deck used to reconstruct the numbered-project model. "
            "When omitted, preserve legacy semantics by using the trace focal deck."
        ),
    )
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
    parser.add_argument(
        "--reuse-trace",
        action="store_true",
        help=(
            "Read --trace as an immutable captured official trace without "
            "building or executing the trace producer."
        ),
    )
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--model-chunk", type=int, default=2)
    parser.add_argument("--logits-atol", type=float, default=2e-4)
    parser.add_argument("--logits-rtol", type=float, default=2e-4)
    parser.add_argument("--value-atol", type=float, default=2e-6)
    parser.add_argument("--deployment-logits-atol", type=float, default=5e-3)
    parser.add_argument("--deployment-logits-rtol", type=float, default=5e-3)
    parser.add_argument("--deployment-value-atol", type=float, default=2e-3)
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
    if args.reuse_trace:
        if not args.trace.is_file():
            raise FileNotFoundError(args.trace)
        trace = load_trace(args.trace)
        digest = hashlib.sha256(args.trace.read_bytes()).hexdigest()
        return {
            "passed": True,
            "scope": "immutable_official_semantic_trace",
            "semantic_trace_records": len(trace),
            "trace_sha256": digest,
            "reused": True,
        }
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
    if (package_root / "strategy").is_dir():
        # 0038 Kaggle candidates carry the same audited compiler under the
        # production package namespace rather than the historical semantic0031
        # directory name.  Supporting it here lets the parity gate exercise the
        # exact deployable files instead of a copied diagnostic-only runtime.
        module_root = "strategy"
        asset_root = package_root / "strategy/assets"
    else:
        module_root = "semantic0031"
        asset_root = package_root / "semantic0031/assets"
    compiler = importlib.import_module(f"{module_root}.features.compiler")
    collate = importlib.import_module(f"{module_root}.features.collate")
    knowledge = importlib.import_module(f"{module_root}.knowledge.state")
    prototypes = importlib.import_module(f"{module_root}.domain.prototypes")
    prototype_index = prototypes.PrototypeIndex.load(
        asset_root / "official_public_prototypes_v1.json",
        asset_root / "official_full_engine_prototypes_v2.json",
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
    first_greedy_divergence: dict[str, Any] | None = None
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
                    if first_greedy_divergence is None:
                        legal_cpu_logits = torch.cat(
                            (cpu_logits[row, :option_count], cpu_logits[row, -1:])
                        ).float()
                        legal_cuda_logits = torch.cat(
                            (cuda_logits[row, :option_count], cuda_logits[row, -1:])
                        ).float()
                        top_count = min(5, int(legal_cpu_logits.numel()))
                        first_greedy_divergence = {
                            "decision": start + row,
                            "cpu_greedy": expected_action,
                            "cuda_greedy": actual_action,
                            "option_count": option_count,
                            "cpu_legal_plus_stop_logits": legal_cpu_logits.cpu().tolist(),
                            "cuda_legal_plus_stop_logits": legal_cuda_logits.cpu().tolist(),
                            "cpu_topk": [
                                {"index": int(index), "logit": float(value)}
                                for value, index in zip(
                                    *legal_cpu_logits.topk(top_count), strict=True
                                )
                            ],
                            "cuda_topk": [
                                {"index": int(index), "logit": float(value)}
                                for value, index in zip(
                                    *legal_cuda_logits.topk(top_count), strict=True
                                )
                            ],
                        }
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


def load_compound_policy(package_root: Path, deck: Sequence[int], device: Any) -> Any:
    """Load the exact 0038 production actor and conditional heads strictly."""

    module = importlib.import_module("strategy.deployment.compound_inference")
    policy = module.PortableCompoundSemanticPolicy.from_checkpoint(
        package_root / "strategy/model.bin", deck
    )
    policy.actor.to(device).eval()
    policy.value_head.to(device).eval()
    policy.allocation_head.to(device).eval()
    if hasattr(policy, "policy_strategy_adapter"):
        policy.value_adapter.to(device).eval()
        policy.policy_strategy_adapter.to(device).eval()
    else:
        policy.meta_head.to(device).eval()
        policy.meta_conditioner.to(device).eval()
    return policy


def decode_strategy_conditioned(policy: Any, mapping: Mapping[str, Any]):
    """Decode the real 0042 Value-to-Policy path for parity comparisons."""

    import torch

    validated, state, options, value, _, context = policy.encode_with_strategy(mapping)
    decoder = policy.actor.action_decoder

    def logits(decoder_state):
        readout, _ = policy.policy_strategy_adapter(decoder_state.hidden, context)
        return decoder.logits(
            validated, options, decoder_state, readout_hidden=readout
        )

    decoder_state = decoder.initialize(validated, state.summary)
    root_logits = logits(decoder_state)
    maximum_steps = min(
        decoder.config.max_action_steps,
        validated.option_count,
        int(validated.max_count.max()),
    )
    sequences = torch.full(
        (validated.batch_size, maximum_steps), -1,
        dtype=torch.long, device=options.device,
    )
    lengths = torch.zeros(
        validated.batch_size, dtype=torch.long, device=options.device
    )
    legal = torch.ones(
        validated.batch_size, dtype=torch.bool, device=options.device
    )
    active = torch.ones_like(legal)
    for step in range(maximum_steps):
        choice = logits(decoder_state).argmax(dim=1)
        chose_stop = choice.eq(validated.option_count)
        selecting = active & ~chose_stop
        legal &= ~(active & chose_stop & lengths.lt(validated.min_count))
        active &= ~chose_stop
        if not bool(selecting.any()):
            break
        chosen = choice.clamp_max(validated.option_count - 1)
        sequences[selecting, step] = chosen[selecting]
        decoder_state = decoder.consume(
            options, decoder_state, torch.where(selecting, chosen, -1)
        )
        lengths += selecting.long()
        active &= ~lengths.ge(validated.max_count)
    legal &= lengths.ge(validated.min_count) & lengths.le(validated.max_count)
    return (
        validated,
        state,
        options,
        root_logits,
        SimpleNamespace(sequences=sequences, lengths=lengths, legal=legal),
        value,
    )


def compare_compound_model_chunks(
    policy: Any,
    collate_records: Any,
    cpu_records: Sequence[Mapping[str, Any]],
    cuda_records: Sequence[Mapping[str, Any]],
    option_counts: Sequence[int],
    *,
    chunk_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Measure policy-intent drift caused only by CPU/CUDA semantic tensors."""

    import torch

    actor = policy.actor
    device = next(actor.parameters()).device
    greedy_divergences = 0
    first_step_top1_divergences = 0
    first_step_top2_set_divergences = 0
    logits_failures = 0
    maximum_absolute_error = 0.0
    absolute_error_sum = 0.0
    absolute_error_values = 0
    minimum_cpu_margin = float("inf")
    minimum_cuda_margin = float("inf")
    first_failure: dict[str, Any] | None = None
    first_greedy_divergence: dict[str, Any] | None = None
    decisions = 0

    def decode(mapping: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        if hasattr(policy, "policy_strategy_adapter"):
            _, state, _, root_logits, greedy, _ = decode_strategy_conditioned(
                policy, mapping
            )
            return root_logits, greedy, state.summary
        validated, state, options = actor.encode(mapping)
        meta_logits = policy.meta_head(state.summary)
        summary = policy.meta_conditioner(state.summary, meta_logits)
        decoder = actor.action_decoder.initialize(validated, summary)
        root_logits = actor.action_decoder.logits(validated, options, decoder)
        greedy = actor.action_decoder.greedy(validated, options, summary)
        return root_logits, greedy, state.summary

    with torch.inference_mode():
        for start in range(0, len(cpu_records), chunk_size):
            stop = min(len(cpu_records), start + chunk_size)
            cpu_mapping = {
                name: tensor.to(device)
                for name, tensor in collate_records(cpu_records[start:stop]).items()
            }
            cuda_mapping = semantic0031_v2_ready_batch({
                name: torch.cat(
                    [record[name] for record in cuda_records[start:stop]], dim=0
                ).to(device)
                for name in cuda_records[start]
            })
            cpu_logits, cpu_greedy, _ = decode(cpu_mapping)
            cuda_logits, cuda_greedy, _ = decode(cuda_mapping)
            for row, option_count in enumerate(option_counts[start:stop]):
                expected_action = extract_greedy(cpu_greedy, row)
                actual_action = extract_greedy(cuda_greedy, row)
                if expected_action != actual_action:
                    greedy_divergences += 1
                    if first_greedy_divergence is None:
                        legal_cpu_logits = torch.cat(
                            (cpu_logits[row, :option_count], cpu_logits[row, -1:])
                        ).float()
                        legal_cuda_logits = torch.cat(
                            (cuda_logits[row, :option_count], cuda_logits[row, -1:])
                        ).float()
                        top_count = min(5, int(legal_cpu_logits.numel()))
                        first_greedy_divergence = {
                            "decision": start + row,
                            "cpu_greedy": expected_action,
                            "cuda_greedy": actual_action,
                            "option_count": option_count,
                            "cpu_legal_plus_stop_logits": legal_cpu_logits.cpu().tolist(),
                            "cuda_legal_plus_stop_logits": legal_cuda_logits.cpu().tolist(),
                            "cpu_topk": [
                                {"index": int(index), "logit": float(value)}
                                for value, index in zip(
                                    *legal_cpu_logits.topk(top_count), strict=True
                                )
                            ],
                            "cuda_topk": [
                                {"index": int(index), "logit": float(value)}
                                for value, index in zip(
                                    *legal_cuda_logits.topk(top_count), strict=True
                                )
                            ],
                        }
                    if first_failure is None:
                        first_failure = {
                            "decision": start + row,
                            "stage": "greedy_action",
                            "cpu_greedy": expected_action,
                            "cuda_greedy": actual_action,
                        }
                expected_logits = torch.cat(
                    (cpu_logits[row, :option_count], cpu_logits[row, -1:])
                ).float()
                actual_logits = torch.cat(
                    (cuda_logits[row, :option_count], cuda_logits[row, -1:])
                ).float()
                finite = torch.isfinite(expected_logits) & torch.isfinite(actual_logits)
                if bool(finite.any()):
                    difference = (expected_logits[finite] - actual_logits[finite]).abs()
                    maximum_absolute_error = max(
                        maximum_absolute_error, float(difference.max().cpu())
                    )
                    absolute_error_sum += float(difference.sum().cpu())
                    absolute_error_values += int(difference.numel())
                close = torch.isclose(
                    expected_logits, actual_logits, atol=atol, rtol=rtol,
                    equal_nan=False,
                )
                if not bool(close.all()):
                    logits_failures += 1
                    if first_failure is None:
                        first_failure = {
                            "decision": start + row,
                            "stage": "root_logits",
                            "max_abs": maximum_absolute_error,
                        }
                top_count = min(2, int(expected_logits.numel()))
                cpu_values, cpu_indices = expected_logits.topk(top_count)
                cuda_values, cuda_indices = actual_logits.topk(top_count)
                if int(cpu_indices[0]) != int(cuda_indices[0]):
                    first_step_top1_divergences += 1
                if set(cpu_indices.tolist()) != set(cuda_indices.tolist()):
                    first_step_top2_set_divergences += 1
                if top_count == 2:
                    minimum_cpu_margin = min(
                        minimum_cpu_margin, float((cpu_values[0] - cpu_values[1]).cpu())
                    )
                    minimum_cuda_margin = min(
                        minimum_cuda_margin, float((cuda_values[0] - cuda_values[1]).cpu())
                    )
                decisions += 1
    return {
        "runtime": "0038_compound_kaggle_candidate_v4",
        "decisions": decisions,
        "logits_atol": atol,
        "logits_rtol": rtol,
        "max_root_logit_absolute_error": maximum_absolute_error,
        "mean_root_logit_absolute_error": (
            absolute_error_sum / max(1, absolute_error_values)
        ),
        "root_logits_tolerance_failures": logits_failures,
        "first_step_top1_divergences": first_step_top1_divergences,
        "first_step_top2_set_divergences": first_step_top2_set_divergences,
        "greedy_action_divergences": greedy_divergences,
        "minimum_cpu_top1_top2_margin": (
            minimum_cpu_margin if minimum_cpu_margin != float("inf") else None
        ),
        "minimum_cuda_top1_top2_margin": (
            minimum_cuda_margin if minimum_cuda_margin != float("inf") else None
        ),
        "first_failure": first_failure,
        "first_greedy_divergence": first_greedy_divergence,
        "value": "strict_loaded_diagnostic_only",
        "passed": logits_failures == 0 and greedy_divergences == 0,
    }


def load_training_actor_critic(
    checkpoint: Path,
    deck: Sequence[int],
    device: Any,
    *,
    training_project: str = "train.0038_action_boundary_rl",
) -> Any:
    """Strictly reconstruct a numbered-project model-only checkpoint."""

    import torch

    if re.fullmatch(r"train\.\d{4}_[a-z0-9_]+", training_project) is None:
        raise ValueError("training project must be an explicit numbered train package")
    actor_critic = importlib.import_module(f"{training_project}.policy.actor_critic")
    adaptation_module = importlib.import_module(f"{training_project}.policy.adaptation")
    integrated_module = importlib.import_module(f"{training_project}.integrated.config")
    storage = importlib.import_module(
        f"{training_project}.training.storage_full_semantic"
    )
    source = importlib.import_module(f"{training_project}.source")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    adaptation_payload = payload["adaptation"]
    strategy_conditioned_adaptation = {
        "no_option_lora": True,
        "value_adapter": "zero_gated_residual",
        "policy_strategy_adapter": "readout_only_zero_gated_residual",
    }
    if adaptation_payload == strategy_conditioned_adaptation:
        adaptation = adaptation_module.AdaptationConfig()
    else:
        adaptation = adaptation_module.AdaptationConfig(**adaptation_payload)
    model, _ = actor_critic.load_actor_critic(
        source.ACTOR_CHECKPOINT,
        tuple(int(card) for card in deck),
        device,
        adaptation=adaptation,
        integrated_flags=integrated_module.IntegratedFlags(
            **payload["integrated_flags"]
        ),
    )
    storage.load_adapted_model_only(model, checkpoint)
    return model.eval()


def compare_value_chunks(
    model: Any,
    collate_records: Any,
    cpu_records: Sequence[Mapping[str, Any]],
    cuda_records: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int,
    atol: float,
) -> dict[str, Any]:
    """Compare V_win under official-package and resident-CUDA feature semantics."""

    import torch

    device = model.device
    maximum_absolute_error = 0.0
    absolute_error_sum = 0.0
    values = 0
    sign_divergences = 0
    first_divergence: dict[str, Any] | None = None
    maximum_divergence: dict[str, Any] | None = None
    sign_divergence_samples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(cpu_records), chunk_size):
            stop = min(len(cpu_records), start + chunk_size)
            cpu_mapping = {
                name: tensor.to(device)
                for name, tensor in collate_records(cpu_records[start:stop]).items()
            }
            cuda_mapping = semantic0031_v2_ready_batch({
                name: torch.cat(
                    [record[name] for record in cuda_records[start:stop]], dim=0
                ).to(device)
                for name in cuda_records[start]
            })
            _, _, _, cpu_value = model.encode(cpu_mapping)
            _, _, _, cuda_value = model.encode(cuda_mapping)
            difference = (cpu_value.float() - cuda_value.float()).abs()
            maximum_absolute_error = max(
                maximum_absolute_error, float(difference.max().cpu())
            )
            absolute_error_sum += float(difference.sum().cpu())
            values += int(difference.numel())
            sign = cpu_value.sign().ne(cuda_value.sign())
            sign_divergences += int(sign.sum().cpu())
            maximum_row = int(difference.argmax())
            maximum_row_error = float(difference[maximum_row].cpu())
            if (
                maximum_divergence is None
                or maximum_row_error > maximum_divergence["absolute_error"]
            ):
                maximum_divergence = {
                    "decision": start + maximum_row,
                    "cpu_value": float(cpu_value[maximum_row].cpu()),
                    "cuda_value": float(cuda_value[maximum_row].cpu()),
                    "absolute_error": maximum_row_error,
                }
            for row in sign.nonzero(as_tuple=False).flatten().tolist():
                if len(sign_divergence_samples) >= 16:
                    break
                sign_divergence_samples.append({
                    "decision": start + int(row),
                    "cpu_value": float(cpu_value[row].cpu()),
                    "cuda_value": float(cuda_value[row].cpu()),
                    "absolute_error": float(difference[row].cpu()),
                })
            if first_divergence is None and bool(difference.gt(atol).any()):
                row = int(difference.gt(atol).nonzero(as_tuple=False)[0])
                first_divergence = {
                    "decision": start + row,
                    "cpu_value": float(cpu_value[row].cpu()),
                    "cuda_value": float(cuda_value[row].cpu()),
                    "absolute_error": float(difference[row].cpu()),
                }
    return {
        "decisions": values,
        "max_value_absolute_error": maximum_absolute_error,
        "mean_value_absolute_error": absolute_error_sum / max(1, values),
        "value_sign_divergences": sign_divergences,
        "first_divergence": first_divergence,
        "maximum_divergence": maximum_divergence,
        "sign_divergence_samples": sign_divergence_samples,
        "atol": atol,
        "passed": maximum_absolute_error <= atol and sign_divergences == 0,
    }


def compare_training_to_package_chunks(
    training_model: Any,
    package_policy: Any,
    collate_records: Any,
    cpu_records: Sequence[Mapping[str, Any]],
    option_counts: Sequence[int],
    *,
    chunk_size: int,
    logits_atol: float,
    logits_rtol: float,
    value_atol: float,
) -> dict[str, Any]:
    """Compare the training checkpoint to its portable merged/FP16 package.

    This is distinct from CPU-vs-CUDA feature parity: both sides consume the
    exact same official CPU feature tensors, so any difference belongs to
    model export, LoRA merging, dtype conversion, or strict component loading.
    """

    import torch

    device = training_model.device
    package_actor = package_policy.actor
    root_max = root_sum = 0.0
    root_values = root_failures = greedy_divergences = 0
    root_top1_divergences = root_top2_set_divergences = 0
    minimum_training_margin = minimum_package_margin = float("inf")
    value_max = value_sum = 0.0
    value_values = value_sign_divergences = 0
    allocation_max = allocation_sum = 0.0
    allocation_values = allocation_top1_divergences = allocation_decisions = 0
    allocation_centered_max = allocation_probability_max = 0.0
    allocation_top2_set_divergences = 0
    minimum_training_allocation_margin = float("inf")
    minimum_package_allocation_margin = float("inf")
    first_root_divergence = None
    first_value_divergence = None
    first_allocation_divergence = None
    dragapult = importlib.import_module("strategy.action_boundary.dragapult")

    def decode_package(mapping):
        if hasattr(package_policy, "policy_strategy_adapter"):
            return decode_strategy_conditioned(package_policy, mapping)
        validated, state, options = package_actor.encode(mapping)
        meta_logits = package_policy.meta_head(state.summary)
        summary = package_policy.meta_conditioner(state.summary, meta_logits)
        decoder = package_actor.action_decoder.initialize(validated, summary)
        logits = package_actor.action_decoder.logits(validated, options, decoder)
        greedy = package_actor.action_decoder.greedy(validated, options, summary)
        value = package_policy.value_from_encoded(validated, state, options)
        return validated, state, options, logits, greedy, value

    def decode_training(mapping):
        if hasattr(training_model, "policy_strategy_adapter"):
            return decode_strategy_conditioned(training_model, mapping)
        validated, state, options = training_model.actor.encode(mapping)
        summary = training_model.actor_summary(state)
        decoder = training_model.actor.action_decoder.initialize(validated, summary)
        logits = training_model.actor.action_decoder.logits(
            validated, options, decoder
        )
        greedy = training_model.actor.action_decoder.greedy(
            validated, options, summary
        )
        value = training_model.value_from_encoded(validated, state, options)
        return validated, state, options, logits, greedy, value

    with torch.inference_mode():
        for start in range(0, len(cpu_records), chunk_size):
            stop = min(len(cpu_records), start + chunk_size)
            mapping = {
                name: tensor.to(device)
                for name, tensor in collate_records(cpu_records[start:stop]).items()
            }
            t_validated, t_state, t_options, t_logits, t_greedy, t_value = (
                decode_training(mapping)
            )
            p_validated, p_state, p_options, p_logits, p_greedy, p_value = (
                decode_package(mapping)
            )
            for row, option_count in enumerate(option_counts[start:stop]):
                decision = start + row
                training_action = extract_greedy(t_greedy, row)
                package_action = extract_greedy(p_greedy, row)
                expected = torch.cat(
                    (t_logits[row, :option_count], t_logits[row, -1:])
                ).float()
                actual = torch.cat(
                    (p_logits[row, :option_count], p_logits[row, -1:])
                ).float()
                finite = torch.isfinite(expected) & torch.isfinite(actual)
                difference = (expected[finite] - actual[finite]).abs()
                if difference.numel():
                    root_max = max(root_max, float(difference.max().cpu()))
                    root_sum += float(difference.sum().cpu())
                    root_values += int(difference.numel())
                close = torch.isclose(
                    expected, actual, atol=logits_atol, rtol=logits_rtol
                )
                if not bool(close.all()):
                    root_failures += 1
                if training_action != package_action:
                    greedy_divergences += 1
                    if first_root_divergence is None:
                        first_root_divergence = {
                            "decision": decision,
                            "training_action": training_action,
                            "package_action": package_action,
                            "max_abs_logit_error": (
                                float(difference.max().cpu())
                                if difference.numel() else 0.0
                            ),
                        }
                top_count = min(2, int(expected.numel()))
                training_values, training_indices = expected.topk(top_count)
                package_values, package_indices = actual.topk(top_count)
                if int(training_indices[0]) != int(package_indices[0]):
                    root_top1_divergences += 1
                if set(training_indices.tolist()) != set(package_indices.tolist()):
                    root_top2_set_divergences += 1
                if top_count == 2:
                    minimum_training_margin = min(
                        minimum_training_margin,
                        float((training_values[0] - training_values[1]).cpu()),
                    )
                    minimum_package_margin = min(
                        minimum_package_margin,
                        float((package_values[0] - package_values[1]).cpu()),
                    )

                value_difference = float(
                    (t_value[row].float() - p_value[row].float()).abs().cpu()
                )
                value_max = max(value_max, value_difference)
                value_sum += value_difference
                value_values += 1
                if int(t_value[row].sign()) != int(p_value[row].sign()):
                    value_sign_divergences += 1
                    if first_value_divergence is None:
                        first_value_divergence = {
                            "decision": decision,
                            "training_value": float(t_value[row].cpu()),
                            "package_value": float(p_value[row].cpu()),
                            "absolute_error": value_difference,
                        }

                phantom_roots = (
                    mapping["option_cat"][row, :option_count, 7]
                    .eq(154).nonzero(as_tuple=False).flatten().tolist()
                )
                if not phantom_roots:
                    continue
                target_rows = (
                    p_validated.card_mask[row]
                    & p_validated.card_cat[row, :, 2].eq(2)
                    & p_validated.card_cat[row, :, 3].eq(6)
                ).nonzero(as_tuple=False).flatten().tolist()
                if not 1 <= len(target_rows) <= 8:
                    continue
                ordered = sorted(
                    target_rows,
                    key=lambda index: (
                        int(p_validated.card_cat[row, index, 1]),
                        int(p_validated.card_cat[row, index, 0]),
                    ),
                )
                targets = tuple(
                    dragapult.StableTargetIdentity(
                        1,
                        int(p_validated.card_cat[row, index, 1]) - 1,
                        int(p_validated.card_cat[row, index, 0]),
                        int(p_validated.card_cat[row, index, 4]) - 1,
                    )
                    for index in ordered
                )
                candidates = dragapult.enumerate_allocations(targets)
                visible = []
                for index in ordered:
                    card_id = int(p_validated.card_cat[row, index, 0])
                    visible.append({
                        "serial": int(p_validated.card_cat[row, index, 1]) - 1,
                        "id": card_id,
                        "hp": float(p_validated.card_num[row, index, 0]),
                        "maxHp": float(p_validated.card_num[row, index, 1]),
                        "prize": int(package_policy.prizes[card_id]),
                        "benchSlot": int(p_validated.card_cat[row, index, 4]) - 1,
                        "energyCards": [0] * max(
                            0, int(round(float(p_validated.card_num[row, index, 2])))
                        ),
                        "statusBits": max(
                            0, int(p_validated.card_cat[row, index, 6]) - 1
                        ),
                        "preEvolution": [0] * max(
                            0, int(round(float(p_validated.card_num[row, index, 5])))
                        ),
                    })
                features = package_policy.planner.visible_features(
                    visible, candidates
                ).to(device=device, dtype=p_state.cards.dtype)
                allocation_count = len(candidates)
                feature_batch = features.unsqueeze(0)
                mask = torch.ones(
                    feature_batch.shape[:-1], dtype=torch.bool, device=device
                )
                allocation_mask = torch.ones(
                    (1, allocation_count), dtype=torch.bool, device=device
                )
                for root in phantom_roots:
                    p_targets = p_state.cards[row, ordered]
                    t_targets = t_state.cards[row, ordered]
                    p_target_batch = p_targets.unsqueeze(0).expand(
                        allocation_count, -1, -1
                    ).unsqueeze(0)
                    t_target_batch = t_targets.unsqueeze(0).expand(
                        allocation_count, -1, -1
                    ).unsqueeze(0)
                    package_logits = package_policy.allocation_head(
                        p_state.summary[row].unsqueeze(0),
                        p_options[row, root].unsqueeze(0),
                        p_target_batch,
                        feature_batch,
                        mask,
                        allocation_mask,
                    )[0].float()
                    training_logits = training_model.allocation_head(
                        t_state.summary[row].unsqueeze(0),
                        t_options[row, root].unsqueeze(0),
                        t_target_batch,
                        feature_batch,
                        mask,
                        allocation_mask,
                    )[0].float()
                    allocation_difference = (
                        training_logits - package_logits
                    ).abs()
                    centered_difference = (
                        (training_logits - package_logits)
                        - (training_logits - package_logits).mean()
                    ).abs()
                    probability_difference = (
                        training_logits.softmax(dim=0)
                        - package_logits.softmax(dim=0)
                    ).abs()
                    allocation_max = max(
                        allocation_max, float(allocation_difference.max().cpu())
                    )
                    allocation_centered_max = max(
                        allocation_centered_max,
                        float(centered_difference.max().cpu()),
                    )
                    allocation_probability_max = max(
                        allocation_probability_max,
                        float(probability_difference.max().cpu()),
                    )
                    allocation_sum += float(allocation_difference.sum().cpu())
                    allocation_values += int(allocation_difference.numel())
                    training_top = int(training_logits.argmax())
                    package_top = int(package_logits.argmax())
                    top_count = min(2, allocation_count)
                    training_values, training_indices = training_logits.topk(top_count)
                    package_values, package_indices = package_logits.topk(top_count)
                    if set(training_indices.tolist()) != set(package_indices.tolist()):
                        allocation_top2_set_divergences += 1
                    if top_count == 2:
                        minimum_training_allocation_margin = min(
                            minimum_training_allocation_margin,
                            float((training_values[0] - training_values[1]).cpu()),
                        )
                        minimum_package_allocation_margin = min(
                            minimum_package_allocation_margin,
                            float((package_values[0] - package_values[1]).cpu()),
                        )
                    allocation_decisions += 1
                    if training_top != package_top:
                        allocation_top1_divergences += 1
                        if first_allocation_divergence is None:
                            first_allocation_divergence = {
                                "decision": decision,
                                "root_index": root,
                                "target_count": len(targets),
                                "training_top1": training_top,
                                "package_top1": package_top,
                                "training_counters": list(
                                    candidates[training_top].counters
                                ),
                                "package_counters": list(
                                    candidates[package_top].counters
                                ),
                                "max_abs_logit_error": float(
                                    allocation_difference.max().cpu()
                                ),
                            }

    return {
        "root": {
            "decisions": len(cpu_records),
            "max_abs_logit_error": root_max,
            "mean_abs_logit_error": root_sum / max(1, root_values),
            "tolerance_failures": root_failures,
            "greedy_action_divergences": greedy_divergences,
            "top1_divergences": root_top1_divergences,
            "top2_set_divergences": root_top2_set_divergences,
            "minimum_training_top1_top2_margin": (
                None if minimum_training_margin == float("inf")
                else minimum_training_margin
            ),
            "minimum_package_top1_top2_margin": (
                None if minimum_package_margin == float("inf")
                else minimum_package_margin
            ),
            "first_divergence": first_root_divergence,
            "atol": logits_atol,
            "rtol": logits_rtol,
        },
        "value": {
            "decisions": value_values,
            "max_abs_error": value_max,
            "mean_abs_error": value_sum / max(1, value_values),
            "sign_divergences": value_sign_divergences,
            "first_sign_divergence": first_value_divergence,
            "atol": value_atol,
        },
        "allocation": {
            "decisions": allocation_decisions,
            "max_abs_logit_error": allocation_max,
            "mean_abs_logit_error": allocation_sum / max(1, allocation_values),
            "max_centered_logit_error": allocation_centered_max,
            "max_probability_error": allocation_probability_max,
            "top1_divergences": allocation_top1_divergences,
            "top2_set_divergences": allocation_top2_set_divergences,
            "minimum_training_top1_top2_margin": (
                None if minimum_training_allocation_margin == float("inf")
                else minimum_training_allocation_margin
            ),
            "minimum_package_top1_top2_margin": (
                None if minimum_package_allocation_margin == float("inf")
                else minimum_package_allocation_margin
            ),
            "first_divergence": first_allocation_divergence,
        },
        "passed": (
            root_failures == 0
            and greedy_divergences == 0
            and value_max <= value_atol
            and value_sign_divergences == 0
            and allocation_top1_divergences == 0
        ),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count <= 0 or args.decision_limit <= 0:
        raise ValueError("seed-count and decision-limit must be positive")
    if args.compare_decisions < 0 or args.model_chunk <= 0:
        raise ValueError("compare-decisions cannot be negative and model-chunk must be positive")
    if min(
        args.value_atol,
        args.deployment_logits_atol,
        args.deployment_logits_rtol,
        args.deployment_value_atol,
    ) < 0:
        raise ValueError("numeric parity tolerances cannot be negative")
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
    compound_package = (args.package / "strategy/model.bin").is_file()
    if not args.skip_model and not compound_package and not args.checkpoint.is_file():
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
    record_actors: list[int] = []
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
            record_actors.append(actor)
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
    package_policy = None
    model_report: dict[str, Any] = {"skipped": True, "passed": False}
    focal_record_indices = [
        index for index, actor in enumerate(record_actors) if actor == 0
    ]
    focal_compiled_records = [compiled_records[index] for index in focal_record_indices]
    focal_encoded_records = [encoded_records[index] for index in focal_record_indices]
    focal_option_counts = [option_counts[index] for index in focal_record_indices]
    if not args.skip_model and state_errors == 0:
        if compound_package:
            policy = load_compound_policy(args.package, deck_rows[0], device)
            package_policy = policy
            model_report = compare_compound_model_chunks(
                policy,
                collate_records,
                focal_compiled_records,
                focal_encoded_records,
                focal_option_counts,
                chunk_size=args.model_chunk,
                atol=args.logits_atol,
                rtol=args.logits_rtol,
            )
        elif not mismatch_counts:
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
                    focal_compiled_records,
                    focal_encoded_records,
                    focal_option_counts,
                    chunk_size=args.model_chunk,
                    atol=args.logits_atol,
                    rtol=args.logits_rtol,
                )
            finally:
                package.close()
        for key in ("first_failure", "first_greedy_divergence"):
            detail = model_report.get(key)
            if isinstance(detail, dict) and isinstance(detail.get("decision"), int):
                focal_index = int(detail["decision"])
                detail["focal_decision_index"] = focal_index
                detail["engine_decision"] = focal_record_indices[focal_index]

    value_report: dict[str, Any] = {"skipped": True, "passed": False}
    deployment_report: dict[str, Any] = {"skipped": True, "passed": False}
    if args.training_checkpoint is not None:
        checkpoint = args.training_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        training_deck = deck_rows[0]
        if args.training_deck is not None:
            training_deck = tuple(
                int(line.strip())
                for line in args.training_deck.resolve().read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
            if len(training_deck) != 60 or any(card <= 0 for card in training_deck):
                raise ValueError("--training-deck must contain exactly 60 positive card IDs")
        training_model = load_training_actor_critic(
            checkpoint,
            training_deck,
            device,
            training_project=args.training_project,
        )
        value_report = compare_value_chunks(
            training_model,
            collate_records,
            focal_compiled_records,
            focal_encoded_records,
            chunk_size=args.model_chunk,
            atol=args.value_atol,
        )
        if package_policy is not None:
            deployment_report = compare_training_to_package_chunks(
                training_model,
                package_policy,
                collate_records,
                focal_compiled_records,
                focal_option_counts,
                chunk_size=args.model_chunk,
                logits_atol=args.deployment_logits_atol,
                logits_rtol=args.deployment_logits_rtol,
                value_atol=args.deployment_value_atol,
            )
        detail = value_report.get("first_divergence")
        if isinstance(detail, dict) and isinstance(detail.get("decision"), int):
            focal_index = int(detail["decision"])
            detail["focal_decision_index"] = focal_index
            detail["engine_decision"] = focal_record_indices[focal_index]
        for key in ("maximum_divergence",):
            detail = value_report.get(key)
            if isinstance(detail, dict) and isinstance(detail.get("decision"), int):
                focal_index = int(detail["decision"])
                detail["focal_decision_index"] = focal_index
                detail["engine_decision"] = focal_record_indices[focal_index]
        for detail in value_report.get("sign_divergence_samples", []):
            if isinstance(detail, dict) and isinstance(detail.get("decision"), int):
                focal_index = int(detail["decision"])
                detail["focal_decision_index"] = focal_index
                detail["engine_decision"] = focal_record_indices[focal_index]

    tensor_passed = not mismatch_counts and state_errors == 0 and replayed == requested
    coverage_complete = requested == len(trace) and replayed == len(trace)
    history_wrap_passed = not args.require_history_wrap or wrapped_history_decisions > 0
    full_passed = (
        tensor_passed
        and coverage_complete
        and history_wrap_passed
        and not args.skip_model
        and bool(model_report.get("passed"))
        and (args.training_checkpoint is None or bool(value_report.get("passed")))
        and (
            not compound_package
            or args.training_checkpoint is None
            or bool(deployment_report.get("passed"))
        )
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
            "option skill/effect prototype relations are compared as exact tensor fields; "
            "a CUDA zero placeholder is a preprocessing mismatch, not a shared-adapter injection"
        ),
        "tensor_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "tensor_mismatch_details": mismatch_details,
        "integer_mask_relation_exact": tensor_passed,
        "model": model_report,
        "value_model": value_report,
        "training_to_package": deployment_report,
        "model_focal_actor": 0,
        "model_focal_decisions": len(focal_record_indices),
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

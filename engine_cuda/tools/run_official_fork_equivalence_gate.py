"""Check CUDA scratch fork equivalence and source isolation.

This is the narrow gate for the search design:

    fork(s) == s

where equality excludes the semantic-history pointer value that must be rebound
inside the destination arena. Hidden-belief redeterminization is intentionally
out of scope here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.native import create_official_engine, load_extension  # noqa: E402


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description="Verify fork(s)==s for CUDA official scratch lanes."
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--deck0",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "fixtures"
        / "0022_deck40"
        / "dragapult_third_7c605fb2"
        / "deck.csv",
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "fixtures"
        / "0022_deck40"
        / "marnie_haggle_be5107ce"
        / "deck.csv",
    )
    parser.add_argument("--source-batch", type=int, default=16)
    parser.add_argument("--scratch-batch", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=61001)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_deck(path: Path) -> list[int]:
    result: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        token = raw_line.split(",", 1)[0].strip()
        if not token:
            continue
        try:
            result.append(int(token))
        except ValueError as error:
            raise ValueError(f"invalid card ID at {path}:{line_number}") from error
    if len(result) != 60:
        raise ValueError(f"deck must contain exactly 60 card IDs: {path}")
    return result


def clone_tensor_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in values.items()}


def tensor_dict_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.keys() == right.keys() and all(
        left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        and bool(left[name].eq(right[name]).all().item())
        for name in left
    )


def hash_tensor(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def zero_rebound_pointer(state_bytes: Any) -> Any:
    extension = load_extension()
    pointer_offset = int(extension.OFFICIAL_SEMANTIC_HISTORY_POINTER_OFFSET)
    pointer_bytes = int(extension.OFFICIAL_POINTER_BYTES)
    result = state_bytes.detach().clone()
    result[:, pointer_offset : pointer_offset + pointer_bytes] = 0
    return result


def main() -> None:
    args = parse_args()
    if args.source_batch <= 0 or args.scratch_batch <= 0:
        raise SystemExit("batch sizes must be positive")
    for path in (args.rules, args.deck0, args.deck1):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)

    rule_pack = args.rules.read_bytes()
    source = create_official_engine(
        rule_pack, batch_size=args.source_batch, device_index=args.device_index
    )
    scratch = create_official_engine(
        rule_pack, batch_size=args.scratch_batch, device_index=args.device_index
    )
    deck0 = read_deck(args.deck0)
    deck1 = read_deck(args.deck1)
    decks = torch.tensor([[deck0, deck1]] * args.source_batch, dtype=torch.int32, device=device)
    seeds = torch.arange(
        args.seed_start,
        args.seed_start + args.source_batch,
        dtype=torch.int64,
        device=device,
    )
    source.reset_seeded_first_min_semantic(decks, seeds)
    torch.cuda.synchronize()

    source_indices = (
        torch.arange(args.scratch_batch, dtype=torch.int64, device=device)
        % args.source_batch
    ).contiguous()
    state_before = source.state_bytes().detach().clone()
    rng_before = source.rng_bytes().detach().clone()
    status_before = source.statuses().detach().clone()
    actions_before = source.action_bytes().detach().clone()
    history_before = clone_tensor_dict(source.semantic_history_raw())
    policy_before = clone_tensor_dict(dict(source.encode_policy_v1()))
    semantic_before = clone_tensor_dict(
        dict(source.encode_semantic0031_v2_lanes(torch.arange(args.source_batch, dtype=torch.int64, device=device)))
    )

    source.fork_lanes_to(scratch, source_indices)
    torch.cuda.synchronize()

    expected_state = zero_rebound_pointer(state_before.index_select(0, source_indices))
    actual_state = zero_rebound_pointer(scratch.state_bytes())
    expected_history = {
        name: value.index_select(0, source_indices) for name, value in history_before.items()
    }
    scratch_history = clone_tensor_dict(scratch.semantic_history_raw())
    source_lanes = torch.arange(args.source_batch, dtype=torch.int64, device=device)
    scratch_lanes = torch.arange(args.scratch_batch, dtype=torch.int64, device=device)
    scratch_policy = clone_tensor_dict(dict(scratch.encode_policy_v1()))
    scratch_semantic = clone_tensor_dict(dict(scratch.encode_semantic0031_v2_lanes(scratch_lanes)))

    expected_policy = {
        name: value.index_select(0, source_indices) for name, value in policy_before.items()
    }
    expected_semantic = {
        name: value.index_select(0, source_indices) for name, value in semantic_before.items()
    }

    checks = {
        "state_equal_ignoring_rebound_pointer": bool(torch.equal(expected_state, actual_state)),
        "rng_equal": bool(torch.equal(rng_before.index_select(0, source_indices), scratch.rng_bytes())),
        "status_equal": bool(torch.equal(status_before.index_select(0, source_indices), scratch.statuses())),
        "action_bytes_equal": bool(torch.equal(actions_before.index_select(0, source_indices), scratch.action_bytes())),
        "semantic_history_equal": tensor_dict_equal(expected_history, scratch_history),
        "policy_codec_equal": tensor_dict_equal(expected_policy, scratch_policy),
        "semantic0031_equal": tensor_dict_equal(expected_semantic, scratch_semantic),
        "source_state_unchanged_after_fork": bool(torch.equal(state_before, source.state_bytes())),
        "source_rng_unchanged_after_fork": bool(torch.equal(rng_before, source.rng_bytes())),
        "source_status_unchanged_after_fork": bool(torch.equal(status_before, source.statuses())),
        "source_actions_unchanged_after_fork": bool(torch.equal(actions_before, source.action_bytes())),
        "source_history_unchanged_after_fork": tensor_dict_equal(history_before, source.semantic_history_raw()),
    }
    passed = all(checks.values())
    report = {
        "schema_version": "official_cuda_fork_equivalence_gate_v1",
        "passed": passed,
        "contract": "fork(s) == s, excluding destination semantic-history pointer rebinding",
        "source_batch": args.source_batch,
        "scratch_batch": args.scratch_batch,
        "seed_start": args.seed_start,
        "source_indices": source_indices.detach().cpu().tolist(),
        "source_state_sha256": hash_tensor(zero_rebound_pointer(state_before)),
        "scratch_state_sha256": hash_tensor(actual_state),
        "checks": checks,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

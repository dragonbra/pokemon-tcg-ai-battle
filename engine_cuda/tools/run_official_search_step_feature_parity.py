"""Check CUDA scratch-search feature parity against direct step.

The contract under test is:

    featurize(search(s, a)) == featurize(step(s, a))

for CUDA learner-visible encoders. In this engine branch, search means
forking a main rollout lane into an isolated scratch arena and applying the
same packed legal action there. This script intentionally does not
redeterminize hidden state or future RNG; random outcomes therefore use the
same copied MT19937 state and should be bitwise identical. Separate hidden
particle invariance is covered by run_official_scratch_fork_smoke.py.
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


STATUS_NEEDS_ACTION = 1
STATUS_ERROR = 3


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description=(
            "Fork ready CUDA lanes to scratch, apply the same legal action in "
            "scratch and main, then compare learner-visible featurizers."
        )
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
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=51001)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("first", "last", "spread"),
        default="spread",
        help=(
            "How to choose a deterministic legal option slice when min_count > 0. "
            "All modes use exactly min_count options."
        ),
    )
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


def hash_tensor(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def clone_tensor_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in values.items()}


def hash_tensor_dict(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("ascii"))
        digest.update(hash_tensor(values[name]).encode("ascii"))
    return digest.hexdigest()


def first_tensor_mismatch(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    if left.keys() != right.keys():
        return {
            "kind": "keys",
            "left_only": sorted(set(left) - set(right)),
            "right_only": sorted(set(right) - set(left)),
        }
    for name in sorted(left):
        a = left[name]
        b = right[name]
        if tuple(a.shape) != tuple(b.shape) or str(a.dtype) != str(b.dtype):
            return {
                "kind": "metadata",
                "name": name,
                "left_shape": list(a.shape),
                "right_shape": list(b.shape),
                "left_dtype": str(a.dtype),
                "right_dtype": str(b.dtype),
            }
        neq = a.ne(b)
        if bool(neq.any().item()):
            index = neq.nonzero(as_tuple=False)[0].detach().cpu().tolist()
            return {
                "kind": "value",
                "name": name,
                "index": index,
                "left": a[tuple(index)].detach().cpu().item(),
                "right": b[tuple(index)].detach().cpu().item(),
            }
    return None


def select_legal_actions(encoded: dict[str, Any], statuses: Any, mode: str) -> tuple[Any, Any]:
    import torch

    option_mask = encoded["option_mask"]
    min_count = encoded["min_count"].to(dtype=torch.int64).contiguous()
    max_count = encoded["max_count"].to(dtype=torch.int64).contiguous()
    batch = int(option_mask.shape[0])
    counts_cpu = min_count.detach().cpu().tolist()
    status_cpu = statuses.detach().cpu().tolist()
    option_cpu = option_mask.detach().cpu()

    effective_counts: list[int] = []
    selected_rows: list[list[int]] = []
    for lane in range(batch):
        if int(status_cpu[lane]) != STATUS_NEEDS_ACTION:
            effective_counts.append(0)
            selected_rows.append([])
            continue
        count = int(counts_cpu[lane])
        maximum = int(max_count[lane].detach().cpu().item())
        legal = [int(i) for i, value in enumerate(option_cpu[lane].tolist()) if int(value) != 0]
        if count < 0 or maximum < count or count > len(legal):
            raise RuntimeError(
                f"invalid legal action cardinality at lane {lane}: "
                f"min={count} max={maximum} legal={len(legal)}"
            )
        if count == 0:
            chosen: list[int] = []
        elif mode == "first":
            chosen = legal[:count]
        elif mode == "last":
            chosen = legal[-count:]
        else:
            start_limit = max(0, len(legal) - count)
            start = lane % (start_limit + 1)
            chosen = legal[start : start + count]
        effective_counts.append(count)
        selected_rows.append(chosen)

    width = max(1, max(effective_counts, default=0))
    actions = torch.full(
        (batch, width),
        -1,
        dtype=torch.int64,
        device=option_mask.device,
    )
    for lane, chosen in enumerate(selected_rows):
        if chosen:
            actions[lane, : len(chosen)] = torch.tensor(
                chosen,
                dtype=torch.int64,
                device=option_mask.device,
            )
    counts = torch.tensor(effective_counts, dtype=torch.int64, device=option_mask.device)
    return actions.contiguous(), counts.contiguous()


def zero_pointer_bytes(state_bytes: Any) -> Any:
    extension = load_extension()
    pointer_offset = int(extension.OFFICIAL_SEMANTIC_HISTORY_POINTER_OFFSET)
    pointer_bytes = int(extension.OFFICIAL_POINTER_BYTES)
    result = state_bytes.detach().clone()
    result[:, pointer_offset : pointer_offset + pointer_bytes] = 0
    return result


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    for path in (args.rules, args.deck0, args.deck1):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)

    rule_pack = args.rules.read_bytes()
    direct = create_official_engine(rule_pack, batch_size=args.batch, device_index=args.device_index)
    scratch = create_official_engine(rule_pack, batch_size=args.batch, device_index=args.device_index)
    deck0 = read_deck(args.deck0)
    deck1 = read_deck(args.deck1)
    decks = torch.tensor([[deck0, deck1]] * args.batch, dtype=torch.int32, device=device)
    seeds = torch.arange(
        int(args.seed_start),
        int(args.seed_start) + args.batch,
        dtype=torch.int64,
        device=device,
    )
    lanes = torch.arange(args.batch, dtype=torch.int64, device=device)
    direct.reset_seeded_first_min_semantic(decks, seeds)
    torch.cuda.synchronize()

    records: list[dict[str, Any]] = []
    checks = {
        "policy_codec_equal": True,
        "semantic0031_equal": True,
        "state_equal_ignoring_rebound_pointer": True,
        "statuses_equal": True,
        "no_error_status": True,
    }
    first_mismatch: dict[str, Any] | None = None

    for step in range(args.steps):
        root_status = direct.statuses().detach().clone()
        ready_count = int(root_status.eq(STATUS_NEEDS_ACTION).sum().item())
        error_count = int(root_status.eq(STATUS_ERROR).sum().item())
        if error_count:
            checks["no_error_status"] = False
            first_mismatch = first_mismatch or {
                "step": step,
                "kind": "root_error_status",
                "error_count": error_count,
            }
            break
        if ready_count == 0:
            records.append(
                {
                    "step": step,
                    "ready": 0,
                    "terminal_or_idle": int(args.batch),
                    "stopped": "no_ready_lanes",
                }
            )
            break

        direct.fork_lanes_to(scratch, lanes)
        torch.cuda.synchronize()

        before_policy = dict(direct.encode_policy_v1())
        actions, counts = select_legal_actions(before_policy, root_status, args.action_mode)
        direct.pack_actions(actions, counts)
        scratch.pack_actions(actions, counts)
        direct.apply_packed_actions()
        scratch.apply_packed_actions()
        direct.advance_to_decision()
        scratch.advance_to_decision()
        torch.cuda.synchronize()

        direct_status = direct.statuses().detach().clone()
        scratch_status = scratch.statuses().detach().clone()
        status_equal = bool(torch.equal(direct_status, scratch_status))
        checks["statuses_equal"] = checks["statuses_equal"] and status_equal
        post_error_count = int(direct_status.eq(STATUS_ERROR).sum().item())
        if post_error_count:
            checks["no_error_status"] = False

        direct_policy = clone_tensor_dict(dict(direct.encode_policy_v1()))
        scratch_policy = clone_tensor_dict(dict(scratch.encode_policy_v1()))
        policy_mismatch = first_tensor_mismatch(direct_policy, scratch_policy)
        policy_equal = policy_mismatch is None
        checks["policy_codec_equal"] = checks["policy_codec_equal"] and policy_equal

        direct_semantic = clone_tensor_dict(dict(direct.encode_semantic0031_v2_lanes(lanes)))
        scratch_semantic = clone_tensor_dict(dict(scratch.encode_semantic0031_v2_lanes(lanes)))
        semantic_mismatch = first_tensor_mismatch(direct_semantic, scratch_semantic)
        semantic_equal = semantic_mismatch is None
        checks["semantic0031_equal"] = checks["semantic0031_equal"] and semantic_equal

        direct_state = zero_pointer_bytes(direct.state_bytes())
        scratch_state = zero_pointer_bytes(scratch.state_bytes())
        state_equal = bool(torch.equal(direct_state, scratch_state))
        checks["state_equal_ignoring_rebound_pointer"] = (
            checks["state_equal_ignoring_rebound_pointer"] and state_equal
        )

        record = {
            "step": step,
            "ready": ready_count,
            "action_count_min": int(counts.min().item()),
            "action_count_max": int(counts.max().item()),
            "direct_status_counts": {
                str(value): int(direct_status.eq(value).sum().item()) for value in range(4)
            },
            "policy_sha256": hash_tensor_dict(direct_policy),
            "semantic0031_sha256": hash_tensor_dict(direct_semantic),
            "policy_equal": policy_equal,
            "semantic0031_equal": semantic_equal,
            "state_equal_ignoring_rebound_pointer": state_equal,
            "statuses_equal": status_equal,
        }
        records.append(record)

        if not status_equal and first_mismatch is None:
            first_mismatch = {
                "step": step,
                "kind": "status",
                "direct": direct_status.detach().cpu().tolist(),
                "scratch": scratch_status.detach().cpu().tolist(),
            }
        if policy_mismatch is not None and first_mismatch is None:
            first_mismatch = {"step": step, "encoder": "encode_policy_v1", **policy_mismatch}
        if semantic_mismatch is not None and first_mismatch is None:
            first_mismatch = {"step": step, "encoder": "encode_semantic0031_v2_lanes", **semantic_mismatch}
        if not state_equal and first_mismatch is None:
            first_mismatch = {
                "step": step,
                "kind": "state_bytes",
                "direct_sha256": hash_tensor(direct_state),
                "scratch_sha256": hash_tensor(scratch_state),
            }
        if post_error_count:
            first_mismatch = first_mismatch or {
                "step": step,
                "kind": "post_step_error_status",
                "error_count": post_error_count,
            }
            break
        if first_mismatch is not None:
            break

    passed = bool(records) and all(checks.values()) and first_mismatch is None
    report = {
        "schema_version": "official_cuda_search_step_feature_parity_v1",
        "passed": passed,
        "contract": "featurize(fork_to_scratch_then_step(s,a)) == featurize(direct_step(s,a))",
        "random_hidden_scope": (
            "No redeterminization is applied in this parity gate; copied RNG and "
            "hidden state should therefore match bitwise except semantic-history pointer rebinding."
        ),
        "rules": str(args.rules.resolve()),
        "deck0": str(args.deck0.resolve()),
        "deck1": str(args.deck1.resolve()),
        "batch": args.batch,
        "steps_requested": args.steps,
        "steps_checked": len(records),
        "seed_start": args.seed_start,
        "action_mode": args.action_mode,
        "checks": checks,
        "first_mismatch": first_mismatch,
        "records": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

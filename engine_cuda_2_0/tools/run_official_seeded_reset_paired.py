from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402


FIXTURE_HEADER = struct.Struct("<8sIIIII36x")
FIXTURE_MAGIC = b"PTCGSR01"
FIXTURE_VERSION = 1
RECORD_PREFIX_BYTES = 64
STATE_ABI_VERSION = 6
STATE_BYTES = 119_936


@dataclass(frozen=True)
class SeededFixture:
    seeds: list[int]
    statuses: list[int]
    states: bytes
    record_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare official seeded-reset fixtures with the CUDA resident "
            "first-min reset, including dtype and masked-lane contracts."
        )
    )
    private_root = (
        CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    )
    parser.add_argument("--rules", type=Path, default=private_root / "official_rules.bin")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=private_root / "official_seeded_reset_marnie_alakazam_s1_10.bin",
    )
    parser.add_argument(
        "--deck0",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT.parent
            / "bc_models"
            / "agent_marnie_prize_control_v4_3121746f_20260728"
            / "deck.csv"
        ),
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT.parent
            / "bc_models"
            / "agent_yushin_idonly_bc_v1_20260723"
            / "deck.csv"
        ),
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--masked-seed-offset", type=int, default=1_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_seeded_reset_cuda_paired.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def read_fixture(path: Path) -> SeededFixture:
    data = path.read_bytes()
    if len(data) < FIXTURE_HEADER.size:
        raise ValueError("official seeded-reset fixture is truncated")
    magic, version, state_abi, state_bytes, record_bytes, record_count = (
        FIXTURE_HEADER.unpack_from(data)
    )
    if magic != FIXTURE_MAGIC or version != FIXTURE_VERSION:
        raise ValueError("official seeded-reset fixture magic/version mismatch")
    if state_abi != STATE_ABI_VERSION or state_bytes != STATE_BYTES:
        raise ValueError("official seeded-reset fixture state ABI mismatch")
    if record_bytes < RECORD_PREFIX_BYTES + STATE_BYTES or record_count <= 0:
        raise ValueError("official seeded-reset fixture record layout mismatch")
    expected_size = FIXTURE_HEADER.size + record_bytes * record_count
    if len(data) != expected_size:
        raise ValueError(
            f"official seeded-reset fixture size mismatch: {len(data)} != {expected_size}"
        )

    seeds: list[int] = []
    statuses: list[int] = []
    states = bytearray(record_count * STATE_BYTES)
    for index in range(record_count):
        record_offset = FIXTURE_HEADER.size + index * record_bytes
        seeds.append(struct.unpack_from("<Q", data, record_offset)[0])
        statuses.append(data[record_offset + 8])
        state_offset = record_offset + RECORD_PREFIX_BYTES
        output_offset = index * STATE_BYTES
        states[output_offset : output_offset + STATE_BYTES] = data[
            state_offset : state_offset + STATE_BYTES
        ]
    return SeededFixture(seeds, statuses, bytes(states), record_bytes)


def first_mismatch(expected: bytes, actual: bytes) -> dict[str, int] | None:
    if len(expected) != len(actual):
        return {"expected_bytes": len(expected), "actual_bytes": len(actual)}
    for offset, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if left != right:
            return {"byte_offset": offset, "expected": left, "actual": right}
    return None


def tensor_bytes(tensor: "Any") -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes(order="C")


def reset_capture(
    engine: "Any",
    decks: "Any",
    seeds: "Any",
) -> tuple[bytes, list[int]]:
    engine.reset_seeded_first_min(decks, seeds)
    return tensor_bytes(engine.state_bytes()), [
        int(value) for value in engine.statuses().detach().cpu().tolist()
    ]


def main() -> None:
    args = parse_args()
    rules = args.rules.resolve()
    fixture_path = args.fixture.resolve()
    deck_paths = [args.deck0.resolve(), args.deck1.resolve()]
    for path in [rules, fixture_path, *deck_paths]:
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    fixture = read_fixture(fixture_path)
    if args.masked_seed_offset == 0:
        raise SystemExit("--masked-seed-offset must be non-zero")

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if int(_ptcg_cuda.OFFICIAL_STATE_ABI_VERSION) != STATE_ABI_VERSION:
        raise RuntimeError("official CUDA extension state ABI mismatch")
    if int(_ptcg_cuda.OFFICIAL_STATE_BYTES) != STATE_BYTES:
        raise RuntimeError("official CUDA extension state size mismatch")
    if str(_ptcg_cuda.OFFICIAL_SEEDED_SETUP_POLICY) != "first_min_v1":
        raise RuntimeError("official CUDA extension seeded setup policy mismatch")

    batch = len(fixture.seeds)
    device = torch.device("cuda", args.device_index)
    deck_rows = [read_deck(path) for path in deck_paths]
    decks_i32 = torch.tensor(
        [deck_rows for _ in range(batch)], dtype=torch.int32, device=device
    )
    decks_i64 = decks_i32.to(torch.int64)
    seeds = torch.tensor(fixture.seeds, dtype=torch.int64, device=device)
    expected_statuses = fixture.statuses

    engine = create_official_engine(
        rules.read_bytes(), batch_size=batch, device_index=args.device_index
    )
    state_i32, status_i32 = reset_capture(engine, decks_i32, seeds)
    state_i64, status_i64 = reset_capture(engine, decks_i64, seeds)

    mutated_seeds = seeds + int(args.masked_seed_offset)
    mutated_states, mutated_statuses = reset_capture(engine, decks_i32, mutated_seeds)
    base_states, base_statuses = reset_capture(engine, decks_i32, seeds)

    lane_mask = torch.tensor(
        [(index % 2) == 0 for index in range(batch)],
        dtype=torch.bool,
        device=device,
    )
    engine.reset_seeded_first_min_masked(decks_i64, mutated_seeds, lane_mask)
    masked_states = engine.state_bytes().detach().contiguous().cpu()
    masked_statuses = [
        int(value) for value in engine.statuses().detach().cpu().tolist()
    ]
    base_state_tensor = torch.frombuffer(bytearray(base_states), dtype=torch.uint8).view(
        batch, STATE_BYTES
    )
    mutated_state_tensor = torch.frombuffer(
        bytearray(mutated_states), dtype=torch.uint8
    ).view(batch, STATE_BYTES)

    selected_lane_matches = []
    unselected_lane_unchanged = []
    masked_status_matches = []
    for index in range(batch):
        if bool(lane_mask[index].item()):
            selected_lane_matches.append(
                bool(torch.equal(masked_states[index], mutated_state_tensor[index]))
            )
            masked_status_matches.append(masked_statuses[index] == mutated_statuses[index])
        else:
            unselected_lane_unchanged.append(
                bool(torch.equal(masked_states[index], base_state_tensor[index]))
            )
            masked_status_matches.append(masked_statuses[index] == base_statuses[index])

    fixture_state_mismatch = first_mismatch(fixture.states, state_i32)
    dtype_state_mismatch = first_mismatch(state_i32, state_i64)
    state_mismatch_lanes = sum(
        fixture.states[index * STATE_BYTES : (index + 1) * STATE_BYTES]
        != state_i32[index * STATE_BYTES : (index + 1) * STATE_BYTES]
        for index in range(batch)
    )
    status_mismatch_lanes = sum(
        expected != actual
        for expected, actual in zip(expected_statuses, status_i32, strict=True)
    )
    passed = (
        fixture_state_mismatch is None
        and status_i32 == expected_statuses
        and dtype_state_mismatch is None
        and status_i64 == status_i32
        and all(selected_lane_matches)
        and all(unselected_lane_unchanged)
        and all(masked_status_matches)
    )
    result: dict[str, Any] = {
        "passed": passed,
        "contract": "official_seeded_reset_cuda_paired_v1",
        "setup_policy": str(_ptcg_cuda.OFFICIAL_SEEDED_SETUP_POLICY),
        "batch": batch,
        "seed_min": min(fixture.seeds),
        "seed_max": max(fixture.seeds),
        "state_abi": STATE_ABI_VERSION,
        "state_bytes": STATE_BYTES,
        "record_bytes": fixture.record_bytes,
        "state_mismatch_lanes": state_mismatch_lanes,
        "status_mismatch_lanes": status_mismatch_lanes,
        "first_fixture_state_mismatch": fixture_state_mismatch,
        "int32_int64_state_equal": dtype_state_mismatch is None,
        "int32_int64_status_equal": status_i64 == status_i32,
        "first_dtype_state_mismatch": dtype_state_mismatch,
        "masked_selected_lane_matches": all(selected_lane_matches),
        "masked_unselected_lane_unchanged": all(unselected_lane_unchanged),
        "masked_status_matches": all(masked_status_matches),
        "masked_selected_lanes": len(selected_lane_matches),
        "masked_unselected_lanes": len(unselected_lane_unchanged),
        "masked_seed_offset": args.masked_seed_offset,
        "expected_statuses": expected_statuses,
        "cuda_statuses": status_i32,
        "arena_allocated_bytes": int(engine.allocated_bytes),
        "device": torch.cuda.get_device_name(args.device_index),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "rule_pack_sha256": sha256_file(rules),
        "deck_sha256": [sha256_file(path) for path in deck_paths],
        "private_fixture_sha256": sha256_file(fixture_path),
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

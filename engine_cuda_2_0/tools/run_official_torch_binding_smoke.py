from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402


STATE_ABI_VERSION = 6
STATE_BYTES = 119_936
ACTION_BYTES = 272
ACTION_COUNT_OFFSET = 256
INVALID_ACTION_COUNT = 129


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise the official CUDA runtime through its PyTorch binding."
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT
            / "generated"
            / "private"
            / "official_3aaeaa92"
            / "official_rules.bin"
        ),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT
            / "generated"
            / "private"
            / "official_3aaeaa92"
            / "official_main_fixture.bin"
        ),
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_torch_binding_smoke.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tensor(tensor: "Any") -> str:
    cpu = tensor.detach().contiguous().cpu()
    return hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest()


def first_main_expected_state(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 64:
        raise ValueError("official main fixture is truncated")
    if data[:8] != b"PTCGMAIN":
        raise ValueError("official main fixture magic mismatch")
    state_abi, state_bytes, record_bytes, record_count = struct.unpack_from(
        "<IIII", data, 12
    )
    if state_abi != STATE_ABI_VERSION or state_bytes != STATE_BYTES:
        raise ValueError("official main fixture ABI mismatch")
    if record_count <= 0 or record_bytes < 2 * STATE_BYTES:
        raise ValueError("official main fixture record layout mismatch")
    record_offset = 64
    # OfficialMainFixtureRecord has 264 bytes of scalar/action metadata before
    # the first alignas(64) state field.
    input_offset = record_offset + 320
    expected_offset = input_offset + STATE_BYTES
    if expected_offset + STATE_BYTES > len(data):
        raise ValueError("official main fixture first record is truncated")
    return data[expected_offset : expected_offset + STATE_BYTES]


def main() -> None:
    args = parse_args()
    if args.batch != 4:
        raise SystemExit("this fixed action-packing smoke requires --batch 4")
    rules = args.rules.resolve()
    if not rules.is_file():
        raise SystemExit(f"official rule pack does not exist: {rules}")
    fixture = args.fixture.resolve()
    if not fixture.is_file():
        raise SystemExit(f"official main fixture does not exist: {fixture}")

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if int(_ptcg_cuda.OFFICIAL_STATE_ABI_VERSION) != STATE_ABI_VERSION:
        raise RuntimeError("official state ABI mismatch")
    if int(_ptcg_cuda.OFFICIAL_STATE_BYTES) != STATE_BYTES:
        raise RuntimeError("official state byte size mismatch")
    if int(_ptcg_cuda.OFFICIAL_ACTION_BYTES) != ACTION_BYTES:
        raise RuntimeError("official action byte size mismatch")

    engine = create_official_engine(
        rules.read_bytes(),
        batch_size=args.batch,
        device_index=args.device_index,
    )
    fixture_state = first_main_expected_state(fixture)
    engine.reset_states(torch.frombuffer(bytearray(fixture_state * args.batch), dtype=torch.uint8))
    codec = engine.encode_policy_v1()
    expected_codec_shapes = {
        "global_cat": (args.batch, 8),
        "global_num": (args.batch, 16),
        "entity_cat": (args.batch, 128, 6),
        "entity_num": (args.batch, 128, 10),
        "entity_parent": (args.batch, 128),
        "entity_mask": (args.batch, 128),
        "option_cat": (args.batch, 80, 12),
        "option_num": (args.batch, 80, 4),
        "option_equiv": (args.batch, 80),
        "option_mask": (args.batch, 80),
        "min_count": (args.batch,),
        "max_count": (args.batch,),
    }
    for name, shape in expected_codec_shapes.items():
        tensor = codec[name]
        if tensor.device.type != "cuda" or tuple(tensor.shape) != shape:
            raise RuntimeError(f"invalid official codec tensor {name}: {tuple(tensor.shape)}")
    codec_option_counts = codec["option_mask"].sum(dim=1).clone()
    codec_entity_counts = codec["entity_mask"].sum(dim=1).clone()
    codec_cpu = {
        name: tensor.detach().contiguous().cpu()
        for name, tensor in codec.items()
    }
    codec_tensor_sha256 = {
        name: sha256_tensor(tensor)
        for name, tensor in codec.items()
    }
    codec_row_consistency = {
        name: all(
            bool(torch.equal(codec_cpu[name][0], codec_cpu[name][env]))
            for env in range(1, args.batch)
        )
        for name in codec_cpu
    }
    first_option_count = int(codec_option_counts[0].item())
    first_entity_count = int(codec_entity_counts[0].item())
    codec_sample = {
        "global_cat": [int(value) for value in codec_cpu["global_cat"][0].tolist()],
        "global_num": [float(value) for value in codec_cpu["global_num"][0].tolist()],
        "entity_cat_head": codec_cpu["entity_cat"][
            0, : min(3, first_entity_count)
        ].tolist(),
        "entity_num_head": codec_cpu["entity_num"][
            0, : min(3, first_entity_count)
        ].tolist(),
        "entity_parent_head": codec_cpu["entity_parent"][
            0, : min(8, first_entity_count)
        ].tolist(),
        "option_cat_head": codec_cpu["option_cat"][
            0, : min(8, first_option_count)
        ].tolist(),
        "option_num_head": codec_cpu["option_num"][
            0, : min(8, first_option_count)
        ].tolist(),
        "option_equiv_head": codec_cpu["option_equiv"][
            0, : min(16, first_option_count)
        ].tolist(),
    }

    raw_states = bytearray(args.batch * STATE_BYTES)
    for env in range(args.batch):
        struct.pack_into("<I", raw_states, env * STATE_BYTES, STATE_ABI_VERSION)
    engine.reset_states(torch.frombuffer(raw_states, dtype=torch.uint8))

    state_view = engine.state_bytes()
    status_view = engine.statuses()
    action_view = engine.action_bytes()
    if state_view.device.type != "cuda" or tuple(state_view.shape) != (
        args.batch,
        STATE_BYTES,
    ):
        raise RuntimeError("invalid official state device view")
    if status_view.device.type != "cuda" or tuple(status_view.shape) != (args.batch,):
        raise RuntimeError("invalid official status device view")
    if action_view.device.type != "cuda" or tuple(action_view.shape) != (
        args.batch,
        ACTION_BYTES,
    ):
        raise RuntimeError("invalid official action device view")

    # This block is the intended asynchronous hot-path contract.  The only
    # synchronization and host reads happen after all operations are queued.
    engine.classify()
    classified = status_view.clone()
    option_indices = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [2, 3, 0, 0], [0, 1, 2, 3]],
        dtype=torch.int64,
        device=state_view.device,
    )
    counts = torch.tensor([0, 1, 2, 5], dtype=torch.int64, device=state_view.device)
    engine.pack_actions(option_indices, counts)
    packed_counts = action_view[:, ACTION_COUNT_OFFSET].to(torch.int32)
    packed_counts = packed_counts + 256 * action_view[:, ACTION_COUNT_OFFSET + 1].to(
        torch.int32
    )
    engine.apply_actions(action_view.clone())
    applied = status_view.clone()
    torch.cuda.synchronize(args.device_index)

    codec_option_counts_host = [int(value) for value in codec_option_counts.cpu().tolist()]
    codec_entity_counts_host = [int(value) for value in codec_entity_counts.cpu().tolist()]
    classified_host = classified.cpu().tolist()
    packed_counts_host = packed_counts.cpu().tolist()
    applied_host = applied.cpu().tolist()
    passed = (
        classified_host == [0, 0, 0, 0]
        and all(value > 0 for value in codec_option_counts_host)
        and all(value > 0 for value in codec_entity_counts_host)
        and all(codec_row_consistency.values())
        and packed_counts_host == [0, 1, 2, INVALID_ACTION_COUNT]
        and applied_host == [3, 3, 3, 3]
    )
    result: dict[str, Any] = {
        "passed": passed,
        "contract": "official_torch_binding_smoke_v1",
        "batch": args.batch,
        "state_abi": STATE_ABI_VERSION,
        "state_bytes": STATE_BYTES,
        "action_bytes": ACTION_BYTES,
        "classified_statuses": classified_host,
        "codec_entity_counts": codec_entity_counts_host,
        "codec_option_counts": codec_option_counts_host,
        "codec_row_consistency": codec_row_consistency,
        "codec_tensor_sha256": codec_tensor_sha256,
        "codec_sample": codec_sample,
        "packed_action_counts": packed_counts_host,
        "post_apply_statuses": applied_host,
        "arena_allocated_bytes": engine.allocated_bytes,
        "device": torch.cuda.get_device_name(args.device_index),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "rule_pack_sha256": sha256_file(rules),
        "private_main_fixture_sha256": sha256_file(fixture),
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        "hot_path_contract": (
            "classify/pack/apply are queued on the current CUDA stream; "
            "host synchronization is used only for final smoke inspection"
        ),
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

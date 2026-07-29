from __future__ import annotations

import importlib
import struct
from typing import Iterable

from .reference import ReferenceResetSpec
from .schema import RulePack


RESET_STRUCT = struct.Struct("<I2H2H2H2H2HH6x")


def pack_reset_specs(specs: Iterable[ReferenceResetSpec]) -> bytes:
    rows = []
    for spec in specs:
        rows.append(
            RESET_STRUCT.pack(
                spec.seed & 0xFFFFFFFF,
                *spec.active_card_ids,
                *spec.active_hp,
                *spec.deck_count,
                *spec.hand_count,
                *spec.policy_ids,
                0,
            )
        )
    return b"".join(rows)


def load_extension():
    try:
        return importlib.import_module("_ptcg_cuda")
    except ImportError as error:
        raise RuntimeError(
            "failed to import _ptcg_cuda; build with "
            "-DPTCG_CUDA_BUILD_TORCH=ON in a matching PyTorch/CUDA environment. "
            f"Loader error: {error}"
        ) from error


def create_engine(
    rule_pack: RulePack,
    *,
    batch_size: int,
    policy_count: int,
    route_capacity: int,
    device_index: int = 0,
):
    """Create the optional PyTorch-bound engine and upload a rule pack once."""

    import torch

    extension = load_extension()
    engine = extension.CudaEngine(batch_size, policy_count, route_capacity, device_index)
    instructions, actions = rule_pack.packed_sections()
    instruction_tensor = torch.frombuffer(bytearray(instructions), dtype=torch.uint8)
    action_tensor = torch.frombuffer(bytearray(actions), dtype=torch.uint8)
    engine.upload_rule_pack(instruction_tensor, action_tensor)
    return engine

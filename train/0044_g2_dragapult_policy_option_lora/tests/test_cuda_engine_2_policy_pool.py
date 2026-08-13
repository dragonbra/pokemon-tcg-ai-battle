from __future__ import annotations

import importlib
from pathlib import Path

import torch


inference = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.inference")
runtime = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.runtime")
ROOT = Path(__file__).resolve().parents[1]


def _pointers(policy):
    return {parameter.untyped_storage().data_ptr() for module in inference._modules(policy) for parameter in module.parameters()}


def test_resident_opponents_load_once_and_use_generation_correct_own_rows() -> None:
    pool = inference.ResidentPolicyPool(ROOT)
    pool.warm()
    assert pool.get("Policy-0809") is pool.get("Policy-0809")
    assert pool.get("Champion-G1") is pool.get("Champion-G1")
    assert pool.get("Champion-G2") is pool.get("Champion-G2")
    assert pool.get("V1-Focal-Seed") is pool.get("V1-Focal-Seed")
    assert pool.load_counts == {
        "V1-Focal-Seed": 1, "Policy-0809": 1, "Champion-G1": 1, "Champion-G2": 1,
    }
    assert pool.own_ids("Champion-G1", ("002", "007", "066")).tolist() == [3, 0, 0]
    assert all(0 <= value < 15 for value in pool.own_ids(
        "Champion-G1", tuple(f"{index:03d}" for index in range(1, 68))
    ).tolist())
    assert pool.own_ids("Champion-G2", ("002", "007", "057")).tolist() == [3, 0, 15]
    assert pool.own_ids("Policy-0809", ("002",)) is None


def test_mutable_focal_and_frozen_opponents_have_disjoint_cuda_storage() -> None:
    pool = inference.ResidentPolicyPool(ROOT)
    pool.warm()
    focal = pool.get("V1-Focal-Seed").policy
    sets = [
        _pointers(focal),
        _pointers(pool.get("Policy-0809").policy),
        _pointers(pool.get("Champion-G1").policy),
        _pointers(pool.get("Champion-G2").policy),
    ]
    assert all(not sets[left] & sets[right] for left in range(4) for right in range(left + 1, 4))
    frozen = next(iter(inference._modules(pool.get("Champion-G1").policy)[0].parameters()))
    before = frozen.detach().clone()
    mutable = next(iter(inference._modules(focal)[0].parameters()))
    mutable.data.view(-1)[0].add_(1)
    assert torch.equal(frozen, before)

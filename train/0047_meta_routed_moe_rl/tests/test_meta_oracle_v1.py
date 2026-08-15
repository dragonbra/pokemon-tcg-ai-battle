from __future__ import annotations

import importlib

import pytest
import torch


PKG = "train.0047_meta_routed_moe_rl"


def _module():
    return importlib.import_module(f"{PKG}.evaluation.meta_oracle_v1")


def test_literal_meta_route_covers_all_29_classes():
    module = _module()
    expected = {
        **{meta: 282 for meta in range(29)},
        **{meta: 200 for meta in (6, 8, 10, 27)},
        **{meta: 40 for meta in (1, 2, 17)},
        3: 90,
    }
    assert {meta: module.route_update_for_meta(meta) for meta in range(29)} == expected
    assert module.route_update_for_meta(0) == 282
    assert module.route_update_for_meta(4) == 282
    assert module.route_update_for_meta(5) == 282
    with pytest.raises(ValueError, match="29-way"):
        module.route_update_for_meta(-1)
    with pytest.raises(ValueError, match="29-way"):
        module.route_update_for_meta(29)


def test_shared_actor_audit_allows_only_decoder_difference():
    module = _module()
    common = torch.tensor([1.0, 2.0])
    states = {
        40: {"state_encoder.weight": common.clone(), "action_decoder.weight": torch.tensor([40.0])},
        282: {"state_encoder.weight": common.clone(), "action_decoder.weight": torch.tensor([282.0])},
    }
    audit = module.audit_shared_actor_states(states)
    assert audit["status"] == "PASS"
    assert audit["shared_tensor_count"] == 1
    assert audit["excluded_routed_actor_tensor_names"] == ["action_decoder.weight"]


def test_shared_actor_audit_fails_on_backbone_difference():
    module = _module()
    states = {
        40: {"state_encoder.weight": torch.tensor([1.0]), "action_decoder.weight": torch.tensor([40.0])},
        282: {"state_encoder.weight": torch.tensor([2.0]), "action_decoder.weight": torch.tensor([282.0])},
    }
    with pytest.raises(RuntimeError, match="shared frozen Actor tensor mismatch"):
        module.audit_shared_actor_states(states)


def test_partition_and_scatter_preserve_schedule_order():
    module = _module()
    rows = [
        {"game_id": "a", "opponent_meta_archetype_id": 3},
        {"game_id": "b", "opponent_meta_archetype_id": 0},
        {"game_id": "c", "opponent_meta_archetype_id": 8},
        {"game_id": "d", "opponent_meta_archetype_id": 2},
    ]
    partitions = module.partition_rows(rows)
    assert {key: [row["game_id"] for row in value] for key, value in partitions.items()} == {
        40: ["d"], 90: ["a"], 200: ["c"], 282: ["b"]
    }
    routed = {"a": 90, "b": 282, "c": 200, "d": 40}
    assert module.scatter_by_game_id(rows, routed) == [90, 282, 200, 40]
    with pytest.raises(RuntimeError, match="game ID inventory"):
        module.scatter_by_game_id(rows, {"a": 90})

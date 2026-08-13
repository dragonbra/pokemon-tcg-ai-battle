from __future__ import annotations

import copy
import importlib

import pytest


runner = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation.run_benchmark_v2"
)


def _report() -> dict:
    return {
        "status": "PASS",
        "benchmark_id": "Benchmark-V2",
        "focal_policy_identity_audit": {"status": "PASS"},
        "focal_opponent_shared_parameter_storages": 0,
        "entries": [
            {
                "valid": True, "error": None, "engine_seed": i + 1,
                "search_seed": i + 3000, "policy_seed": i + 6000,
                "coin_winner_seed": i + 9000, "focal_won_toss": bool(i & 1),
                "first_player_choice": {"focal_first": bool(i & 1)},
                "opponent_meta_archetype_id": runner.SELECTED_CLASS_IDS[i // 128],
            }
            for i in range(2048)
        ],
        "schedule": {
            "contract_id": runner.CONTRACT_ID,
            "games": 2048, "opponent_policy_id": "Policy-0809",
            "opponent_effective_policy_sha256": "o" * 64,
            "common_random_numbers": True,
            "seed_derivation_excludes": ["focal_deployment_identity"],
            "selected_class_ids": list(runner.SELECTED_CLASS_IDS),
            "excluded_class_ids": [
                value for value in range(29) if value not in runner.SELECTED_CLASS_IDS
            ],
        },
        "opponent_policy_identity_audit": {
            "status": "PASS", "requested_policy_id": "Policy-0809",
            "effective_policy_sha256": "o" * 64,
        },
        "collector_metrics": {
            "rollout/lane_routing_audit_failures": 0,
            "rollout/lane_routing_audit_pass": 1,
            "rollout/cuda_feature_d2h_bytes": 0,
            "rollout/unfinished_games": 0,
        },
    }


def test_benchmark_v2_report_gate_passes() -> None:
    runner.validate_report(_report())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schedule", "opponent_policy_id"), "Champion-G2"),
        (("schedule", "common_random_numbers"), False),
        (("schedule", "selected_class_ids"), list(range(16))),
        (("schedule", "seed_derivation_excludes"), []),
        (("opponent_policy_identity_audit", "status"), "FAIL"),
        (("focal_policy_identity_audit", "status"), "FAIL"),
        (("collector_metrics", "rollout/lane_routing_audit_failures"), 1),
        (("focal_opponent_shared_parameter_storages",), 1),
    ],
)
def test_benchmark_v2_report_gate_fails_closed(path: tuple[str, ...], value) -> None:
    report = copy.deepcopy(_report())
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError, match="Benchmark V2"):
        runner.validate_report(report)


def test_policy0809_benchmark_module_inventory_is_supported() -> None:
    opponent = runner.load_policy("Policy-0809", deck_id="001")
    modules = runner.policy_modules(opponent)
    assert modules == (opponent.model,)
    assert sum(parameter.numel() for module in modules for parameter in module.parameters()) > 0

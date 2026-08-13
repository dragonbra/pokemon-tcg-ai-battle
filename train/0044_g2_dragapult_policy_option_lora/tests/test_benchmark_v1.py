from __future__ import annotations

import importlib

import pytest


runner = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation.run_benchmark_v1"
)


def _report() -> dict:
    entries = [{
        "valid": True, "error": None, "outcome": 1,
        "opponent_id": f"{index % 67 + 1:03d}",
        "opponent_meta_archetype_id": index % 29,
    } for index in range(2048)]
    return {
        "status": "PASS", "focal_policy_id": "Champion-G2",
        "focal_opponent_shared_parameter_storages": 0,
        "entries": entries,
        "schedule": {
            "opponent_policy_id": "Champion-G2", "games": 2048,
            "empty_class_ids": [14], "opponent_effective_policy_sha256": "a" * 64,
        },
        "opponent_policy_identity_audit": {
            "requested_policy_id": "Champion-G2", "effective_policy_sha256": "a" * 64,
        },
        "collector_metrics": {
            "rollout/lane_routing_audit_failures": 0,
            "rollout/cuda_feature_d2h_bytes": 0,
            "rollout/lane_routing_audit_pass": 1,
        },
    }


def test_benchmark_report_accepts_only_complete_fixed_g2_identity() -> None:
    runner.validate_report(_report())
    for mutation, match in (
        (("schedule", "opponent_policy_id", "Champion-G1"), "opponent identity"),
        (("collector_metrics", "rollout/cuda_feature_d2h_bytes", 8), "residency"),
    ):
        report = _report()
        report[mutation[0]][mutation[1]] = mutation[2]
        with pytest.raises(RuntimeError, match=match):
            runner.validate_report(report)


def test_benchmark_report_rejects_unfinished_games() -> None:
    report = _report()
    report["entries"][10]["valid"] = False
    with pytest.raises(RuntimeError, match="terminal"):
        runner.validate_report(report)

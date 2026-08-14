from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch


def _handoff():
    return importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.handoff_v9_u50_to_v10"
    )


def _checkpoint(path: Path, version: str) -> None:
    torch.save({
        "schema_version": "0044_focal_v1_model_only_v1",
        "update": 50,
        "metadata": {"version": version},
        "state_dict": {},
    }, path)


def test_handoff_gate_requires_matching_canonical_eval_metric(tmp_path: Path) -> None:
    handoff = _handoff()
    checkpoint = tmp_path / "update-000050.pt"
    report = tmp_path / "report.json"
    metrics = tmp_path / "training_metrics.jsonl"
    _checkpoint(checkpoint, handoff.run_v10.PARENT_VERSION)
    report.write_text(json.dumps({
        "status": "PASS", "benchmark_id": "Benchmark-V2",
        "focal_checkpoint_update": 50,
        "summary": {"games": 2048, "win_rate": 0.5},
    }), encoding="utf-8")
    metrics.write_text("", encoding="utf-8")
    assert handoff.build_handoff_manifest(
        checkpoint=checkpoint, report_path=report, metrics_path=metrics,
    ) is None
    metrics.write_text(json.dumps({
        "eval/checkpoint_update": 50,
        "eval/benchmark_v2": 1.0,
        "eval/games": 2048,
    }) + "\n", encoding="utf-8")
    manifest = handoff.build_handoff_manifest(
        checkpoint=checkpoint, report_path=report, metrics_path=metrics,
    )
    assert manifest is not None
    assert manifest["status"] == "PASS"
    assert manifest["parent_checkpoint_update"] == 50
    assert manifest["optimizer_boundary"] == "fresh_v10"


def test_v10_launch_command_is_indefinite_and_online() -> None:
    handoff = _handoff()
    command = handoff.v10_command()
    assert "--updates" not in command
    assert command[-2:] == ("--wandb-mode", "online")
    assert "run_v10" in " ".join(command)

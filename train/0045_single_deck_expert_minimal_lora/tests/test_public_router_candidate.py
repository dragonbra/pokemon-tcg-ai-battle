from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT / "train/0045_single_deck_expert_minimal_lora"
V12 = (
    ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions/"
    "V12_dragapult_007_public_meta_router_v3_grass_fold_u200_cuda2048/"
    "artifact/public_router_evaluation"
)


def test_compact_candidate_matches_v12_and_loads_fp32(tmp_path: Path):
    candidate = importlib.import_module(
        "train.0045_single_deck_expert_minimal_lora.evaluation.public_router_candidate"
    )
    router = importlib.import_module(
        "train.0045_single_deck_expert_minimal_lora.semantic_runtime.deployment.public_meta_router"
    )
    assets = importlib.import_module(
        "train.0045_single_deck_expert_minimal_lora.assets"
    )
    portables = {
        update: V12 / f"materialization/update-{update:06d}/model.bin"
        for update in (40, 90, 200, 282)
    }
    report = json.loads((V12 / "report.json").read_text())
    output = tmp_path / "router_heads.bin"
    audit = candidate.materialize_compact_heads(
        portable_candidates=portables, qualifying_report=report, output=output
    )
    assert audit["status"] == "PASS"
    assert audit["composite_effective_sha256"] == report[
        "focal_public_router_identity_audit"
    ]["composite_effective_sha256"]
    assert output.stat().st_size < 12_000_000
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert set(payload["routed_head_state_dicts"]) == {"40", "90", "200"}
    floating = [
        value
        for route in payload["routed_head_state_dicts"].values()
        for state in route.values()
        for value in state.values()
        if torch.is_floating_point(value)
    ]
    assert floating and {value.dtype for value in floating} == {torch.float16}

    registry = assets.AssetRegistry.load(PROJECT_ROOT)
    deck_asset = next(row for row in registry.decks if row.deck_id == "007")
    deck = tuple(map(int, (PROJECT_ROOT / deck_asset.deck_path).read_text().splitlines()))
    policy = router.PublicRoutedCompoundPolicy.from_compact_checkpoint(
        portables[282], output, deck
    )
    assert policy.active_update == 282
    assert policy.deployment_identity["composite_effective_sha256"] == audit[
        "composite_effective_sha256"
    ]
    parameters = [
        parameter for module in (
            policy.actor, policy.value_head, policy.value_adapter,
            *(head.action_decoder for head in policy.heads.values()),
            *(head.policy_option_lora for head in policy.heads.values()),
            *(head.allocation_head for head in policy.heads.values()),
        ) for parameter in module.parameters() if torch.is_floating_point(parameter)
    ]
    assert parameters and {parameter.dtype for parameter in parameters} == {torch.float32}


def test_exporter_resolves_policy0809_from_identity_audit():
    exporter = importlib.import_module(
        "train.0045_single_deck_expert_minimal_lora.evaluation.export_public_router_kaggle"
    )
    report = json.loads((V12 / "report.json").read_text())
    assert exporter._validated_opponent_id(report) == "Policy-0809"

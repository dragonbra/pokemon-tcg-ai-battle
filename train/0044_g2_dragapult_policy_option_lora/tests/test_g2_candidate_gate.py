from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest
import torch


gate = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation.g2_candidate_gate"
)
renderer = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation.render_g2_candidate_gate"
)


def test_seeded_cohorts_are_exact_and_disjoint() -> None:
    assert gate.RANDOM_SAMPLE_SEED == 430044407
    assert gate.RANDOM_DECK_IDS == (
        "031", "039", "040", "042", "047",
        "056", "057", "058", "061", "064",
    )
    assert gate.FIXED_DECK_IDS == tuple(f"{index:03d}" for index in range(1, 11))
    assert len(gate.EVALUATION_DECK_IDS) == 20
    assert len(set(gate.EVALUATION_DECK_IDS)) == 20
    assert gate.EXECUTION_DECK_IDS[:10] == gate.RANDOM_DECK_IDS
    assert gate.EXECUTION_DECK_IDS[10:] == gate.FIXED_DECK_IDS
    assert gate.MAX_PARALLEL_ARMS == 2


def test_cohort_assets_are_immutable_exact_60() -> None:
    audit = gate.audit_deck_scope()
    assert audit["status"] == "PASS"
    assert audit["deck_ids"] == list(gate.EVALUATION_DECK_IDS)
    assert all(row["card_count"] == 60 for row in audit["decks"])
    assert all(len(row["exact_deck_sha256"]) == 64 for row in audit["decks"])


def test_u407_checkpoint_gate_rejects_wrong_update(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong.pt"
    torch.save({"schema_version": "0044_focal_v1_model_only_v1", "update": 406}, checkpoint)
    with pytest.raises(RuntimeError, match="U407"):
        gate.validate_u407_checkpoint(checkpoint)


def _valid_report() -> dict:
    return {
        "status": "PASS",
        "checkpoint_update": 407,
        "summary": {"games": 2048},
        "entries": [
            {"valid": True, "error": None, "outcome": 1}
            for _ in range(2048)
        ],
        "candidate_deployment_identity_audit": {
            "status": "PASS",
            "contract_id": "kaggle_fp16_storage_fp32_runtime_v1",
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
        },
        "opponent_policy_identity_audit": {
            "status": "PASS", "requested_policy_id": "Policy-0809",
        },
        "focal_opponent_shared_parameter_storages": 0,
        "collector_metrics": {
            "rollout/lane_routing_audit_failures": 0,
            "rollout/cuda_feature_d2h_bytes": 0,
        },
    }


def test_arm_validation_is_fail_closed() -> None:
    gate.validate_arm_report(_valid_report(), arm="g2_candidate", deck_id="001")
    broken = _valid_report()
    broken["focal_opponent_shared_parameter_storages"] = 1
    with pytest.raises(RuntimeError, match="storage"):
        gate.validate_arm_report(broken, arm="g2_candidate", deck_id="001")
    broken = _valid_report()
    broken["collector_metrics"]["rollout/cuda_feature_d2h_bytes"] = 4
    with pytest.raises(RuntimeError, match="routing/residency"):
        gate.validate_arm_report(broken, arm="g2_candidate", deck_id="001")


def test_state_rejects_unknown_or_duplicate_arm(tmp_path: Path) -> None:
    state = gate.initial_state()
    assert state["status"] == "WAITING_FOR_U407"
    assert state["completed_arms"] == []
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    loaded = gate.load_state(path)
    assert loaded == state
    loaded["completed_arms"] = ["001:g1", "001:g1"]
    path.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        gate.load_state(path)


def test_g1_materialization_uses_native_diagnostic_adapters(tmp_path: Path) -> None:
    registry = gate.AssetRegistry.load(gate.PROJECT_ROOT)
    asset = next(row for row in registry.decks if row.deck_id == "001")
    deck = tuple(map(int, (gate.PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
    model, audit = gate._g1_candidate(
        deck, "001", 2, tmp_path / "g1.bin", torch.device("cpu")
    )
    assert audit.status == "PASS"
    assert callable(model.value_adapter.effective_residual_ratio)
    assert callable(model.policy_strategy_adapter.effective_residual_ratio)
    hidden = torch.randn(3, 320)
    own = torch.full((3,), 2, dtype=torch.long)
    _, delta = model.value_adapter(hidden, own)
    assert model.value_adapter.effective_residual_ratio(hidden, delta).shape == (3,)


def test_independent_difference_interval_and_report_boundary() -> None:
    interval = renderer._difference_interval(1024, 1124, 2048)
    delta = (1124 - 1024) / 2048
    assert interval[0] < delta < interval[1]
    assert interval[0] < interval[1]
    source = inspect.getsource(renderer.render)
    assert "PARTIAL_EVALUATION_IN_PROGRESS" in source
    assert "不作逐局 paired 推断" in source
    assert "Only human PROMOTE / HOLD / REJECT" in source
    assert "HUMAN_DECISION_REQUIRED" in inspect.getsource(renderer.aggregate)


def test_reporting_meta_taxonomy_covers_exact_001_067() -> None:
    taxonomy = renderer._meta_taxonomy()
    assert len(taxonomy["classes"]) == 29
    assert set(taxonomy["deck_to_class"]) == {
        f"{value:03d}" for value in range(1, 68)
    }
    memberships = [
        deck_id for row in taxonomy["classes"] for deck_id in row["deck_ids"]
    ]
    assert len(memberships) == 67
    assert len(set(memberships)) == 67
    assert taxonomy["reporting_only"] is True
    assert taxonomy["model_opponent_meta_head_unchanged"] is True


def test_meta_aggregation_conserves_outcomes_and_rejects_unknown() -> None:
    taxonomy = renderer._meta_taxonomy()
    report = {"entries": [
        {"opponent_id": "001", "outcome": 1},
        {"opponent_id": "005", "outcome": -1},
        {"opponent_id": "007", "outcome": 0},
    ]}
    rows = renderer._by_meta_archetype(report, taxonomy)
    assert len(rows) == 29
    assert sum(row["games"] for row in rows) == 3
    assert sum(row["wins"] for row in rows) == 1
    assert sum(row["losses"] for row in rows) == 1
    assert sum(row["draws"] for row in rows) == 1
    grimmsnarl = next(row for row in rows if row["archetype_id"] == 2)
    assert grimmsnarl["deck_ids"] == ["001", "005", "017", "031", "037"]
    assert grimmsnarl["observed_deck_ids"] == ["001", "005"]
    assert grimmsnarl["games"] == 2
    no_sample = next(row for row in rows if row["games"] == 0)
    assert no_sample["win_rate"] is None
    with pytest.raises(RuntimeError, match="unknown opponent deck"):
        renderer._by_meta_archetype(
            {"entries": [{"opponent_id": "999", "outcome": 1}]}, taxonomy
        )


def test_meta_opponent_art_covers_all_decks() -> None:
    art = renderer._deck_representative_art()
    assert set(art) == {f"{value:03d}" for value in range(1, 68)}
    assert all(1 <= len(cards) <= 2 for cards in art.values())
    assert all(
        card["image_url"] and card["name"] and card["card_id"] > 0
        for cards in art.values() for card in cards
    )

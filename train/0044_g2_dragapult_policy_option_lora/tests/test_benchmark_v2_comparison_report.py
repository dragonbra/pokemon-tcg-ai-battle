from __future__ import annotations

import copy
import importlib

import pytest


renderer = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation.render_benchmark_v2_comparison"
)


def _report(update: int, wins_by_meta: int) -> dict:
    entries = []
    selected = list(range(14)) + [17, 27]
    for index in range(2048):
        entries.append({
            "game_id": f"g-{index}", "opponent_id": f"{index % 50 + 1:03d}",
            "opponent_meta_archetype_id": selected[index // 128],
            "outcome": 1 if index % 128 < wins_by_meta else -1,
            "engine_seed": index + 1, "search_seed": index + 2,
            "policy_seed": index + 3, "coin_winner_seed": index + 4,
        })
    return {
        "status": "PASS",
        "focal_checkpoint_update": update, "focal_deck_id": "007",
        "focal_deck_display_name": "Dragapult", "focal_deck_cards": [1] * 60,
        "focal_policy_identity_audit": {"source_checkpoint_sha256": str(update) * 64},
        "focal_deployment_effective_sha256": "e" * 64,
        "schedule": {
            "contract_id": renderer.PROJECT_ROOT.name, "selected_class_ids": selected,
            "common_random_schedule_sha256": "c" * 64, "schedule_sha256": "s" * 64,
        },
        "summary": {
            "games": 2048, "wins": wins_by_meta * 16,
            "losses": (128 - wins_by_meta) * 16, "draws": 0,
            "win_rate": wins_by_meta / 128,
        },
        "collector_metrics": {
            "rollout/lane_routing_audit_failures": 0.0,
            "rollout/cuda_feature_d2h_bytes": 0.0,
        },
        "entries": entries,
    }


def test_comparison_requires_paired_jobs_and_aggregates_delta(monkeypatch) -> None:
    monkeypatch.setattr(renderer, "validate_report", lambda report: None)
    monkeypatch.setattr(renderer, "_representative_cards", lambda cards: [])
    monkeypatch.setattr(renderer, "_deck_representative_art", lambda: {
        f"{value:03d}": [] for value in range(1, 68)
    })
    classes = [{
        "archetype_id": value, "display_name": f"Meta {value}",
        "definition": "definition", "deck_ids": [f"{value + 1:03d}"],
    } for value in range(29)]
    monkeypatch.setattr(renderer, "_meta_taxonomy", lambda: {"classes": classes})
    payload = renderer.aggregate(_report(0, 64), _report(30, 72))
    assert payload["status"] == "PASS"
    assert payload["delta"] == pytest.approx(8 / 128)
    assert len(payload["by_meta_archetype"]) == 16
    assert payload["paired_transitions"]["loss_to_win"] == 8 * 16

    mismatched = _report(30, 72)
    mismatched["entries"][0]["engine_seed"] = 999
    with pytest.raises(RuntimeError, match="job identity"):
        renderer.aggregate(_report(0, 64), mismatched)

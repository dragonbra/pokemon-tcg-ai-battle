from __future__ import annotations

import importlib

import pytest


renderer = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation."
    "render_v18_late_checkpoint_selection"
)


LABELS = ("U14", "U15", "U16", "U17", "EMA U14–U17")
SELECTED = list(range(14)) + [17, 27]


def _stat(wins: int, games: int = 128) -> dict[str, float | int]:
    return {
        "games": games, "wins": wins, "losses": games - wins,
        "draws": 0, "win_rate": wins / games,
    }


def _report(update: int, wins_per_meta: int) -> dict:
    entries = []
    for index in range(2048):
        entries.append({
            "game_id": f"game-{index}",
            "opponent_id": f"{index % 32 + 1:03d}",
            "opponent_meta_archetype_id": SELECTED[index // 128],
            "outcome": 1 if index % 128 < wins_per_meta else -1,
            "engine_seed": index + 1,
            "search_seed": index + 2,
            "policy_seed": index + 3,
            "coin_winner_seed": index + 4,
        })
    return {
        "status": "PASS", "focal_deck_id": "069",
        "focal_checkpoint_update": update, "entries": entries,
        "summary": _stat(wins_per_meta * 16, 2048),
        "schedule": {
            "contract_id": "benchmark-v2", "selected_class_ids": SELECTED,
            "common_random_schedule_sha256": "c" * 64,
            "schedule_sha256": str(update) * 64,
        },
        "focal_policy_identity_audit": {
            "source_checkpoint_sha256": str(update) * 64,
        },
        "focal_deployment_effective_sha256": "e" * 64,
    }


def _baseline() -> dict:
    meta = [{
        "archetype_id": value, "g1": _stat(70),
        "g2_candidate": _stat(71),
    } for value in SELECTED]
    by_deck = {f"{value:03d}": _stat(2, 4) for value in range(1, 33)}
    return {"rows": [{
        "deck_id": "069", "g1": _stat(1120, 2048),
        "g2_candidate": _stat(1136, 2048), "by_meta_archetype": meta,
        "g1_by_opponent": by_deck, "g2_by_opponent": by_deck,
    }]}


def test_aggregate_selects_best_and_requires_common_jobs(monkeypatch) -> None:
    monkeypatch.setattr(renderer, "validate_report", lambda report: None)
    monkeypatch.setattr(renderer, "_meta_taxonomy", lambda: {"classes": [
        {"archetype_id": value, "display_name": f"Meta {value}",
         "deck_ids": [f"{value + 1:03d}"]}
        for value in range(28)
    ]})
    reports = {
        label: _report(17 if label.startswith("EMA") else int(label[1:]), 70 + index)
        for index, label in enumerate(LABELS)
    }
    payload = renderer.aggregate(baseline_manifest=_baseline(), reports=reports)
    assert payload["best_observed"]["label"] == "EMA U14–U17"
    assert len(payload["by_meta_archetype"]) == 16
    assert len(payload["by_opponent_deck"]) == 32

    reports["U17"]["entries"][0]["engine_seed"] = 999
    with pytest.raises(RuntimeError, match="job identity mismatch"):
        renderer.aggregate(baseline_manifest=_baseline(), reports=reports)


from __future__ import annotations

import importlib


renderer = importlib.import_module(
    "train.0043_champion_league_rl.evaluation.render_benchmark_v1"
)


def test_rendered_benchmark_has_meta_deck_counts_and_card_art(monkeypatch) -> None:
    monkeypatch.setattr(renderer, "validate_report", lambda report: None)
    art = {f"{value:03d}": [{
        "card_id": value, "name": f"Card {value}", "image_url": f"https://img/{value}.png"
    }] for value in range(1, 68)}
    monkeypatch.setattr(renderer, "_deck_representative_art", lambda: art)
    classes = [{
        "archetype_id": value, "name": f"meta_{value}",
        "display_name": f"Meta {value}", "definition": "definition",
        "parent_archetype": None, "deck_ids": [] if value == 14 else [f"{value % 67 + 1:03d}"],
    } for value in range(29)]
    monkeypatch.setattr(renderer, "_meta_taxonomy", lambda: {"classes": classes})
    monkeypatch.setattr(renderer, "_representative_cards", lambda cards: art["002"])
    entries = [{
        "opponent_id": f"{index % 67 + 1:03d}",
        "opponent_meta_archetype_id": index % 14,
        "outcome": 1 if index % 2 else -1,
    } for index in range(2048)]
    report = {
        "entries": entries, "focal_deck_cards": [1] * 60,
        "focal_deck_id": "002", "focal_deck_display_name": "Alakazam",
        "focal_policy_id": "Champion-G2", "focal_exact_deck_sha256": "d" * 64,
        "focal_policy_identity_audit": {"effective_policy_sha256": "f" * 64},
        "focal_deployment_effective_sha256": "e" * 64,
        "schedule": {
            "contract_id": "0043_benchmark_v1_meta_balanced_g2_cuda2048_v1",
            "schedule_sha256": "s" * 64, "opponent_policy_id": "Champion-G2",
            "opponent_effective_policy_sha256": "o" * 64, "master_seed": 341512902,
        },
        "cuda_engine_identity_audit": {"engine_source_sha256": "c" * 64},
        "summary": {
            "games": 2048, "wins": 1024, "losses": 1024, "draws": 0,
            "win_rate": .5, "focal_first_games": 1024, "focal_first_win_rate": .51,
            "focal_second_games": 1024, "focal_second_win_rate": .49,
            "games_per_second": 6.0, "elapsed_seconds": 341.3,
        },
    }
    payload = renderer.aggregate(report)
    page = renderer.render(payload)
    assert len(payload["by_meta_archetype"]) == 29
    assert len(payload["by_opponent_deck"]) == 67
    assert "n=0 · 无已注册卡组/样本" in page
    assert "By opponent exact deck" in page
    assert "Against Meta Archetype" in page
    assert "https://img/" in page
    assert "n=2048" in page

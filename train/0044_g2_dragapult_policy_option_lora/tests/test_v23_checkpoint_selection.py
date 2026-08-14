from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


builder = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation."
    "build_v23_u7_u10_ema"
)
renderer = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation."
    "render_v23_checkpoint_selection"
)

LABELS = ("U7", "U8", "U9", "U10", "EMA U7–U10")
SELECTED = list(range(14)) + [17, 27]


def test_ema_uses_audited_normalized_decay_half_weights(tmp_path: Path) -> None:
    output = tmp_path / "model.pt"
    manifest_path = tmp_path / "manifest.json"
    manifest = builder.build(
        output_checkpoint=output, output_manifest=manifest_path
    )

    assert manifest["status"] == "PASS"
    assert manifest["source_version"] == "V23_v22_u9_final_entropy_0015"
    assert manifest["source_updates"] == [7, 8, 9, 10]
    assert manifest["normalized_weights"] == {
        "7": 1 / 15, "8": 2 / 15, "9": 4 / 15, "10": 8 / 15,
    }
    assert manifest["source_checkpoint_sha256"] == builder.EXPECTED_SOURCE_HASHES
    assert manifest["floating_accumulation_dtype"] == "torch.float32"
    assert manifest["non_floating_source_update"] == 10
    assert len(manifest["output_checkpoint_sha256"]) == 64

    derived = torch.load(output, map_location="cpu", weights_only=True)
    assert derived["schema_version"] == "0044_focal_v1_model_only_v1"
    assert derived["update"] == 10
    assert derived["metadata"]["version"] == builder.DERIVED_VERSION
    assert derived["derived_checkpoint"]["source_updates"] == [7, 8, 9, 10]
    assert derived["derived_checkpoint"]["not_a_true_optimizer_update"] is True

    sources = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in builder.SOURCE_CHECKPOINTS
    ]
    float_key = next(
        key for key, value in derived["state_dict"].items()
        if value.dtype == torch.float32
    )
    expected = sum(
        sources[index]["state_dict"][float_key].float() * weight
        for index, weight in enumerate((1 / 15, 2 / 15, 4 / 15, 8 / 15))
    )
    torch.testing.assert_close(derived["state_dict"][float_key], expected)

    integer_key = next(
        key for key, value in derived["state_dict"].items()
        if not value.is_floating_point()
    )
    assert torch.equal(
        derived["state_dict"][integer_key],
        sources[-1]["state_dict"][integer_key],
    )


def _stat(wins: int, games: int = 128) -> dict[str, float | int | list[float]]:
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "draws": 0,
        "win_rate": wins / games,
        "wilson_95": [0.4, 0.6],
        "focal_first_games": games // 2,
        "focal_first_win_rate": wins / games,
        "focal_second_games": games - games // 2,
        "focal_second_win_rate": wins / games,
    }


def _report(update: int, wins_per_meta: int) -> dict:
    identity_digit = str(wins_per_meta % 10)
    entries = [{
        "game_id": f"game-{index}",
        "opponent_id": f"{index % 32 + 1:03d}",
        "opponent_meta_archetype_id": SELECTED[index // 128],
        "outcome": 1 if index % 128 < wins_per_meta else -1,
        "engine_seed": index + 1,
        "search_seed": index + 2,
        "policy_seed": index + 3,
        "coin_winner_seed": index + 4,
    } for index in range(2048)]
    return {
        "status": "PASS",
        "focal_deck_id": "069",
        "focal_checkpoint_update": update,
        "entries": entries,
        "summary": _stat(wins_per_meta * 16, 2048),
        "schedule": {
            "contract_id": "benchmark-v2",
            "selected_class_ids": SELECTED,
            "common_random_schedule_sha256": "c" * 64,
            "schedule_sha256": str(update % 10) * 64,
        },
        "focal_policy_identity_audit": {
            "status": "PASS",
            "contract_id": "kaggle_fp16_storage_fp32_runtime_v1",
            "source_checkpoint_sha256": identity_digit * 64,
            "portable_checkpoint_sha256": identity_digit * 64,
            "effective_candidate_sha256": identity_digit * 64,
        },
        "focal_deployment_effective_sha256": identity_digit * 64,
        "opponent_policy_identity_audit": {"status": "PASS"},
    }


def _baseline() -> dict:
    meta = [{
        "archetype_id": value,
        "g1": _stat(70),
        "g2_candidate": _stat(71),
    } for value in SELECTED]
    by_deck = {f"{value:03d}": _stat(2, 4) for value in range(1, 33)}
    return {"rows": [{
        "deck_id": "069",
        "g1": _stat(1120, 2048),
        "g2_candidate": _stat(1136, 2048),
        "by_meta_archetype": meta,
        "g1_by_opponent": by_deck,
        "g2_by_opponent": by_deck,
    }]}


def test_aggregate_selects_best_and_requires_common_jobs(monkeypatch) -> None:
    monkeypatch.setattr(renderer, "validate_report", lambda report: None)
    monkeypatch.setattr(renderer, "_meta_taxonomy", lambda: {"classes": [{
        "archetype_id": value,
        "display_name": f"Meta {value}",
        "deck_ids": [f"{value + 1:03d}"],
    } for value in range(29)]})
    reports = {
        label: _report(10 if label.startswith("EMA") else int(label[1:]), 70 + index)
        for index, label in enumerate(LABELS)
    }
    payload = renderer.aggregate(
        baseline_manifest=_baseline(), reports=reports
    )
    assert payload["best_observed"]["label"] == "EMA U7–U10"
    assert payload["best_observed"]["promotion_decision"] == "none"
    assert payload["ema_contract"]["source_updates"] == [7, 8, 9, 10]
    assert len(payload["by_meta_archetype"]) == 16
    assert len(payload["by_opponent_deck"]) == 32

    reports["U10"]["entries"][0]["engine_seed"] = 999
    with pytest.raises(RuntimeError, match="job identity mismatch"):
        renderer.aggregate(baseline_manifest=_baseline(), reports=reports)

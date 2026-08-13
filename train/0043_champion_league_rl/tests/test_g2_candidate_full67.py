from __future__ import annotations

import importlib


full = importlib.import_module(
    "train.0043_champion_league_rl.evaluation.g2_candidate_full67"
)


def test_full67_runs_reverse_and_inherits_v7_without_repeating() -> None:
    assert full.DECK_IDS == tuple(f"{value:03d}" for value in range(67, 0, -1))
    assert len(full.DECK_IDS) == 67
    assert full.MAX_PARALLEL_ARMS == 2
    assert full.VERSION == "V8_u407_g2_candidate_full67_cuda2048"
    assert set(full.gate.EVALUATION_DECK_IDS).issubset(set(full.DECK_IDS))


def test_full67_source_prefers_local_then_v7(monkeypatch, tmp_path) -> None:
    local = tmp_path / "local"
    inherited = tmp_path / "v7"
    monkeypatch.setattr(full, "REPORT_ROOT", local)
    monkeypatch.setattr(full.gate, "REPORT_ROOT", inherited)
    assert full.source_report("067", "g1") is None
    old = inherited / "067/g1/report.json"
    old.parent.mkdir(parents=True)
    old.write_text("{}", encoding="utf-8")
    assert full.source_report("067", "g1") == old
    new = local / "067/g1/report.json"
    new.parent.mkdir(parents=True)
    new.write_text("{}", encoding="utf-8")
    assert full.source_report("067", "g1") == new

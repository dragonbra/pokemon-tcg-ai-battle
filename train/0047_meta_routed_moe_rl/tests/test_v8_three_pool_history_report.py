from __future__ import annotations

import importlib

import pytest


PKG = "train.0047_meta_routed_moe_rl"


def _module():
    return importlib.import_module(f"{PKG}.evaluation.render_v8_three_pool_history")


def test_v8_history_has_exact_updates_and_dual_taxonomy_totals():
    module = _module()
    history = module.load_default_history()
    expected = tuple(range(200, 276, 5)) + tuple(range(276, 282)) + tuple(range(282, 299, 2))
    assert history.updates == expected
    assert tuple(history.current_names) == tuple(range(29))
    assert history.active_current_ids == tuple(i for i in range(29) if i != 14)
    assert tuple(history.value_names) == tuple(range(15))
    assert len(history.updates) == 31
    for update in history.updates:
        assert sum(row[update].games for row in history.current.values()) == 1536
        assert sum(row[update].games for row in history.value.values()) == 1536
        assert history.overall[update].games == 1536


def test_value_taxonomy_is_exact_training_classifier_not_old_own_id():
    module = _module()
    history = module.load_default_history()
    assert history.deck_to_current["066"] == 15
    assert history.deck_to_value["066"] == 0
    assert history.deck_to_current["023"] == 27
    assert history.deck_to_value["023"] == 6
    assert history.current_to_value[15] == (0,)
    assert history.current_to_value[27] == (6,)
    assert all(len(labels) == 1 for labels in history.current_to_value.values())


def test_cuda2048_history_includes_every_valid_measured_update():
    module = _module()
    history = module.load_cuda2048_history()
    assert history.updates == (40,) + tuple(range(90, 201, 5)) + (282,)
    assert len(history.updates) == 25
    for update in history.updates:
        assert history.overall[update].games == 2048
        assert sum(row[update].games for row in history.current.values()) == 2048
        assert sum(row[update].games for row in history.value.values()) == 2048
    u190_audit = next(row for row in history.report_audits if row["update"] == 190)
    assert "identity_fixed" in u190_audit["path"]


def test_entry_aggregation_fails_closed_for_unmapped_deck():
    module = _module()
    with pytest.raises(KeyError, match="999"):
        module.aggregate_entries(
            [{"opponent_id": "999", "opponent_meta_archetype_id": 0, "outcome": 1}],
            deck_to_current={},
            deck_to_value={},
        )


def test_row_heat_is_relative_and_marks_all_tied_maxima():
    module = _module()
    cells = module._row_heat_cells([
        module.OutcomeStats(1, 3, 0),
        module.OutcomeStats(3, 1, 0),
        module.OutcomeStats(3, 1, 0),
        module.OutcomeStats(),
    ])
    assert cells.count('class="row-heat row-best"') == 2
    assert cells.count("行内最高") == 4  # title + visible label for each tied maximum
    assert "rgba(23,107,75,0.080)" in cells
    assert "rgba(23,107,75,0.660)" in cells
    assert "n=0" in cells


def test_render_contains_two_complete_meta_tables_and_conflict_audit():
    module = _module()
    html = module.render(module.load_default_history())
    assert 'id="meta-current-29"' in html
    assert 'id="meta-value-15"' in html
    assert 'id="taxonomy-crosswalk"' in html
    assert html.count('data-current-meta-row="1"') == 29
    assert html.count('data-value-meta-row="1"') == 15
    assert html.count('data-cuda-current-meta-row="1"') == 29
    assert html.count('data-cuda-value-meta-row="1"') == 15
    assert html.count('data-cuda-tradeoff-row="1"') == 21
    for update in (40,) + tuple(range(90, 201, 5)) + (282,):
        assert f">U{update}<" in html
    for update in tuple(range(200, 276, 5)) + tuple(range(276, 282)) + tuple(range(282, 299, 2)):
        assert f">U{update}<" in html
    assert "990-546-0" in html
    assert "Value-loss 15-way" in html
    assert "opponent_archetypes_v1" in html
    assert "classify_target" in html
    assert 'id="meta-tradeoffs"' in html
    assert 'data-tradeoff-row="1"' in html
    assert 'class="row-heat row-best"' in html
    assert "浅 → 深：本 Meta 行内胜率低 → 高" in html


def test_v9_render_appends_cheating_oracle_last_with_full_audit():
    module = _module()
    oracle = module.load_oracle_report()
    html = module.render(module.load_default_history(), oracle_report=oracle)
    assert 'id="meta-oracle-v1"' in html
    assert html.count('data-oracle-meta-row="1"') == 16
    assert "1381-667-0" in html
    assert "67.43%" in html
    assert "Experimental-Oracle-MetaRouter-V1" in html
    assert "diagnostic_oracle_not_promote_not_kaggle" in html
    assert "拼好龙 V1 作弊版" in html
    assert "shared↔head 0 · head↔head 0 · focal↔opponent 0" in html
    assert html.index('id="meta-oracle-v1"') > html.index("证据与可比性")

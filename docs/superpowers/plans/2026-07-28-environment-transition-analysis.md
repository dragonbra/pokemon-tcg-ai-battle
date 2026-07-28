# Environment Transition Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one auditable, card-image-led HTML page that explains the 2026-07-26 through 2026-07-28 Top 100 environment transition without overstating cross-contract evidence.

**Architecture:** A date-parameterized renderer parses the published daily rank tables and uses the immutable 0727/0728 snapshot payloads for exact deck, card-pool, and Meta-derived metrics. It emits one static page beside the daily archive, uses the 0726 visual baseline, and updates the environment index with a dedicated analysis entry.

**Tech Stack:** Python 3.11 standard library HTML parsing/JSON, the existing official card CSV and card-image helpers, static HTML/CSS/JavaScript, `unittest`.

## Global Constraints

- 0727 and 0728 must remain bounded by their embedded complete snapshot audits; do not query Kaggle or alter their frozen evidence.
- 0726 is a roster-history anchor only; its differently scoped Meta fields must be labelled non-comparable.
- The output must retain the daily report visual language, responsive layout, card thumbnails, large-card hover preview, evidence boundaries, and no fabricated zero-valued evidence.
- The new page must be generated under `docs/environment-daily_kaggle_top100/` and linked from its date-sorted index.
- Do not modify `engine/source/`, existing daily reports, or unrelated working-tree changes.

---

### Task 1: Add source parsing and transition calculations

**Files:**
- Create: `data/processed/environment_daily/generate_transition_analysis.py`
- Test: `tests/test_environment_transition_analysis.py`

**Interfaces:**
- Consumes: `docs/environment-daily_kaggle_top100/daily/2026-07-26.html`, `2026-07-27.html`, `2026-07-28.html`, `.tmp/environment_daily/2026-07-27/run-20260726T205643Z-stabilized/snapshot.json`, and `.tmp/environment_daily/2026-07-28/run-20260727T202906Z/snapshot.json`.
- Produces: `load_daily_roster(path: Path) -> list[DailyPlayer]` and `build_transition_data(...) -> TransitionData` with roster overlap, rank deltas, archetype counts, exact-deck changes, and 0727-to-0728 card-pool deltas.

- [x] **Step 1: Write the failing parsing and comparison tests**

```python
def test_transition_data_keeps_three_rosters_and_exact_two_day_deck_delta() -> None:
    data = build_transition_data(...)
    assert [snapshot.date for snapshot in data.snapshots] == ["2026-07-26", "2026-07-27", "2026-07-28"]
    assert data.strict_comparison_dates == ("2026-07-27", "2026-07-28")
    assert data.card_pool_delta
```

- [x] **Step 2: Run the focused test and verify it fails before the renderer exists**

Run: `python3 -m unittest -v tests.test_environment_transition_analysis`
Expected: FAIL with an import error for `generate_transition_analysis`.

- [x] **Step 3: Implement strict source loading and calculations**

```python
def build_transition_data(...) -> TransitionData:
    rosters = [load_daily_roster(path) for path in report_paths]
    audited = [load_complete_snapshot(path) for path in snapshot_paths]
    return TransitionData(rosters=rosters, audited=audited)
```

- [x] **Step 4: Run the focused test and verify it passes**

Run: `python3 -m unittest -v tests.test_environment_transition_analysis`
Expected: PASS with three 100-player rosters and two complete audited snapshots.

### Task 2: Render and link the transition page

**Files:**
- Modify: `data/processed/environment_daily/generate_transition_analysis.py`
- Create: `docs/environment-daily_kaggle_top100/environment-transition.html`
- Modify: `docs/environment-daily_kaggle_top100/index.html`
- Test: `tests/test_environment_transition_analysis.py`

**Interfaces:**
- Consumes: `TransitionData`, `_card_catalog()`, `_card_thumb()`, and `_baseline_css()`.
- Produces: `render_transition_report(data: TransitionData, output: Path) -> None` and an index article linking `environment-transition.html`.

- [x] **Step 1: Extend the test with the page contract**

```python
def test_rendered_transition_page_has_evidence_and_daily_ui_sections() -> None:
    render_transition_report(data, report)
    assert "id=\"transition-overview\"" in report.read_text(encoding="utf-8")
    assert "id=\"evidence-boundary\"" in report.read_text(encoding="utf-8")
```

- [x] **Step 2: Run the focused test and verify it fails for missing report sections**

Run: `python3 -m unittest -v tests.test_environment_transition_analysis`
Expected: FAIL with the first missing section ID.

- [x] **Step 3: Render the responsive transition UI and update the archive index**

```python
render_transition_report(data, REPORT_ROOT / "environment-transition.html")
update_transition_index(INDEX_PATH)
```

- [x] **Step 4: Run the focused test and verify it passes**

Run: `python3 -m unittest -v tests.test_environment_transition_analysis`
Expected: PASS with card thumbnails, three dated roster columns, strict-comparison labels, and a valid index link.

### Task 3: Verify generated evidence and visual structure

**Files:**
- Modify: `tests/test_environment_transition_analysis.py`
- Verify: `docs/environment-daily_kaggle_top100/environment-transition.html`

**Interfaces:**
- Consumes: the generated page and embedded `transition-data` JSON.
- Produces: a static validation result confirming pages, counts, dates, and evidence boundary language.

- [x] **Step 1: Add a validation assertion for audit scope and source links**

```python
assert payload["strict_comparison_dates"] == ["2026-07-27", "2026-07-28"]
assert payload["audit"]["2026-07-28"]["consecutive_stable_sweeps"] == 2
```

- [x] **Step 2: Run the focused test and verify it passes**

Run: `python3 -m unittest -v tests.test_environment_transition_analysis`
Expected: PASS.

- [x] **Step 3: Run repository-facing checks**

Run: `python3 -m unittest -v tests.test_environment_daily_contract tests.test_environment_transition_analysis && git diff --check`
Expected: PASS with no whitespace errors.

## Self-Review

- Coverage: Tasks 1–3 cover the requested independent analysis page, daily visual language, card-media requirement, evidence boundaries, index navigation, and regression validation.
- Placeholder scan: no unimplemented behavior or unspecified interface remains in the task steps.
- Type consistency: `TransitionData` is the sole data contract between parsing, rendering, and validation.

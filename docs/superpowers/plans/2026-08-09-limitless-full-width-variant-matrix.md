# Limitless Full-Width Variant Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both Limitless matchup matrices readable across a desktop viewport without horizontal scrollbar dragging, and replace the primary matrix's aggregated Dragapult entry with the five frozen Labs variants.

**Architecture:** Keep `snapshot.json` immutable and make the change in the deterministic HTML renderer. Build the headline matrix from `matchups_variant`, select all five Dragapult labels plus the most-played non-Dragapult variants, and apply a desktop-only full-viewport layout while preserving horizontal overflow as a small-screen fallback.

**Tech Stack:** Python 3.11 renderer, frozen JSON snapshot, semantic HTML/CSS, `unittest` contract tests.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve the frozen Limitless snapshot and its published statistics.
- Treat the unqualified Labs `Dragapult` label as `Dragapult（基础/未单列）`; do not claim exact-deck reclassification.
- Keep `docs/environment/limitless.html` byte-identical to `render_report(snapshot)`.
- Preserve unrelated user changes in the dirty worktree.

---

### Task 1: Lock the matrix selection and UI contract

**Files:**
- Modify: `tests/test_environment_limitless.py`

**Interfaces:**
- Consumes: `render_report(snapshot: dict[str, Any]) -> str`.
- Produces: assertions for five explicit Dragapult headers, absence of the aggregated primary matrix source, and full-width matrix CSS hooks.

- [ ] **Step 1: Add a failing report contract test**

```python
def test_matchup_matrices_use_full_width_and_explicit_dragapult_variants(self):
    report = render_report(self.snapshot)
    primary = report.split('id="primary-heatmap"', 1)[1].split('</table>', 1)[0]
    for label in (
        "Dragapult（基础/未单列）",
        "Dragapult Dusknoir",
        "Dragapult Blaziken",
        "Dragapult Dudunsparce",
        "Dragapult Froslass",
    ):
        self.assertIn(label, primary)
    self.assertIn('class="panel matrix-panel" id="matchup-matrix"', report)
    self.assertIn("width:calc(100vw - 24px)", report)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python3 -m unittest -v tests.test_environment_limitless.ReportContractTests.test_matchup_matrices_use_full_width_and_explicit_dragapult_variants`

Expected: FAIL because the primary matrix still uses `matchups_primary` and the matrix section is constrained by `.page`.

- [ ] **Step 3: Commit the contract**

```bash
git add tests/test_environment_limitless.py
git commit -m "test: define full-width Limitless matrix contract"
```

### Task 2: Render the variant-expanded full-width matrices

**Files:**
- Modify: `data/processed/environment_limitless/render.py`
- Regenerate: `docs/environment/limitless.html`

**Interfaces:**
- Consumes: ordered `labs_variant_meta` and `matchups_variant` rows from the frozen snapshot.
- Produces: a 16×16 `primary-heatmap` containing every Dragapult variant and the eleven most-played non-Dragapult variants; a full-width `dragapult-heatmap`; compact visible W-L-D and sample counts with complete details in cell tooltips.

- [ ] **Step 1: Select the explicit headline matrix labels**

```python
drag_ids = [row["deck_id"] for row in drag_variants]
headline_ids = drag_ids + [
    row["deck_id"]
    for row in variant_meta
    if row["deck_id"] not in drag_ids
][: 16 - len(drag_ids)]
matrix_names = {row["deck_id"]: row["name"] for row in variant_meta}
matrix_names["dragapult-ex"] = "Dragapult（基础/未单列）"
```

- [ ] **Step 2: Render the headline matrix from variant matchup facts**

Pass `headline_ids`, `matrix_names`, and `snapshot["matchups_variant"]` to `_matrix_table`; update the section title and evidence-boundary copy so it explicitly says the matrix uses Labs labels.

- [ ] **Step 3: Compact each cell without removing evidence**

Render W-L-D and `n` on separate short lines. Keep the exact record, Wilson interval, and evidence grade in `title` and `aria-label`.

- [ ] **Step 4: Add desktop full-viewport layout**

Add `matrix-panel` to both matrix sections. At desktop widths, break those sections out of `.page`, set the matrix table to fixed layout and remove per-column minimum widths. Keep `.matrix-wrap{overflow:auto}` and restore a minimum table width below the desktop breakpoint so phones and narrow windows remain usable.

- [ ] **Step 5: Regenerate the checked-in report**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from data.processed.environment_limitless.render import render_report

snapshot = json.loads(Path("data/processed/environment_limitless/snapshot.json").read_text())
Path("docs/environment/limitless.html").write_text(render_report(snapshot), encoding="utf-8")
PY
```

- [ ] **Step 6: Run the focused and full contracts**

Run: `python3 -m unittest -v tests.test_environment_limitless.ReportContractTests`

Run: `python3 -m unittest -v tests.test_environment_limitless`

Expected: all tests PASS, including checked-in HTML equality and frozen statistical audits.

- [ ] **Step 7: Perform static HTML checks**

Run: `rg -n 'matrix-panel|Dragapult（基础/未单列）|id="primary-heatmap"|id="dragapult-heatmap"' docs/environment/limitless.html`

Expected: both sections have `matrix-panel`, and the explicit base/unspecified label occurs in the matrices.

- [ ] **Step 8: Commit the renderer and generated report**

```bash
git add data/processed/environment_limitless/render.py docs/environment/limitless.html
git commit -m "feat: expand Limitless matchup matrices"
```

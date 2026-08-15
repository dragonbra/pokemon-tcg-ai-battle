# 0045 V8 Dual-Taxonomy Evaluation Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one auditable 0045 longitudinal HTML report for every comparable U200–U298 Policy-0809 three-pool evaluation, with separate current 29-class and Value-network 15-class Meta tables plus trade-off analysis.

**Architecture:** Read every PASS `policy0809_three_pool_cuda512` report from the comparable V4 U200–U275, V7 U276–U281, and V8 U282–U298 lineages. Validate deployment/opponent/schedule identity, aggregate the original game entries by current `archetype_id` and by the actual training-time 15-way opponent target generated from each deck's exact 60 cards through `OpponentArchetypeTaxonomy.classify_target`, derive pairwise first-difference trade-off diagnostics, then render a new immutable experiment HTML and add it to the 0045 evaluation index. The V7 CUDA-2048 snapshot remains untouched.

**Tech Stack:** Python 3 standard library, JSON, standalone HTML/CSS, pytest.

## Global Constraints

- Do not overwrite or relabel the historical V7 CUDA-2048 reports.
- Every consumed evaluation must be `PASS`, 1,536/1,536 terminal, zero error/unfinished, candidate deployment identity `PASS`, and complete immutable `Policy-0809` opponent identity `PASS`.
- Current taxonomy is `own_archetypes_v2`; legacy taxonomy is the frozen `0042_own_archetypes_v1` used by the original 15-class Value classification loss.
- Current 29-way aggregation must use the audited `deck_own_archetype_mapping_v2.json` exact deck mapping; legacy Value-loss aggregation must use the frozen `assets/meta/opponent_archetypes_v1.json` trigger classifier over the same registry deck's exact 60 cards. `old_archetype_id` is own-policy migration provenance and must not impersonate the opponent Meta training target.
- The report must label three-pool CUDA-512 evaluation as a 1,536-game diagnostic and must not impersonate Benchmark V2 CUDA-2048.
- Meta `02/03/05` and `00/01/04/27` retain their two existing highlight groups.

---

### Task 1: Dual-taxonomy aggregation and validation

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/render_v8_three_pool_history.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_v8_three_pool_history_report.py`

**Interfaces:**
- Consumes: V8 `artifact/periodic_evaluation/update-*/report.json`, `own_archetypes_v2.json`, `deck_own_archetype_mapping_v2.json`, deck registry exact 60-card files, and `assets/meta/opponent_archetypes_v1.json` through the project-local classifier.
- Produces: `load_history(report_paths: tuple[Path, ...]) -> HistoryAudit` and deterministic current/legacy aggregate rows.

- [ ] **Step 1: Write failing tests** proving exact update inventory, hard failure for an unmapped deck, current-table totals of 1,536 games per update, Value-loss-table totals of 1,536 games per update, and exact parity between every report entry's deck-derived 15-way label and a fresh call to `OpponentArchetypeTaxonomy.classify_target`.
- [ ] **Step 2: Run the focused test** with `python3 -m pytest train/0045_single_deck_expert_minimal_lora/tests/test_v8_three_pool_history_report.py -q`; expect failure because the renderer module does not exist.
- [ ] **Step 3: Implement strict report validation and entry-level aggregation**. Reject any missing update, failed audit, schedule mismatch, mismatched entry Meta, or unmapped deck ID.
- [ ] **Step 4: Run the focused test** and require PASS.

### Task 2: Render and publish the V8 HTML

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/evaluation/render_v8_three_pool_history.py`
- Create: `experiments/0045_single_deck_expert_minimal_lora/evaluation/V8_dragapult_007_u200_u298_policy0809_three_pool_cuda512.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/evaluation/index.html`
- Modify: `train/0045_single_deck_expert_minimal_lora/tests/test_v8_three_pool_history_report.py`

**Interfaces:**
- Consumes: validated `HistoryAudit` from Task 1.
- Produces: `render(history: HistoryAudit) -> str`, a standalone HTML report, and an immutable index entry.

- [ ] **Step 1: Extend tests** to require all 31 U200–U298 comparable columns, pool summary rows, all 29 current rows, all 15 Value-loss rows, W-L-D plus percentages, constituent V2 class IDs, within-class U298 range diagnostics, and a ranked first-difference trade-off table.
- [ ] **Step 2: Implement responsive tables** with sticky Meta/name columns, horizontal scrolling, target/protection highlights, provenance notes, and embedded compact audit JSON.
- [ ] **Step 3: Generate the new V8 report atomically** and add its summary row to the evaluation index without changing historical HTML.
- [ ] **Step 4: Run focused and full 0045 tests**; require no regressions.

### Task 3: Visual and arithmetic verification

**Files:**
- Verify: `experiments/0045_single_deck_expert_minimal_lora/evaluation/V8_dragapult_007_u282_u298_policy0809_three_pool_cuda512.html`

**Interfaces:**
- Consumes: generated HTML from Task 2.
- Produces: verified standalone report ready for browser inspection.

- [ ] **Step 1: Parse the HTML** and assert exactly one current-29 table, one legacy-15 table, nine update columns, and no external script dependency.
- [ ] **Step 2: Cross-check arithmetic**: every update totals 1,536 in both taxonomies; U298 equals 990-546-0 overall; the latest report/path and all identity statuses are embedded.
- [ ] **Step 3: Render a browser screenshot or equivalent local visual inspection** and verify wide-table scrolling, highlight colors, sticky labels, and mobile readability.

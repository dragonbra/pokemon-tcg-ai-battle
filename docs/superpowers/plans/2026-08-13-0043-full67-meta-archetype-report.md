# 0043 Full-67 Meta Archetype Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-focal-deck matchup performance grouped by the audited 29-class 0043 deck taxonomy to every U407 Full-67 report detail page.

**Architecture:** The renderer loads `own_archetypes_v2.json` and `deck_own_archetype_mapping_v2.json` as a reporting-only taxonomy, maps every real per-game `opponent_id` to exactly one class, and aggregates the independently scheduled G1 and U407 arms. Each class retains its complete 001–067 membership plus the actually observed opponent-deck subset; this analysis never changes the frozen 15-way opponent Meta prediction head or model weights.

**Tech Stack:** Python 3.11, JSON, static HTML, pytest.

## Global Constraints

- Use only real completed CUDA-2048 game entries; do not infer outcomes from deck names.
- Use `own_archetypes_v2` only as an offline reporting taxonomy; do not alter actor/value inputs or the 15-way opponent Meta head.
- Require exactly one taxonomy mapping for every deck 001–067 and fail closed on unknown per-game opponent IDs.
- Show no-sample classes as “无样本”, never as 0% win rate.
- G1 and U407 schedules are identity-bound independent samples; do not present class deltas as paired estimates.
- Preserve all candidate, Policy-0809, exact-deck and CUDA identity evidence.

---

### Task 1: Taxonomy-backed matchup aggregation

**Files:**
- Modify: `train/0043_champion_league_rl/evaluation/render_g2_candidate_gate.py`
- Modify: `train/0043_champion_league_rl/tests/test_g2_candidate_gate.py`

**Interfaces:**
- Consumes: `own_archetypes_v2.json`, `deck_own_archetype_mapping_v2.json`, and real report `entries`.
- Produces: `_meta_taxonomy()`, `_by_meta_archetype(report, taxonomy)`, and row-level `by_meta_archetype` payloads.

- [ ] Write tests requiring complete 001–067 coverage, exact class membership, outcome conservation and unknown-opponent hard failure.
- [ ] Implement the taxonomy loader and per-arm aggregation.
- [ ] Run the focused renderer tests.

### Task 2: Detail-page analysis UI

**Files:**
- Modify: `train/0043_champion_league_rl/evaluation/render_g2_candidate_gate.py`
- Modify: `train/0043_champion_league_rl/evaluation/render_g2_candidate_full67.py`

**Interfaces:**
- Consumes: row-level `by_meta_archetype` payloads.
- Produces: one 29-row matchup table per focal-deck detail page with class membership and observed-deck coverage.

- [ ] Render G1/U407 sample size, W-L-D, win rate and independent-arm delta for every class.
- [ ] List full class deck IDs and observed opponent IDs, and mark zero-game classes as “无样本”.
- [ ] Republish all 67 detail pages at the existing user-selected path.
- [ ] Validate 67 pages, 29 class rows per page, embedded JSON, and aggregate game-count conservation.

# Current Top 500 Deck Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the currently visible Kaggle Top 500, retain up to two replay-proven distinct decks per player, and publish an auditable score-band deck-distribution report against exact decks 001-070 and the 29-class Own Archetype V2 taxonomy.

**Architecture:** Add a ranked-snapshot pipeline beside the Top 100 daily pipeline. It reuses the existing score-bound submission and replay parsers, but keeps a separate Top 500 evidence directory and schema; classification first performs canonical exact-list matching against the 0045 registry, then assigns non-catalog decks to the nearest registered 29-meta class with the complete similarity audit retained in JSON. A deterministic renderer consumes only the frozen JSON and produces the interactive HTML report.

**Tech Stack:** Python 3, Kaggle Python API, JSON/CSV, stdlib HTML rendering, pytest/unittest contract tests.

## Global Constraints

- Do not submit, retry a submission, upload a dataset, or perform another externally visible Kaggle action.
- Do not modify `engine/source/`.
- Freeze leaderboard rows before Episode/replay collection and never reuse an old snapshot as current evidence.
- Freeze rank, score, and leaderboard submission time from one server-generated leaderboard CSV. Bind to an equal-`publicScore` active submission where available; otherwise retain the highest scored active submission present at the cutoff, record its score delta, and label the row as dynamic evidence rather than claiming it uniquely produced the frozen leaderboard score.
- Each displayed deck must come from the matching submission's own unique player index in a `PUBLIC + COMPLETED` Episode replay and contain exactly 60 cards.
- Count score-band people once per leaderboard row; report primary-deck distribution separately from all retained deck evidence so a two-deck player is never silently double-counted.
- Exact 001-070 identity takes precedence over 29-meta classification.
- Non-catalog report IDs use `00_<meta_slug>_Pnn`, with a stable per-report sequence inside each meta class.
- Score bands are `1000+`, `900-1000`, `800-900`, and `700-800`; stop at rank 500 or the lowest available score, and state explicitly when 700 is not reached.
- Raw evidence stays under `.tmp/environment_daily/2026-08-16/`; the frozen ranked JSON and formal HTML live under `docs/environment-daily_kaggle_top100/ranked/`.

---

### Task 1: Ranked Snapshot Contracts

**Files:**
- Create: `data/processed/environment_daily/generate_current_top500_report.py`
- Test: `tests/test_environment_top500_contract.py`

**Interfaces:**
- Consumes: score/deck helpers from `data.processed.environment_daily.generate_live_snapshot` and the 0045 deck registry/taxonomy assets.
- Produces: `collect_snapshot(work: Path, snapshot_path: Path) -> dict[str, object]`, `classify_decks(payload: dict[str, object]) -> None`, and `validate_snapshot(payload: dict[str, object], replay_root: Path) -> dict[str, object]`.

- [x] **Step 1: Write failing tests for exact matching, unknown-meta assignment, score bands, dual-deck deduplication, and fail-closed 60-card validation.**
- [x] **Step 2: Run `python3 -m pytest -q tests/test_environment_top500_contract.py` and verify the new module is missing.**
- [x] **Step 3: Implement immutable snapshot collection, canonical deck hashing, 001-070 loading, 29-meta similarity classification, stable `Pnn` IDs, and validation.**
- [x] **Step 4: Run `python3 -m pytest -q tests/test_environment_top500_contract.py` and require all tests to pass.**

### Task 2: Deterministic Report Renderer

**Files:**
- Modify: `data/processed/environment_daily/generate_current_top500_report.py`
- Modify: `tests/test_environment_top500_contract.py`

**Interfaces:**
- Consumes: a validated `pokemon_tcg_current_top500_decks_v1` snapshot.
- Produces: `render_report(payload: dict[str, object], output: Path) -> None` with summary metrics, score-band population, 001-070 chart with card thumbnails, 29-meta fallback distribution, ranked player table, submission timestamps, and expandable exact decks.

- [x] **Step 1: Add failing HTML contract tests for all required sections, 70 deck columns, score/submission fields, card images, and dual-deck rendering.**
- [x] **Step 2: Implement the responsive report using the established ranked/Top 100 visual language and deterministic embedded data.**
- [x] **Step 3: Run the focused tests and `tests.test_environment_daily_contract`.**

### Task 3: Current Kaggle Freeze And Audit

**Files:**
- Create: `.tmp/environment_daily/2026-08-16/run-*/snapshot.json`
- Create: `.tmp/environment_daily/2026-08-16/run-*/representative_replays/episode-*.json`
- Create: `docs/environment-daily_kaggle_top100/ranked/data/2026-08-16-top500-exact.json`
- Create: `docs/environment-daily_kaggle_top100/ranked/2026-08-16-top500.html`
- Modify: `docs/environment-daily_kaggle_top100/README.md`

**Interfaces:**
- Consumes: authenticated read-only Kaggle leaderboard, team submission, Episode Meta, and replay endpoints.
- Produces: a frozen 500-row report, or the maximum rank returned by Kaggle if fewer than 500 rows exist, with an explicit coverage boundary.

- [x] **Step 1: Verify Kaggle credentials without printing the token and freeze the live leaderboard.**
- [x] **Step 2: Bind strict rows to their score-producing submission and label drifted rows as active-submission evidence, then collect primary and alternate replay-proven distinct decks.**
- [x] **Step 3: Validate rank uniqueness, binding semantics, timestamp cutoff, own-player replay identity, exact 60-card decks, hashes, score-band totals, and classification totals.**
- [x] **Step 4: Render the report, update the ranked-report README entry, and run the complete focused contract suite.**
- [x] **Step 5: Inspect the HTML at desktop and mobile widths when a browser runner is available; otherwise perform structural HTML assertions and report the visual-test limitation.**

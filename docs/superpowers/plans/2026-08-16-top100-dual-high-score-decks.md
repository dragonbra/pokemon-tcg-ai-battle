# Top 100 Dual High-Score Decks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the current Kaggle Top 100 and show up to two highest-scoring distinct audited decks for each ranked identity, with score and snapshot-relative submission age beside that identity.

**Architecture:** Keep the leaderboard-best submission as the canonical row for rank, matchup, and win-rate statistics. During collection, walk each team’s scored submissions in descending score order, materialize only enough cutoff-bounded public Episodes to discover two distinct exact deck hashes, and attach those immutable deck evidences to the player record. Rendering uses this separate evidence list without mixing alternate-submission games into leaderboard statistics.

**Tech Stack:** Python 3, Kaggle API, official Episode replay JSON, deterministic static HTML, `unittest`.

## Global Constraints

- Never modify `engine/source/`.
- A leaderboard row remains bound only by exact `leaderboard score = submission publicScore`; date is only a same-score tiebreaker.
- Every displayed alternate deck must trace to a scored submission, a cutoff-bounded `PUBLIC + COMPLETED` Episode, the submission’s unique player index, and an exact 60-card replay deck.
- Relative ages are computed against `captured_at_utc`; the exact UTC submission timestamp remains in HTML metadata/title text.
- Existing historical reports are immutable; publish only `daily/2026-08-16.html` and update the index in descending date order.

---

### Task 1: Distinct-deck evidence selection

**Files:**
- Modify: `data/processed/environment_daily/generate_live_snapshot.py`
- Test: `tests/test_environment_daily_contract.py`

**Interfaces:**
- Consumes: Kaggle team submission objects, `captured_at_utc`, existing Episode/replay helpers.
- Produces: `player["high_score_decks"]: list[dict]`, ordered by score descending and deduplicated by full deck SHA-256, with at most two entries.

- [ ] **Step 1: Write failing tests** for descending score order, distinct-deck deduplication, cutoff-bounded Episode selection, and a single-deck fallback.
- [ ] **Step 2: Run** `python3 -m unittest tests.test_environment_daily_contract -v` and confirm the new tests fail for missing selectors.
- [ ] **Step 3: Implement** a pure scored-submission ordering helper and a collector that stops after two distinct exact decks while preserving submission ID, public score, submitted-at timestamp, Episode ID/index, deck hash, archetype, and 60-card list.
- [ ] **Step 4: Extend snapshot validation** so every displayed high-score deck is independently replay-verified and the first item equals the canonical leaderboard deck.
- [ ] **Step 5: Re-run** `python3 -m unittest tests.test_environment_daily_contract -v` and confirm selection and validation tests pass.

### Task 2: Relative submission age and dual-deck UI

**Files:**
- Modify: `data/processed/environment_daily/generate_live_snapshot.py`
- Modify: `docs/environment-daily_kaggle_top100/README.md`
- Test: `tests/test_environment_daily_contract.py`

**Interfaces:**
- Consumes: `player["high_score_decks"]`, `captured_at_utc`.
- Produces: stable Chinese relative-age labels and identity/deck evidence markup in the rank, personal, and detail views.

- [ ] **Step 1: Write failing tests** for minute/hour/day age boundaries, future/invalid fail-closed behavior, exact timestamp metadata, and two visible deck panels for a dual-deck player.
- [ ] **Step 2: Run** the focused tests and confirm they fail before implementation.
- [ ] **Step 3: Implement** snapshot-relative labels such as `3 小时前` and `2 天前`, with the exact UTC timestamp in a `title`/`datetime` attribute.
- [ ] **Step 4: Render** score and relative submission age beside each identity, and render both audited high-score decks only when two distinct hashes exist; preserve the canonical 100-row/100-detail contract.
- [ ] **Step 5: Update** the README evidence boundary so alternate decks cannot be mistaken for leaderboard rank or matchup evidence.
- [ ] **Step 6: Run** the full daily contract suite and inspect generated fixture markup.

### Task 3: Live freeze, publication, and audit

**Files:**
- Create: `.tmp/environment_daily/2026-08-16/run-*/snapshot.json` and associated ignored evidence files.
- Create: `docs/environment-daily_kaggle_top100/daily/2026-08-16.html`
- Modify: `docs/environment-daily_kaggle_top100/index.html`

**Interfaces:**
- Consumes: the canonical generator CLI and authenticated Kaggle API.
- Produces: one complete immutable live snapshot, one published report, and an updated descending index.

- [ ] **Step 1: Verify** Kaggle authentication is present without printing credentials and confirm the target report does not already exist.
- [ ] **Step 2: Run** `python3 -m data.processed.environment_daily.generate_live_snapshot --date 2026-08-16` and allow the monotonic Episode union to reach two stable full sweeps.
- [ ] **Step 3: Run** the generator in `--validate-only` mode against the completed run and require the full identity/deck audit to pass.
- [ ] **Step 4: Run** `python3 -m unittest tests.test_environment_daily_contract -v` and perform structural counts for 100 leaderboard rows, 100 player details, 60 cards per canonical deck, dual-deck panels, card images, required section IDs, relative ages, and descending index order.
- [ ] **Step 5: Review** `git diff --check` and report the published HTML and evidence paths without committing unrelated workspace changes.


# Daily Deck Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible root-level `deck/` catalog containing one self-contained exact 60-card deck per unique full `deck_sha256` from the latest audited Kaggle daily report, with a readable card-image index.

**Architecture:** A local generator reads only the frozen 2026-07-30 daily HTML and the read-only official card catalog. It groups the 100 audited leaderboard entries by full deck hash, writes ASCII snake_case directories named from readable key Pokemon names plus a per-archetype sequence, emits exact 60-line `deck.csv` files and manifests, and renders a compact report UI consistent with the daily report's card-image language.

**Tech Stack:** Python 3.11 standard library, official `data/official/EN_Card_Data.csv`, static HTML/CSS/JavaScript.

## Global Constraints

- Use the frozen local daily snapshot only; do not query Kaggle or mutate external state.
- Treat the full 64-character `deck_sha256` as the unique deck identity; displayed 12-character hashes are labels only.
- Keep `deck.csv` as exact 60 lines of engine Card IDs; use human-readable Pokemon names for directory names and UI.
- Do not modify `engine/source/` or unrelated user changes.
- Preserve provenance, rank, score, record, archetype, and source report in manifests.

### Task 1: Reproducible extraction and catalog generation

**Files:**
- Create: `deck/build_catalog.py`
- Create: `deck/README.md`
- Create: `deck/index.html`
- Create: `deck/manifest.json`
- Create: `deck/<key_pokemon_names>_<NNN>/deck.csv`
- Create: `deck/<key_pokemon_names>_<NNN>/manifest.json`

**Interfaces:**
- `python3 deck/build_catalog.py --report docs/environment-daily_kaggle_top100/daily/2026-07-30.html --output deck`
- The generator consumes the report's `snapshot-audit` JSON and `data-player-detail` exact deck cards.
- It produces one directory per unique full hash and an index that exposes `window.DECK_CATALOG`.

- [ ] Parse all 100 audit rows and all 100 exact deck details.
- [ ] Group by full hash, reject missing or inconsistent 60-card contents.
- [ ] Name directories from readable key Pokemon card names and sequence them within each key.
- [ ] Write manifests, exact Card-ID deck files, and the index UI.

### Task 2: Verification

**Files:**
- Test: `deck/build_catalog.py --check`

- [ ] Verify 100 audited rows, unique hash grouping, 60 cards per deck, no duplicate directory names, and every card ID in the official catalog.
- [ ] Verify each generated manifest hash recomputes from sorted deck IDs using the daily report contract.
- [ ] Verify the index includes every generated deck and displays source/rank/score/record metadata.

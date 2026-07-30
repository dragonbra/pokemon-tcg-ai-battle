# 0022 Curated Deck Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the `0022` exact-deck catalog with corrected human-readable Kaggle classifications plus separately identified Limitless and official arena reference decks.

**Architecture:** The frozen 2026-07-30 daily report remains the sole source for the 100-row, 32-hash Kaggle section. External references use exact 60-card local deck files and explicit provenance manifests; the static index renders both groups with the same official-card lookup and card-image components while never assigning Kaggle rank, score, usage, or win-rate facts to external references.

**Tech Stack:** Python 3.11 standard library, JSON manifests, official `data/official/EN_Card_Data.csv`, static HTML/CSS/JavaScript.

## Global Constraints

- Keep the Kaggle totals at exactly 100 audited rows and 32 unique full hashes.
- Treat Limitless and arena decks as external references, not additional Kaggle hashes.
- Every `deck.csv` must contain exactly 60 integer Card IDs and its sorted-ID hash must match its manifest.
- Preserve `engine/source/` and all unrelated user changes.
- Use English key Pokemon names in directory names and readable card names/images in `index.html`.

---

### Task 1: Correct Kaggle identities

**Files:**
- Modify: `train/0022_league_training/deck/build_catalog.py`
- Modify: `train/0022_league_training/deck/manifest.json`
- Modify: `train/0022_league_training/deck/*/manifest.json`

**Interfaces:**
- Consumes: the daily report's exact full hash and card list.
- Produces: `raging_bolt_ex_james_cox_henry_chao_001`, `dragapult_ex_dunsparce_001`, and contiguous Kangaskhan/Crustle numbering.

- [x] Classify James Cox & Henry Chao's exact hash as `Raging Bolt ex / Mega Kangaskhan ex`.
- [x] Classify the Kaggle Dragapult list containing Dunsparce as `Dragapult ex / Dunsparce`.
- [x] Synchronize every directory field and reject stale paths.

### Task 2: Add external references

**Files:**
- Modify: `train/0022_league_training/deck/dragapult_ex_limitless/manifest.json`
- Create: `train/0022_league_training/deck/ionos_bellibolt_ex_kilowattrel_01/deck.csv`
- Create: `train/0022_league_training/deck/ionos_bellibolt_ex_kilowattrel_01/manifest.json`
- Create: `train/0022_league_training/deck/mega_lucario_ex_solrock_002/deck.csv`
- Create: `train/0022_league_training/deck/mega_lucario_ex_solrock_002/manifest.json`
- Create: `train/0022_league_training/deck/ns_zoroark_ex_001/deck.csv`
- Create: `train/0022_league_training/deck/ns_zoroark_ex_001/manifest.json`
- Modify: `train/0022_league_training/deck/README.md`

**Interfaces:**
- Consumes: the selected Limitless tournament list and `evaluation/arena/opponents/ionos_bellibolt_ex_kilowattrel_01/deck.csv`.
- Produces: four explicit `external_references` entries with exact hashes and provenance.

- [x] Preserve the Limitless list's 1-1 Dunsparce/Dudunsparce tech note.
- [x] Copy the arena deck byte-for-byte and identify it as an official recorded arena reference useful as a benchmark target.
- [x] Import the user-provided Lucario list and replace the mis-pasted second list with the corrected exact N's Zoroark ex list.
- [x] Do not attach Kaggle leaderboard metrics to any external reference.

### Task 3: Render and verify the combined catalog

**Files:**
- Modify: `train/0022_league_training/deck/index.html`
- Modify: `train/0022_league_training/deck/manifest.json`

**Interfaces:**
- Consumes: 32 Kaggle manifests and two external-reference manifests.
- Produces: one filterable card-image page with visibly separate evidence classes.

- [x] Render all 36 exact decks using official card names and card images.
- [x] Display Kaggle usage/rank/score/record only for Kaggle decks and source facts only for external references.
- [x] Verify 100 Kaggle memberships, 32 unique Kaggle hashes, four external references, 36 exact 60-card files, matching hashes, valid links, and no stale names.

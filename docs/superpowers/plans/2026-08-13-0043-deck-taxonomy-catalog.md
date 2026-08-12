# 0043 Deck Taxonomy Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct deck 062's formal display classification and publish auditable text/HTML profiles for all 65 exact decks inside the 0043 experiment assets.

**Architecture:** The immutable deck ID/content/hash remain unchanged; a content-bound display-name correction overlays the erroneous upstream label while retaining upstream provenance. A project-local generator joins the 65-deck registry, Own Archetype V2 registry/mapping, and official card data into 65 plain-text profiles plus one indexed HTML report.

**Tech Stack:** Python 3.11, JSON, CSV, HTML, SHA-256, pytest.

## Global Constraints

- Deck 062 content, deck ID, and content SHA-256 must not change.
- The corrected display name is `Mega Starmie ex / Mega Froslass ex`; the erroneous upstream label remains provenance only.
- Own Archetype V2 is actor conditioning and must not be confused with the unchanged opponent Meta classifier.
- Every text profile must enumerate exact card ID, card name, count, category, and sum to exactly 60 cards.
- Every profile must record taxonomy version/hash, Own Archetype ID/name, decision, parent, strategic axis, definition, rationale, tags, exact deck hash, and source provenance.
- Reports are deterministic project assets under `experiments/0043_champion_league_rl/deck_taxonomy/`.

---

### Task 1: Content-Bound 062 Display Correction

**Files:**
- Modify: `train/0043_champion_league_rl/import_frozen_pool65.py`
- Modify: `train/0043_champion_league_rl/assets/decks/registry.json`
- Test: `train/0043_champion_league_rl/tests/test_deck_taxonomy_catalog.py`

**Interfaces:**
- Produces: stable corrected `name`/`archetype` for content hash `654f54...c933` while preserving ZIP source manifest/name.

- [ ] Assert 062 remains the same 60 cards and hash.
- [ ] Apply the exact-content display override in the idempotent importer.
- [ ] Assert no other deck's ID/content/name is unintentionally changed.

### Task 2: Deterministic 65-Deck Text and HTML Generator

**Files:**
- Create: `train/0043_champion_league_rl/generate_deck_taxonomy_catalog.py`
- Create: `experiments/0043_champion_league_rl/deck_taxonomy/text/001.txt` through `065.txt`
- Create: `experiments/0043_champion_league_rl/deck_taxonomy/index.html`
- Create: `experiments/0043_champion_league_rl/deck_taxonomy/manifest.json`

**Interfaces:**
- Consumes: exact deck registry, V2 taxonomy/mapping, `data/official/EN_Card_Data.csv`.
- Produces: human-readable per-deck text profiles and linked/filterable HTML tables with deterministic asset hashes.

- [ ] Join exact cards to official names/categories without changing deck order/content semantics.
- [ ] Render one complete text profile per deck with 60-card totals and strategy metadata.
- [ ] Render overview/archetype distribution and expandable 65-deck details in HTML.
- [ ] Write a manifest pinning every text file and source registry/taxonomy hash.

### Task 3: Synchronize Audit and Validate

**Files:**
- Modify: `experiments/0043_champion_league_rl/OWN_ARCHETYPE_TAXONOMY_V2_AUDIT.md`
- Modify: `experiments/0043_champion_league_rl/OWN_ARCHETYPE_TAXONOMY_V2_AUDIT.html`
- Modify: `experiments/0043_champion_league_rl/manifest.json`
- Test: `train/0043_champion_league_rl/tests/test_deck_taxonomy_catalog.py`

**Interfaces:**
- Produces: consistent 062 naming in all current reports and a hard gate for 65/65 profiles, exact counts, hashes, classes, and links.

- [ ] Regenerate the taxonomy audit from corrected registry facts.
- [ ] Test 65 text files, 65 HTML details, exact 60 counts, mapping identity, and deterministic regeneration.
- [ ] Run the complete 0043 test suite and preserve unrelated user/CUDA worktree changes.

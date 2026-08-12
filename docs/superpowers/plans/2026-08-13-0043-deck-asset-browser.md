# 0043 Deck Asset Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `train/0043_champion_league_rl/assets/decks/` the self-contained, human-readable source for the 65 exact decks, with a Policy-0809 CUDA-2048-style catalog and image-rich per-deck pages.

**Architecture:** Extend the existing deterministic taxonomy catalog generator so its canonical output is the deck asset tree. Keep `registry.json` and `definitions/<ID>/deck.csv` authoritative, add generated `index.html`, `manifest.json`, `text/<ID>.txt`, and `definitions/<ID>/index.html`, and render card art from official expansion/collection metadata using the same public image conventions as the Policy-0809 report.

**Tech Stack:** Python standard library, static HTML/CSS/JavaScript, pytest, existing 0043 asset/taxonomy registries.

## Global Constraints

- Do not modify `engine/source/`.
- Keep all 65 exact deck hashes and 60-card files unchanged.
- The displayed class is Own Deck Strategy Archetype V2; do not relabel it as the independent opponent Meta classifier.
- Use deterministic generation and hash every generated lookup artifact.
- Preserve the Policy-0809 CUDA-2048 visual language: dark green header, pale green-gray canvas, summary metrics, representative-card thumbnails, compact catalog, and responsive layout.
- Card images may use the same external Pokémon TCG / Scrydex image endpoints used by existing reports; render an explicit fallback when an image is unavailable.

---

### Task 1: Define asset-browser acceptance tests

**Files:**
- Modify: `train/0043_champion_league_rl/tests/test_deck_taxonomy_catalog.py`

**Interfaces:**
- Consumes: `assets/decks/registry.json`, `assets/decks/definitions/<ID>/deck.csv`.
- Produces: assertions for canonical catalog and detail artifacts.

- [ ] Add a test requiring `assets/decks/index.html`, `assets/decks/manifest.json`, 65 text profiles, and 65 `definitions/<ID>/index.html` pages.
- [ ] Require representative card images and full card-grid images in the HTML.
- [ ] Require every detail page to show exact hash, Own Meta classification, rationale, 60-card total, and a link to its physical `deck.csv`.
- [ ] Run `pytest -q train/0043_champion_league_rl/tests/test_deck_taxonomy_catalog.py` and confirm the new assertions fail before implementation.

### Task 2: Generate the canonical asset browser

**Files:**
- Modify: `train/0043_champion_league_rl/generate_deck_taxonomy_catalog.py`
- Create: generated files under `train/0043_champion_league_rl/assets/decks/`

**Interfaces:**
- Consumes: `AssetRegistry`, `OwnArchetypeVocabulary`, `data/official/EN_Card_Data.csv`.
- Produces: deterministic index, detail pages, text profiles, and content manifest.

- [ ] Add expansion-to-image URL mapping and deterministic representative-card selection.
- [ ] Render the green Policy-0809-style top-level catalog with thumbnail pairs, filters, taxonomy distribution, and links to detail pages.
- [ ] Render one responsive detail page per deck with identity metrics, classification rationale, strategic characteristics, composition, image-rich card tiles, clickable large preview, provenance, and navigation.
- [ ] Write all generated hashes to `assets/decks/manifest.json` and ensure each physical deck directory remains the canonical location of `deck.csv`.
- [ ] Run the focused test and confirm it passes.

### Task 3: Update project lookup links and verify determinism

**Files:**
- Modify: `experiments/0043_champion_league_rl/DESIGN.md`
- Modify: `experiments/0043_champion_league_rl/DESIGN.html`
- Modify: `experiments/0043_champion_league_rl/OWN_ARCHETYPE_TAXONOMY_V2_AUDIT.md`
- Modify: `experiments/0043_champion_league_rl/OWN_ARCHETYPE_TAXONOMY_V2_AUDIT.html`

**Interfaces:**
- Consumes: canonical asset-browser entry point.
- Produces: accurate project documentation links.

- [ ] Point project documentation at `train/0043_champion_league_rl/assets/decks/index.html` as the primary browser.
- [ ] Regenerate twice and assert the full generated-file hash map is unchanged.
- [ ] Run the full `train/0043_champion_league_rl/tests` suite.
- [ ] Run `git diff --check` and verify all 65 physical `deck.csv` hashes still match `registry.json`.

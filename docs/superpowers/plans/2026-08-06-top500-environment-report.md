# Top 500 Environment Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the supplied 2026-08-06 Top 500 exact-deck snapshot into an auditable, image-rich ranked metagame report focused on deck construction rather than 15-game reward samples.

**Architecture:** Add one deterministic Python renderer under the existing environment-daily package. It validates the frozen JSON, derives rank bands, archetypes, identical 60-card variants, card-pool statistics, and notable outliers, then emits one self-contained HTML report under a new `ranked/` documentation section. The raw snapshot remains beside the report as provenance; the old rough HTML is retained as a source artifact rather than treated as authoritative output.

**Tech Stack:** Python 3.11 standard library, official `EN_Card_Data.csv`, static HTML/CSS/JavaScript, `unittest` contract checks.

## Global Constraints

- Do not use the recent 15-game reward sample as deck-strength evidence.
- Compute only Top 100 and Top 500 deck distributions from this 500-deck snapshot; mark Top 1000 and Top 5000 unavailable.
- Preserve and disclose score-binding mismatches instead of presenting all rows as a strict leaderboard-to-submission identity closure.
- Every deck must contain exactly 60 card IDs and every rank from 1 through 500 must appear exactly once.
- Reuse the established daily-report card image and preview conventions.
- Do not modify `engine/source/` or perform any externally visible Kaggle action.

---

### Task 1: Deterministic ranked report renderer

**Files:**
- Create: `data/processed/environment_daily/generate_ranked_deck_report.py`
- Create: `tests/test_ranked_environment_report.py`

**Interfaces:**
- Consumes: `render_report(snapshot_path: Path, output_path: Path) -> dict[str, object]`
- Produces: self-contained HTML and a validation summary.

- [x] **Step 1: Write contract tests for snapshot validation and required report sections**
- [x] **Step 2: Run the focused tests and confirm the renderer is initially missing**
- [x] **Step 3: Implement validation, aggregation, card metadata, and HTML rendering**
- [x] **Step 4: Run the focused tests and fix all failures**

### Task 2: Publish the frozen snapshot and report

**Files:**
- Create: `docs/environment-daily_kaggle_top100/ranked/data/2026-08-06-top500-exact.json`
- Create: `docs/environment-daily_kaggle_top100/ranked/source/2026-08-06-top500-rough.html`
- Create: `docs/environment-daily_kaggle_top100/ranked/2026-08-06-top500.html`
- Modify: `docs/environment-daily_kaggle_top100/index.html`
- Modify: `docs/environment-daily_kaggle_top100/README.md`

**Interfaces:**
- Consumes: the renderer from Task 1 and the two user-supplied root files.
- Produces: a discoverable documentation report with preserved provenance.

- [x] **Step 1: Move the two temporary root artifacts into the ranked documentation tree**
- [x] **Step 2: Generate the formal report from the frozen exact-deck JSON**
- [x] **Step 3: Add the ranked-analysis entry to the documentation index and README**
- [x] **Step 4: Regenerate and byte-compare output to verify determinism**

### Task 3: Visual and structural verification

**Files:**
- Verify: `docs/environment-daily_kaggle_top100/ranked/2026-08-06-top500.html`

**Interfaces:**
- Consumes: published static report.
- Produces: verified desktop/mobile rendering and report audit counts.

- [x] **Step 1: Run focused and existing environment-report contract tests**
- [x] **Step 2: Serve the repository locally and capture desktop/mobile screenshots**
- [x] **Step 3: Check navigation, tables, card images, preview behavior, filters, and text overflow**
- [x] **Step 4: Record final Top 100/Top 500 findings and evidence limitations**

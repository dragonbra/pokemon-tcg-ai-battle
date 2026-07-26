# Arena BC Opponent Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the broken Alakazam 03 opponent, promote the validated Marnie BC package into the formal arena, and visibly identify the three BC-agent opponents across catalog-derived displays and reports.

**Architecture:** The catalog is the single identity source for formal opponents. Remove `alakazam_dudunsparce_03`, copy the Marnie policy/deck payload into a fresh self-contained `marnies_grimmsnarl_ex_froslass_03` package with the baseline `cg/`, and add `bc_agent` plus a display-name suffix for Alakazam 04, Marnie 03, and Team Rocket 01. Existing catalog consumers use `display_name`, so the marker flows to CLI, evaluation reports, and combat-matrix pages without renderer-specific forks.

**Tech Stack:** Python 3.11, standard-library unittest, official CG runtime, JSON catalog, arena evaluation CLI.

## Global Constraints

- Do not modify `engine/source/`.
- Delete only the explicit broken `alakazam_dudunsparce_03` opponent and its temporary candidate copy.
- A formal opponent remains an independent package with `main.py`, 60-card `deck.csv`, and physical baseline `cg/`.
- The catalog must only reference `evaluation/arena/opponents/` packages.
- Every fixed-pool change refreshes the long-lived combat matrix with ten official-runtime games per directed pairing.

---

### Task 1: Specify the curated catalog and package regression checks

**Files:**
- Modify: `tests/test_evaluation_assets.py`
- Modify: `evaluation/configs/opponents.json`

**Interfaces:**
- Consumes: catalog entries with `name`, `package`, `display_name`, `representative_card_ids`, `enabled`, and `tags`.
- Produces: exact enabled catalog set with 03 removed, Marnie 03 added, and `bc_agent` markers on Alakazam 04, Marnie 03, and Team Rocket 01.

- [ ] **Step 1: Write failing catalog assertions**

Assert `alakazam_dudunsparce_03` is absent, `marnies_grimmsnarl_ex_froslass_03` is enabled, and each of the three named BC packages has `bc_agent` in tags and `[BC Agent]` in `display_name`.

- [ ] **Step 2: Run the asset test and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_assets`

Expected: FAIL before the catalog/package update.

- [ ] **Step 3: Update catalog entries**

Remove the Alakazam 03 object. Add Marnie 03 with IDs `[648, 104]` and tags `marnies_grimmsnarl_ex`, `froslass`, `bc_agent`; append `[BC Agent]` to the specified three `display_name` values.

- [ ] **Step 4: Re-run the asset test**

Run: `python3 -m unittest -v tests.test_evaluation_assets`

Expected: PASS and the catalog resolves every enabled package.

### Task 2: Curate physical packages

**Files:**
- Delete: `evaluation/arena/opponents/alakazam_dudunsparce_03/`
- Delete: `evaluation/arena/candidates/alakazam_dudunsparce_03_candidate/`
- Create: `evaluation/arena/opponents/marnies_grimmsnarl_ex_froslass_03/`

**Interfaces:**
- Consumes: `agent_bc_marnie_1100` policy payload and a fixed-pool baseline `cg/`.
- Produces: self-contained formal Marnie 03 package whose CG tree hash matches each enabled opponent.

- [ ] **Step 1: Build the Marnie formal package**

Copy Marnie `main.py`, `idonly_policy.py`, `policy.pt`, and `deck.csv` into Marnie 03; copy `cg/` from `alakazam_dudunsparce_01`; do not carry the candidate's obsolete CG wrapper or temporary compatibility directories.

- [ ] **Step 2: Delete the explicit broken assets**

Delete the named 03 opponent and its temporary candidate copy only after the catalog no longer references the opponent.

- [ ] **Step 3: Validate packages**

Run: `python3 -m evaluation validate evaluation/arena/opponents/marnies_grimmsnarl_ex_froslass_03` and `python3 -m unittest -v tests.test_evaluation_assets`.

Expected: 60-card valid, baseline CG-compatible, and no stale catalog path.

### Task 3: Refresh fixed-pool evidence and verification

**Files:**
- Modify: `evaluation/arena/combat_mat/matrix.json`
- Modify: `evaluation/arena/combat_mat/index.html`
- Create: `evaluation/arena/combat_mat/reports/<package>/<run_id>/report.html`

**Interfaces:**
- Consumes: the updated enabled catalog.
- Produces: a 20-by-20 directed, ten-games-per-pair official-runtime combat matrix with BC labels supplied by catalog display names.

- [ ] **Step 1: Verify the new Marnie package against the fixed pool**

Run: `python3 -m evaluation run --candidate evaluation/arena/opponents/marnies_grimmsnarl_ex_froslass_03 --opponents all --games 10 --workers 8 --worker-cpu-threads 1 --output .tmp/evaluation/marnie_03_promotion`.

- [ ] **Step 2: Refresh the combat matrix**

Run the combat-matrix entry point for every enabled package with ten games against the updated catalog, retaining reports under `evaluation/arena/combat_mat/reports/` and regenerating `matrix.json` and `index.html`.

- [ ] **Step 3: Run complete verification**

Run: `python3 -m unittest discover -s tests -p 'test_*.py'` and `python3 -m unittest -v tests.test_evaluation_assets`.

Expected: PASS; confirm `alakazam_dudunsparce_03` is absent from catalog and combat-matrix data, while all three BC labels are visible in catalog-derived report identities.

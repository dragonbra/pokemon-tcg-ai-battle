# Four BC-Lineage Arena Opponents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anonymously archive the four new Arena packages with dated BC/RL identities and produce official-engine reports for each against the resulting 30-opponent catalog.

**Architecture:** The catalog remains the authoritative identity and classification source. Each incoming package becomes a self-contained package under `evaluation/arena/opponents/`, using the baseline physical `cg/` runtime; report output is retained under the package's long-lived combat-matrix report directory.

**Tech Stack:** Python 3.11, standard-library `unittest`, official engine runtime through `python3 -m evaluation`.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve anonymous archetype names and append `v20260729_bc`.
- Add `[BC]` display labels and `bc_agent` catalog tags to the three BC packages.
- Classify the `0017` PPO package as `[RL]` with the `rl_agent` catalog tag.
- Use 10 official-engine games per opponent and all 30 enabled opponents.

---

### Task 1: Normalize and archive the packages

**Files:**
- Rename: the four untracked package directories under `evaluation/arena/opponents/`
- Replace: non-baseline or missing package-local `cg/` directories

- [ ] Rename the packages to `dragapult_ex_03_v20260729_rl`, `mega_kangaskhan_ex_crustle_01_v20260729_bc`, `marnies_grimmsnarl_ex_froslass_06_v20260729_bc`, and `dragapult_ex_04_v20260729_bc`.
- [ ] Physically copy the baseline `cg/` tree into every incoming package whose runtime hash differs or is missing.
- [ ] Run `python3 -m evaluation validate` on all four final paths.

### Task 2: Register dated BC identities

**Files:**
- Modify: `evaluation/configs/opponents.json`
- Modify: `tests/test_evaluation_assets.py`
- Modify: `evaluation/combat_matrix.py`
- Modify: `tests/test_evaluation_combat_matrix.py`

- [ ] Add all four catalog entries in archetype/sequence order with representative card IDs from their exact decks.
- [ ] Extend the asset contract to require exactly 30 enabled names and visible BC classification.
- [ ] Make combat-matrix archetype grouping strip `_NN_vYYYYMMDD_bc` identities correctly.
- [ ] Run focused asset and combat-matrix tests.

### Task 3: Produce retained official-engine reports

**Files:**
- Create: `evaluation/arena/combat_mat/reports/<new-package>/<run-id>/report.html`

- [ ] Run each new package as candidate against `--opponents all --games 10` with 30 enabled opponents.
- [ ] Confirm every report contains 300 games, 30 opponent rows, and no unfinished worker sessions.
- [ ] Report exact win/loss/draw/error totals and clickable retained report paths.

### Task 4: Final repository verification

**Files:**
- Verify only; do not rewrite unrelated user changes.

- [ ] Run `python3 -m unittest -v tests.test_evaluation_assets tests.test_evaluation_combat_matrix`.
- [ ] Run `python3 -m compileall -q evaluation`.
- [ ] Inspect `git diff --check` and `git status --short`.

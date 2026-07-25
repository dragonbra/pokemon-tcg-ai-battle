# Repository Structure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quickly reorganize raw data and documentation, retire obsolete Kaggle/BC assets, and remove generated local state without changing stable evaluation, engine, or concurrent experiment-project architecture boundaries.

**Architecture:** Run independent inventory/mapping work in parallel, then apply filesystem migrations in the main workspace so ignored 11 GB data is moved only once. Keep `evaluation/` and `engine/` stable, flatten `docs/reports/` into topic directories, and retain only explicitly approved candidates and submissions.

**Tech Stack:** Git, filesystem operations, Python 3.11+, standard-library `unittest`, Markdown/HTML documentation.

## Global Constraints

- Never modify `engine/source/`.
- Do not touch `rl_environment/runs.py`, `tests/test_experiment_projects.py`, the 2026-07-26 experiment-project architecture spec/plan, its SDD ledger, new `experiments/`, or historical `rl_runs` projects `0001`–`0012`.
- Keep `evaluation/arena/opponents/`, catalog, and combat matrix unchanged.
- Do not implement the daily environment-report pipeline.
- Move the ignored replay dataset; do not copy its approximately 11 GB payload.
- Do not commit unless the user separately asks; project policy requires an immediate push after every commit.
- Prefer lightweight verification and finish promptly.

---

### Task 1: Build Exact Migration and Deletion Manifests

**Files:**
- Read: `docs/reports/**`
- Read: `evaluation/arena/candidates/**`
- Read: `submission/**`
- Read: `tests/test_*.py`
- Read: `.gitignore`

**Interfaces:**
- Consumes: approved design in `docs/superpowers/specs/2026-07-26-repository-structure-cleanup-design.md`.
- Produces: exact source→destination document mapping, obsolete-test list, candidate keep/delete list, and submission/dist mapping used by Tasks 2–5.

- [ ] **Step 1: Classify all `docs/reports/**` files into the approved direct `docs/` topics.**
- [ ] **Step 2: Identify tests and references whose only consumer is `train/kaggle_bc_top20`, `notebooks/kaggle_bc_worker`, deleted candidates, or deleted submissions.**
- [ ] **Step 3: Confirm candidates matching `0009`–`0012` are retained and `rank_*`/`meta*` candidates are deleted.**
- [ ] **Step 4: Match the three retained submission packages to existing dist archives by current names/content; do not guess missing mappings.**

### Task 2: Move Raw Episodes and Establish Data Contracts

**Files:**
- Move: `replays/raw/**` → `data/raw/episodes/**`
- Create: `data/raw/leaderboards/README.md`
- Create: `data/processed/environment_daily/README.md`
- Modify: `.gitignore`
- Modify: applicable references found in Task 1

**Interfaces:**
- Consumes: current replay layout and existing manifests/hashes.
- Produces: `data/raw/episodes/` as the sole raw-episode root and documented future input/output directories.

- [ ] **Step 1: Record source file count and total byte count for `replays/raw/`.**
- [ ] **Step 2: Create destination parents and move each source subtree without copying archives.**
- [ ] **Step 3: Update ignore rules from old replay paths to `data/raw/episodes/**`, preserving intentionally tracked manifest rules if present.**
- [ ] **Step 4: Add concise README contracts stating that leaderboards are immutable raw snapshots and environment_daily contains derived machine-readable output; no pipeline exists yet.**
- [ ] **Step 5: Compare destination file count and bytes with the source record, then remove the empty `replays/` tree.**

### Task 3: Flatten and Reindex Documentation

**Files:**
- Move: `docs/reports/**` → approved topic directories under `docs/`
- Create/Modify: `docs/README.md`
- Create/Modify: topic `README.md` indexes
- Modify: repository-local references to moved docs
- Do not modify: excluded concurrent architecture spec/plan

**Interfaces:**
- Consumes: exact mapping from Task 1.
- Produces: no `docs/reports/` directory, discoverable Top100/Top20 snapshots under `docs/environment-daily_kaggle_top100/daily/`, and repaired internal links.

- [ ] **Step 1: Create the direct topic roots: `environment`, `evaluation`, `training`, `decks`, `rules`, `research`, `implementation`, `references`, and `history`.**
- [ ] **Step 2: Move every `docs/reports/**` asset according to the manifest, grouping Top100/Top20 frozen assets by dated snapshot.**
- [ ] **Step 3: Add `docs/README.md` and concise topic indexes linking current, frozen, historical, and superseded material.**
- [ ] **Step 4: Repair Markdown, HTML, source comments/help, and non-excluded policy-document links to moved paths.**
- [ ] **Step 5: Remove the empty `docs/reports/` tree and search for remaining live `docs/reports/` references.**

### Task 4: Retire Obsolete Kaggle/BC Implementations and Tests

**Files:**
- Delete: `notebooks/kaggle_bc_worker/`
- Delete: `train/kaggle_bc_top20/`
- Delete/Modify: implementation-specific `tests/test_*.py`
- Modify: `.gitignore`, `pyproject.toml`, README/docs references where applicable
- Do not modify: `tests/test_experiment_projects.py`

**Interfaces:**
- Consumes: dependency/test classification from Task 1.
- Produces: no runnable generic Kaggle Top20 BC project and no tests/imports claiming it remains supported.

- [ ] **Step 1: Remove the notebook worker and generic training package.**
- [ ] **Step 2: Delete tests that exclusively protect removed implementations/assets; migrate path assertions in tests that still protect raw-data or visualization contracts.**
- [ ] **Step 3: Remove obsolete package includes, dynamic module names, commands, and ignore entries that have no surviving consumer.**
- [ ] **Step 4: Search for live `kaggle_bc_top20`, `kaggle_bc_worker`, and old replay-root references and repair only surviving historical/documentary links as historical references.**

### Task 5: Prune Candidates and Submissions

**Files:**
- Delete: `evaluation/arena/candidates/rank_*`
- Delete: `evaluation/arena/candidates/meta*`
- Preserve: candidates clearly mapped to `0009`–`0012`
- Rename: three approved `submission/` packages
- Prune/Rename: `archive/submission/dist/`

**Interfaces:**
- Consumes: candidate and archive mappings from Task 1.
- Produces: only approved historical candidates and exactly three project-numbered submission packages plus proven matching archives.

- [ ] **Step 1: Delete obsolete `rank_*` and `meta*` candidate directories without modifying opponents or catalog.**
- [ ] **Step 2: Rename the approved packages to `0010_alakazam_sota_model_v4_loss_best`, `0011_alakazam_sota_reward_weighted_bc_v3_loss_best`, and `0012_alakazam_sota_feature_engineering_v9_v1_exact_best`.**
- [ ] **Step 3: Retain and rename only proven matching dist archives with the same project prefixes; report any missing/ambiguous archive.**
- [ ] **Step 4: Delete all other submission packages and dist artifacts, then repair surviving references to the three renamed packages.**

### Task 6: Remove Local Generated State and Perform Lightweight Verification

**Files:**
- Delete: `.venv/`, `.tmp/**`, `pokemon_tcg_ai_battle_research.egg-info/`, `.pytest_cache/`, all `__pycache__/`, all `*.pyc`
- Preserve: `.claude/`, `.superpowers/`, ignore rules for regenerable state

**Interfaces:**
- Consumes: completed Tasks 2–5.
- Produces: clean local tree and a concise verification report.

- [ ] **Step 1: Check for active processes using `.venv` or `.tmp`; stop and report rather than deleting in-use state.**
- [ ] **Step 2: Delete the approved generated directories/files while preserving `.claude/` and `.superpowers/`.**
- [ ] **Step 3: Run focused surviving tests for moved data/docs paths, replay visualization, evaluation assets, and renamed submissions where such tests exist.**
- [ ] **Step 4: Run `python3 -m compileall -q evaluation visualization rl_environment train` and report failures without broad debugging.**
- [ ] **Step 5: Run final searches for removed live paths and `git status --short`; distinguish this task's changes from concurrent pre-existing changes.**

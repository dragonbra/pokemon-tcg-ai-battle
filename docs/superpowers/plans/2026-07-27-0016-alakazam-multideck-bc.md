# 0016 Alakazam Multideck BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one auditable multi-expert Alakazam BC policy from winning official Episodes and deploy it with the frozen Battle Cage + Wondrous Patch 60-card deck.

**Architecture:** Physically freeze the 0014 V12 `R2StrongScenarioPolicy` implementation inside project 0016, then add only a small declared source-persona residual required by the repository's multi-team label contract. Build a new replay-to-decision pipeline locally, preserve exact registered-deck conditioning per Episode, and export the frozen 0016 target deck rather than inheriting a source package deck.

**Tech Stack:** Python 3.11, PyTorch, standard-library `unittest`, official Kaggle replay JSON, repository `rl_environment` logging/version infrastructure, official engine evaluation.

## Global Constraints

- Project ID is exactly `0016_alakazam_multideck_bc`.
- The backbone is 0014 `V12_r2_strong_scenario_scalegate`; no 0014 executable import is permitted.
- Training starts from a fresh seeded initialization unless the user later explicitly authorizes a warm start.
- Multiple teams require explicit source conditioning and source-grouped evaluation.
- Every Episode binds exact team identity, exact 60-card deck, player index, reward, payload hash and split.
- Only winning Alakazam player trajectories enter the primary BC dataset.
- The target `deck.csv` is a canonical 60-line training asset and candidate export source.
- `engine/source/` is read-only; strength claims require official-engine games.
- Formal runs use new immutable `rl_runs/0016_alakazam_multideck_bc/versions/V<n>_<tag>/` directories and W&B online logging.
- Do not commit, push, upload a dataset, admit an opponent, or submit to Kaggle without explicit authorization.

---

### Task 1: Freeze project identity, target deck and provenance

**Files:**
- Create: `train/0016_alakazam_multideck_bc/__init__.py`
- Create: `train/0016_alakazam_multideck_bc/deck.csv`
- Create: `train/0016_alakazam_multideck_bc/deck_manifest.json`
- Create: `experiments/0016_alakazam_multideck_bc/manifest.json`
- Test: `train/0016_alakazam_multideck_bc/tests/test_project_contract.py`

**Interfaces:**
- Produces: `PROJECT_ID`, `TARGET_DECK_PATH`, and a canonical 60-card multiset consumed by data materialization, online inference and candidate export.

- [x] Write a test asserting project ID, 60 deck rows, exact counted manifest equality, required card counts `1146:1` and `1264:4`, and the declared V12 provenance.
- [x] Run `python3 -m unittest -v train.0016_alakazam_multideck_bc.tests.test_project_contract` and confirm it fails before the assets exist.
- [x] Add the identity and deck assets with the exact user-provided counts and SHA-256 metadata.
- [x] Re-run the focused test and require it to pass.

### Task 2: Freeze the V12 R2 implementation and source contract

**Files:**
- Create: the self-contained model/feature/runtime closure under `train/0016_alakazam_multideck_bc/`
- Create: `train/0016_alakazam_multideck_bc/source_model.py`
- Test: `train/0016_alakazam_multideck_bc/tests/test_model_contract.py`

**Interfaces:**
- Consumes: the frozen card ontology and V12 R2 input batch.
- Produces: `SourceConditionedR2Policy(config, source_config, ontology_path)` whose R2 backbone tensors and scenario ScaleGate equations remain unchanged.

- [x] Copy the required V12 source closure physically and remove every executable `0014` import/path.
- [x] Test the exact R2 defaults: width 320, 8 heads, 4 board layers, 2 scenario layers, ScaleGate initialization 1.0 and 17,387,842 backbone parameters.
- [x] Add a bounded source embedding residual to encoded state/options and reserve source ID 0 as the neutral deployment token.
- [ ] Test that different source IDs change logits, neutral source is deterministic, and removing source IDs fails closed.
- [ ] Run the focused model test on CPU with finite forward/backward gradients.

### Task 3: Build the replay selection and dataset audit

**Files:**
- Create: `train/0016_alakazam_multideck_bc/data/replay_catalog.py`
- Create: `train/0016_alakazam_multideck_bc/data/build_dataset.py`
- Create: `train/0016_alakazam_multideck_bc/data/source_vocabulary.py`
- Test: `train/0016_alakazam_multideck_bc/tests/test_replay_catalog.py`

**Interfaces:**
- Consumes: official archive zip members and targeted replay manifests.
- Produces: chronological episode-player decision rows with exact deck/source/outcome provenance and collision-safe Unicode source IDs.

- [ ] Test duplicate Episode rejection, exact team matching, win-only selection, 60-card validation and source-vocabulary stability.
- [ ] Implement structured replay parsing and actor-only projection without importing another numbered project.
- [ ] Group splits by whole Episode and prevent Episode, content-hash and source/session leakage.
- [ ] Emit counts by source, deck hash, card coverage, outcome, split and action length; fail if target cards 1146 or 1264 lack declared coverage.
- [ ] Run the catalog tests against small synthetic replay fixtures.

### Task 4: Materialize the faithful causal cache

**Files:**
- Modify: `train/0016_alakazam_multideck_bc/features/compiler.py`
- Modify: `train/0016_alakazam_multideck_bc/training/materialized.py`
- Test: `train/0016_alakazam_multideck_bc/tests/test_materialization.py`

**Interfaces:**
- Consumes: contiguous chronological decision rows from Task 3.
- Produces: `faithful_board_causal_cache_v1` shards plus `source_id`, with V12 tensors and masks unchanged.

- [ ] Test one episode with a changing causal ledger, registered deck alignment, source ID propagation and actions of length 1 and 16.
- [ ] Replace the retired 0014 builder guard with an actual local row compiler and sharded writer.
- [ ] Preserve V12's `a0_eligible <= 16` training gate while auditing longer accepted actions separately.
- [ ] Validate materialized/online parity and reject unknown source IDs or malformed masks.

### Task 5: Train, checkpoint and export

**Files:**
- Create: `train/0016_alakazam_multideck_bc/run.py`
- Create: `train/0016_alakazam_multideck_bc/export_candidate.py`
- Test: `train/0016_alakazam_multideck_bc/tests/test_export.py`

**Interfaces:**
- Consumes: audited cache, fixed source vocabulary and target deck.
- Produces: immutable formal run artifacts and a self-contained CPU candidate package.

- [ ] Preserve V12 optimizer defaults: AdamW, LR `3e-4`, weight decay `0.02`, batch 256, validation batch 512, BF16, seed `20260723`, patience 5 and min delta `0.001`.
- [ ] Log one online training pass and one complete teacher+greedy validation pass per epoch to JSONL, TensorBoard and W&B.
- [ ] Select checkpoints by the declared validation rule, not official-engine outcomes.
- [ ] Export the 0016 `deck.csv`, physical `cg/`, ontology, model and portable source closure; reject any source-package deck substitution.
- [ ] Validate the package and run an official-engine smoke report under `.tmp/evaluation/0016_smoke/`.

### Task 6: Keep the authoritative design synchronized

**Files:**
- Create: `experiments/0016_alakazam_multideck_bc/DESIGN.md`
- Create: `experiments/0016_alakazam_multideck_bc/DESIGN.html`

**Interfaces:**
- Consumes: current code shapes, dataset audit, model contract and target deck manifest.
- Produces: matching human-readable authority for inputs, architecture, loss, phases and package boundary.

- [x] Record official rules, runtime facts and project hypotheses as separate evidence layers.
- [x] Document every tensor family, source conditioning, V12 backbone provenance, BC objective, wins-only bias and target-deck boundary.
- [ ] Cross-check parameter counts, hashes, paths and current phase against code and manifests.
- [ ] Run `python3 -m compileall -q train/0016_alakazam_multideck_bc` and the complete focused unittest package.

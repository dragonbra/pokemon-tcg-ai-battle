# 0020 Pluggable Deck RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the selected 0019 epoch-13 universal checkpoint into a self-contained 0020 foundation, then measure its zero-training official-engine performance with five exact Arena decks before any deck-specific RL.

**Architecture:** Project 0020 owns a physically frozen copy of the 0019 model, feature compiler, online causal state, ontology, and model-only weights; it never imports executable code from another numbered project. A package builder emits five standard Arena candidates that differ only in exact `deck.csv` and immutable deck identity, while all use neutral `source_id=0` and the same checkpoint hash. Each candidate receives an isolated formal evaluation version and report.

**Tech Stack:** Python 3.11, PyTorch CPU inference, standard-library `unittest`, repository `evaluation` CLI, official engine runtime.

## Global Constraints

- Do not modify `engine/source/`.
- The foundation must use selected checkpoint `epoch-0013-da9b13d6f82d19d4.pt`, SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.
- Runtime inference must use exact 60-card deck conditioning and neutral deployment persona `source_id=0`.
- Project 0020 must not import executable code from `train/0019_universal_winner_bc` or any other numbered project.
- Zero-shot reports use official engine runtime, the fixed enabled opponent catalog snapshot, at least 10 games per opponent, `--workers 8 --worker-cpu-threads 1`, and no Team Rocket test deck.
- Offline legality and validation are supporting evidence only; gameplay strength claims come from formal Arena reports.
- No RL optimization, opponent admission, Kaggle submission, git commit, or git push is part of this phase.

---

### Task 1: Freeze the 0020 foundation contract

**Files:**
- Create: `train/0020_pluggable_deck_rl/`
- Create: `experiments/0020_pluggable_deck_rl/manifest.json`
- Create: `experiments/0020_pluggable_deck_rl/DESIGN.md`
- Create: `experiments/0020_pluggable_deck_rl/DESIGN.html`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13/artifact/`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13/checkpoint/`
- Test: `train/0020_pluggable_deck_rl/tests/test_foundation_contract.py`

**Interfaces:**
- Consumes: 0019 epoch-13 model-only checkpoint and its metadata.
- Produces: immutable `FoundationManifest` facts and a self-contained 0020 runtime source tree.

- [ ] **Step 1: Write contract tests**

Assert the project ID, checkpoint hash, epoch 13, model-only payload, neutral source ID, no cross-numbered imports, ontology hash, and synchronized Markdown/HTML design facts.

- [ ] **Step 2: Run tests and verify the new project is absent**

Run: `python3 -m unittest -v train.0020_pluggable_deck_rl.tests.test_foundation_contract`
Expected: FAIL because the 0020 modules and manifests do not exist.

- [ ] **Step 3: Copy and freeze the minimum 0019 runtime implementation**

Copy the model, feature, knowledge-state, and online inference implementation into 0020; replace all project identifiers and remove training-only imports from the runtime boundary.

- [ ] **Step 4: Copy checkpoint and ontology with immutable manifests**

Record source path, source project/version, epoch, SHA-256, byte size, payload schema, model config, feature switches, and `optimizer_state_saved=false`.

- [ ] **Step 5: Run foundation tests**

Run: `python3 -m unittest -v train.0020_pluggable_deck_rl.tests.test_foundation_contract`
Expected: PASS.

### Task 2: Implement exact-deck neutral inference and package generation

**Files:**
- Create: `train/0020_pluggable_deck_rl/inference.py`
- Create: `train/0020_pluggable_deck_rl/package_builder.py`
- Create: `train/0020_pluggable_deck_rl/tests/test_inference.py`
- Create: `train/0020_pluggable_deck_rl/tests/test_package_builder.py`
- Create: `evaluation/arena/candidates/0020_zero_shot_<deck>/` for five decks.

**Interfaces:**
- Consumes: `load_policy(checkpoint, ontology, deck, source_id=0)`.
- Produces: standard candidate packages whose `main.py` returns the package deck at registration and selects only official legal options thereafter.

- [ ] **Step 1: Write inference and builder tests**

Assert that two different decks produce different registered-card tensors while sharing the exact checkpoint hash, source ID remains zero, reset clears causal state, deck files contain exactly 60 integer lines, and every package physically contains `cg/`.

- [ ] **Step 2: Run tests and verify failure**

Run: `python3 -m unittest -v train.0020_pluggable_deck_rl.tests.test_inference train.0020_pluggable_deck_rl.tests.test_package_builder`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement fail-closed checkpoint loading**

Load the 0019 R15 config from checkpoint metadata, require the expected hash/schema/epoch/source ID, strictly load weights, and reject decks that are not exactly 60 official integer card IDs.

- [ ] **Step 4: Implement deterministic greedy runtime**

Build online causal features from the package deck, inject `source_id=torch.tensor([0])`, use inference mode, and retain the established count-valid fallback only for declared capacity/runtime failures with an auditable counter.

- [ ] **Step 5: Generate the five candidates**

Use decks from `alakazam_dudunsparce_04_sota`, `dragapult_ex_03_v20260729_rl`, `marnies_grimmsnarl_ex_froslass_05_bc`, `mega_lucario_ex_solrock_07_bc`, and `mega_kangaskhan_ex_crustle_01_v20260729_bc`.

- [ ] **Step 6: Validate packages**

Run `python3 -m evaluation validate <candidate>` for all five packages.
Expected: five successful validations.

### Task 3: Freeze the zero-shot evaluation matrix

**Files:**
- Create: `experiments/0020_pluggable_deck_rl/evaluation/V2_zero_shot_alakazam.html`
- Create: `experiments/0020_pluggable_deck_rl/evaluation/V3_zero_shot_dragapult.html`
- Create: `experiments/0020_pluggable_deck_rl/evaluation/V4_zero_shot_marnie.html`
- Create: `experiments/0020_pluggable_deck_rl/evaluation/V5_zero_shot_lucario.html`
- Create: `experiments/0020_pluggable_deck_rl/evaluation/V6_zero_shot_kangaskhan.html`
- Create: matching `rl_runs/.../versions/<version>/artifact/evaluation.json` records.
- Refresh: `experiments/0020_pluggable_deck_rl/evaluation/index.html`.

**Interfaces:**
- Consumes: five validated candidate packages and a recorded opponent catalog hash/snapshot.
- Produces: five comparable official-engine baselines with win/loss/draw/error, completion, first/second-seat, and metric payloads.

- [ ] **Step 1: Record evaluation protocol**

Freeze candidate/deck/checkpoint hashes, catalog hash and enabled names, games per opponent, worker settings, seed behavior, metric profile, and official runtime identity before execution.

- [ ] **Step 2: Run a one-game temporary smoke for each candidate**

Write reports under `.tmp/evaluation/0020_zero_shot_smoke/`; confirm non-error completion before formal runs.

- [ ] **Step 3: Run five formal evaluations**

For each candidate run all enabled Arena opponents with 10 games per opponent, 8 workers, one CPU thread per worker, and write its immutable versioned report.

- [ ] **Step 4: Verify and summarize results**

Require the expected game count, zero worker errors, full completion, both seat orders, identical checkpoint hash across candidates, distinct exact deck hashes, and refreshed evaluation index.

### Task 4: Close the 0019 generalization gate

**Files:**
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.md`
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.html`
- Create: `experiments/0020_pluggable_deck_rl/GENERALIZATION_GATE.md`

**Interfaces:**
- Consumes: the five formal Arena reports.
- Produces: an evidence-bounded decision on whether the shared checkpoint is a useful zero-shot base and which deck should enter the first isolated RL version.

- [ ] **Step 1: Compare without inventing an after-the-fact threshold**

Report per-deck and aggregate strength, completion/error rate, seat sensitivity, fallback incidence, and matchup spread. Distinguish “plays legally”, “finishes games”, and “wins competitively”.

- [ ] **Step 2: Synchronize the authoritative design documents**

Update current stage, frozen schema, zero-shot results, limitations, and the explicit next-stage RL boundary in both Markdown and HTML.

- [ ] **Step 3: Run project and repository checks**

Run: `python3 -m unittest -v train.0020_pluggable_deck_rl.tests.test_foundation_contract train.0020_pluggable_deck_rl.tests.test_inference train.0020_pluggable_deck_rl.tests.test_package_builder tests.test_evaluation_assets`

Run: `python3 -m compileall -q train/0020_pluggable_deck_rl evaluation`

Expected: all tests pass and compilation succeeds.

# Raging Bolt Ogerpon User Zero-Shot Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained zero-training candidate for the user-supplied exact 60-card Raging Bolt ex / Teal Mask Ogerpon ex deck and evaluate it against the enabled Frozen-51 catalog with the official engine.

**Architecture:** Reuse the validated 0020 universal frozen foundation candidate package, replacing only the exact deck and provenance manifest. Register a deck-specific diagnostic profile, validate package invariants, then publish an immutable V19 official-engine report and its reverse-link status record.

**Tech Stack:** Python 3.11, PyTorch CUDA inference, official engine runtime, repository `evaluation` CLI, unittest.

## Global Constraints

- Keep the package candidate-only under `evaluation/arena/candidates/`; do not modify `evaluation/arena/opponents/` or the frozen catalog.
- Use foundation checkpoint SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`, `source_id=0`, and `training_updates=0`.
- `deck.csv` must contain exactly 60 canonical simulator card IDs and the package must contain no symlink, cache, or external runtime dependency.
- Formal evaluation must use 51 enabled opponents, 10 games per opponent, eight workers, one CPU thread per worker, balanced seats, CUDA inference, and the official engine.
- Publish to a new immutable version; never overwrite an existing report or version directory.

---

### Task 1: Candidate Package

**Files:**
- Create: `evaluation/arena/candidates/raging_bolt_ogerpon_user_zero_shot/`
- Create: `evaluation/arena/candidates/raging_bolt_ogerpon_user_zero_shot/deck.csv`
- Create: `evaluation/arena/candidates/raging_bolt_ogerpon_user_zero_shot/manifest.json`

**Interfaces:**
- Consumes: validated `evaluation/arena/candidates/0024_lucario_hariyama_zero_shot/` package layout and frozen foundation model.
- Produces: a self-contained candidate accepted by `python3 -m evaluation validate`.

- [ ] **Step 1: Resolve the exact deck**

Use canonical IDs `63x4, 96x4, 226x1, 1227x4, 1198x4, 1121x4, 1122x4, 1124x4, 1094x4, 1127x3, 1118x2, 1080x1, 1x13, 4x4, 6x4`; assert the sum is 60.

- [ ] **Step 2: Copy the self-contained foundation package**

Copy the validated package physically, replace `deck.csv`, and update `manifest.json` with user provenance, exact hashes, `source_id=0`, and `training_updates=0`.

- [ ] **Step 3: Validate package invariants**

Run:

```bash
python3 -m evaluation validate evaluation/arena/candidates/raging_bolt_ogerpon_user_zero_shot
```

Expected: valid exact 60-card package, canonical `cg/`, no symlink or missing runtime asset.

- [ ] **Step 4: Smoke the policy on CUDA**

Load the candidate policy server on `cuda:0` and assert initialization observation `{"select": null}` returns the exact deck.

### Task 2: Metric Profile Coverage

**Files:**
- Modify: `evaluation/metrics/league_profiles.py`
- Modify: `tests/test_evaluation_league_profiles.py`

**Interfaces:**
- Consumes: candidate identity `raging_bolt_ogerpon_user_zero_shot`.
- Produces: profile ID `raging_bolt_ogerpon_energy` using the generic auditable metric vocabulary.

- [ ] **Step 1: Add a failing profile assertion**

Assert the new candidate resolves to a profile focused on first attack, energy setup, attack continuity, and Prize conversion.

- [ ] **Step 2: Run the profile test**

```bash
python3 -m unittest -v tests.test_evaluation_league_profiles
```

Expected before implementation: failure because the candidate identity is absent.

- [ ] **Step 3: Register the minimal profile**

Add the candidate identity to a Raging Bolt / Ogerpon energy-engine profile with card IDs `63`, `96`, and `226` and an explicit warning that energy acceleration must be validated through attacks and wins.

- [ ] **Step 4: Run metric and reporting tests**

```bash
python3 -m unittest -v tests.test_evaluation_league_profiles tests.test_evaluation_league_quality tests.test_evaluation_reporting
```

Expected: all tests pass.

### Task 3: Formal Frozen-51 Evaluation

**Files:**
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V19_raging_bolt_ogerpon_user_zero_shot_frozen51/artifact/status.json`
- Create: `experiments/0020_pluggable_deck_rl/evaluation/V19_raging_bolt_ogerpon_user_zero_shot_frozen51.html`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V19_raging_bolt_ogerpon_user_zero_shot_frozen51/artifact/evaluation.json`
- Modify: `experiments/0020_pluggable_deck_rl/evaluation/index.html`

**Interfaces:**
- Consumes: validated candidate and `league_deck_quality` metric profile.
- Produces: immutable official-engine result with exact outcome, seat split, per-opponent results, process metrics, report hash, and reverse link.

- [ ] **Step 1: Allocate the unused V19 version**

Create empty `artifact/`, `tensorboard/`, `checkpoint/`, and `wandb/` directories and write an allocated evaluation-only status with W&B disabled because no training occurs.

- [ ] **Step 2: Run the official evaluation**

```bash
python3 -m evaluation run \
  --candidate evaluation/arena/candidates/raging_bolt_ogerpon_user_zero_shot \
  --opponents all --games 10 --workers 8 --worker-cpu-threads 1 \
  --candidate-device cuda:0 --opponent-device cuda:0 \
  --metric-profile league_deck_quality \
  --output experiments/0020_pluggable_deck_rl/evaluation/V19_raging_bolt_ogerpon_user_zero_shot_frozen51.html
```

Expected: 510/510 complete, 255 games in each seat, zero engine errors, zero unfinished games, and zero league-quality metric errors.

- [ ] **Step 3: Finalize the version record**

Record the run ID, exact outcome, seat split, process metrics, wall time, report SHA-256, and candidate-only promotion state in `status.json`; verify `evaluation.json` matches.

- [ ] **Step 4: Perform final audit**

Run `git diff --check`, all metric/report tests, exact deck count/hash checks, symlink/cache checks, and confirm the project index links V19.

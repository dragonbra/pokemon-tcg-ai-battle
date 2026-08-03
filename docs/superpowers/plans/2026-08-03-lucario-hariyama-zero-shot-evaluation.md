# Lucario Hariyama Zero-Shot Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained zero-shot candidate for the supplied Mega Lucario ex / Hariyama 60-card deck and evaluate it against the current 51-deck Frozen Arena.

**Architecture:** Resolve every `(name, set, number)` entry against `data/official/EN_Card_Data.csv`, then materialize a new candidate by physically copying the validated 0020 neutral Foundation package and replacing only its deck and provenance manifest. Allocate a new immutable 0020 evaluation-only version and run the unchanged official engine with the `league_deck_quality` profile.

**Tech Stack:** Python 3.11, repository evaluation CLI, PyTorch CUDA shared inference, JSON manifests, HTML report-only evaluation artifacts.

## Global Constraints

- Do not modify `engine/source/`.
- Keep `evaluation/arena/candidates/0024_lucario_hariyama_zero_shot/` self-contained with `main.py`, exactly 60 `deck.csv` rows, physical `cg/`, model weights, ontology, and strategy code.
- Do not add the candidate to `evaluation/configs/frozen.json` or `evaluation/arena/frozen/` without later explicit user approval.
- Use the neutral frozen 0019 Epoch 13 Foundation checkpoint with `source_id=0`; no source/team identity may alter logits.
- Write the formal report to `experiments/0020_pluggable_deck_rl/evaluation/V17_lucario_hariyama_zero_shot_frozen51.html` and its backlink to the matching version artifact directory.
- Run 51 opponents x 10 games with balanced seats, 8 workers, one CPU thread per worker, shared `cuda:0` inference, and `league_deck_quality` metrics.

---

### Task 1: Resolve And Audit The Exact Deck

**Files:**
- Read: `data/official/EN_Card_Data.csv`
- Create: `evaluation/arena/candidates/0024_lucario_hariyama_zero_shot/deck.csv`

**Interfaces:**
- Consumes: the user-supplied `(count, card name, expansion, collection number)` list.
- Produces: a 60-row integer card-ID list whose grouped counts round-trip to all 30 supplied lines.

- [ ] **Step 1: Resolve all entries with the CSV parser and fail on missing or ambiguous identities**

Run a read-only `csv.DictReader` audit keyed by normalized card name, expansion, and collection number. Expected: one unique internal card ID per supplied entry.

- [ ] **Step 2: Verify category totals and total deck size**

Expected: 15 Pokemon, 34 Trainer, 11 Energy, 60 total.

- [ ] **Step 3: Search current frozen and candidate assets for the resulting exact deck hash**

Expected: report any exact prior identity as provenance, but do not reuse a different policy package silently.

### Task 2: Materialize And Validate The Candidate

**Files:**
- Copy: `evaluation/arena/candidates/0020_zero_shot_lucario/`
- Create: `evaluation/arena/candidates/0024_lucario_hariyama_zero_shot/`
- Modify: `evaluation/arena/candidates/0024_lucario_hariyama_zero_shot/deck.csv`
- Modify: `evaluation/arena/candidates/0024_lucario_hariyama_zero_shot/manifest.json`

**Interfaces:**
- Consumes: the audited 60-row deck from Task 1 and the frozen 0019 Foundation runtime.
- Produces: a standard Submission Package loadable by `python3 -m evaluation validate` and shared GPU inference.

- [ ] **Step 1: Physically copy the validated neutral zero-shot package**

Copy all runtime files, removing only generated `__pycache__` and `.pyc` files.

- [ ] **Step 2: Replace deck.csv and write provenance manifest**

Manifest fields must identify `0024_lucario_hariyama_zero_shot`, `V1_frozen_0019_epoch13`, checkpoint SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`, `source_id=0`, and `training_updates=0`.

- [ ] **Step 3: Validate package and initialization contract**

Run `python3 -m evaluation validate evaluation/arena/candidates/0024_lucario_hariyama_zero_shot`. Expected: 60-card valid, canonical frozen `cg` hash, and initialization observation returns the exact deck.

- [ ] **Step 4: Run shared GPU inference smoke**

Instantiate `PolicyServer` on `cuda:0`; expected: Foundation policy loads and returns all 60 deck IDs.

### Task 3: Run And Archive Frozen-51 Evaluation

**Files:**
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V17_lucario_hariyama_zero_shot_frozen51/artifact/status.json`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V17_lucario_hariyama_zero_shot_frozen51/artifact/evaluation.json`
- Create: `experiments/0020_pluggable_deck_rl/evaluation/V17_lucario_hariyama_zero_shot_frozen51.html`
- Modify: `experiments/0020_pluggable_deck_rl/evaluation/index.html`

**Interfaces:**
- Consumes: the validated candidate package from Task 2.
- Produces: an immutable official-engine report, report backlink, index row, and completed status record.

- [ ] **Step 1: Allocate an unused formal version**

Create the version directories and an `allocated` evaluation-only `status.json`; reject existing files or report paths.

- [ ] **Step 2: Run the standard Frozen Arena contract**

Run `python3 -m evaluation run --candidate evaluation/arena/candidates/0024_lucario_hariyama_zero_shot --opponents all --games 10 --workers 8 --worker-cpu-threads 1 --candidate-device cuda:0 --opponent-device cuda:0 --metric-profile league_deck_quality --output experiments/0020_pluggable_deck_rl/evaluation/V17_lucario_hariyama_zero_shot_frozen51.html`.

- [ ] **Step 3: Verify report integrity and complete status**

Expected: 510 total games, 255 first and 255 second, zero errors, zero unfinished, report SHA matching `artifact/evaluation.json`, and an updated project evaluation index.

- [ ] **Step 4: Report outcome without promoting the candidate**

Summarize overall and seat win rates, strongest and weakest matchup groups, process metrics, exact deck identity, and clickable candidate/report paths. Keep the package only under `arena/candidates/`.

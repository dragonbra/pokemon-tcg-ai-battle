# Policy-0806 Seeded-512 Full Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run all 55 Frozen-0806 exact decks from frequency number 001 onward under the official seeded-512 contract, publish auditable HTML reports, and rename deck asset directories to stable 001–055 presentation names without losing immutable deck identities.

**Architecture:** Keep schedule deck IDs and exact-deck hashes as immutable audit keys, while treating the directory basename as presentation-only `NNN_slug`. Preserve the incomplete 2026-08-07 256-game report tree unchanged and publish the new evaluation into a distinct `seeded_512_v1` tree. Run official CPU engine games with resident CUDA policy inference; benchmark topology first, then execute candidates in numeric order and incrementally refresh the index.

**Tech Stack:** Python 3.11, official engine runtime, PyTorch resident inference, repository evaluation runner, unittest, HTML/embedded JSON reports.

## Global Constraints

- Never modify `engine/source/`.
- Frozen-0806 strength evidence requires 512/512 finished, zero error, zero unfinished, evaluation seed `341512806`, and two copies of the committed 256-game distribution.
- Preserve the incomplete 2026-08-07 25/55 legacy 256-game report tree; never overwrite or merge it into the seeded-512 reports.
- Deck numbering is frequency-descending with `best_rank` and immutable deck ID as tie breaks.
- Internal `deck_id`, exact-deck SHA-256, source ranks, and schedule SHA-256 remain unchanged.
- Formal reports live below `evaluation/arena/combat_mat/policy_0806/`; temporary benchmark artifacts live below `.tmp/evaluation/`.
- Loop-guard outcomes are project-level adjudications and must remain distinguishable from native official-engine terminal reasons.

---

### Task 1: Numbered Frozen Pool Directories

**Files:**
- Modify: `evaluation/frozen_0806.py`
- Modify: `evaluation/frozen_0806_assets.py`
- Modify: `evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks/*`
- Test: `tests/test_evaluation_frozen_0806.py`

**Interfaces:**
- Consumes: committed `schedule.json` entries and each deck's `manifest.json.deck_id`.
- Produces: loader support for `NNN_slug` directory names while returning the original immutable `Frozen0806Deck.deck_id`.

- [ ] **Step 1: Write failing tests for exactly 001–055 directory prefixes and manifest-derived deck IDs.**
- [ ] **Step 2: Run `python3 -m unittest -v tests.test_evaluation_frozen_0806` and confirm the new assertions fail.**
- [ ] **Step 3: Change the loader to validate directory presentation names independently from immutable manifest deck IDs.**
- [ ] **Step 4: Rename all 55 directories from hash-suffixed IDs to their canonical `NNN_slug` values, without changing deck.csv or immutable manifest identity fields.**
- [ ] **Step 5: Run Frozen pool/runtime asset tests and verify 55 exact 60-card decks, schedule hash, and policy hashes remain unchanged.**

### Task 2: Independent Seeded-512 Policy-0806 Publisher

**Files:**
- Modify: `evaluation/frozen_0806_full_evaluation.py`
- Modify: `evaluation/arena/combat_mat/index.html`
- Create: `evaluation/arena/combat_mat/policy_0806/index.html`
- Create during evaluation: `evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_seeded_512_v1/`
- Test: `tests/test_evaluation_frozen_0806_full_evaluation.py`

**Interfaces:**
- Consumes: `evaluation_counts(schedule)`, `evaluation_schedule_id(base_sha)`, and `FROZEN_0806_EVALUATION_SEED`.
- Produces: one 512-game report per numbered deck and an incremental 55-row summary index.

- [ ] **Step 1: Add tests asserting 512 games, 256/256 seat balance, seeded-512 schedule ID, and a new output root distinct from the legacy tree.**
- [ ] **Step 2: Run the focused tests and confirm failure under the current 256-game implementation.**
- [ ] **Step 3: Wire the full evaluator to the canonical seeded-512 constants and expanded schedule counts.**
- [ ] **Step 4: Update HTML copy/manifest schema so every report states fixed seed, 512 games, completion/error contract, and 001–055 identity.**
- [ ] **Step 5: Add policy-level navigation linking legacy 256 evidence and the new seeded-512 tree without moving or rewriting legacy reports.**
- [ ] **Step 6: Run focused and evaluation regression tests.**

### Task 3: Throughput Calibration and Ordered Full Run

**Files:**
- Write temporary evidence: `.tmp/evaluation/frozen_0806_seeded512_benchmark/`
- Write formal reports: `evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_seeded_512_v1/`

**Interfaces:**
- Consumes: numbered candidates, shared Policy-0806 inference server, official seeded runtime.
- Produces: complete per-deck HTML reports and incrementally refreshed aggregate manifest/index.

- [ ] **Step 1: Benchmark small even game subsets for viable worker/engine-pool topologies with CPU threads fixed to one.**
- [ ] **Step 2: Select the fastest zero-error topology that respects local CPU/GPU memory limits and record it in the run manifest.**
- [ ] **Step 3: Start the formal run in numeric deck order beginning at 001; do not reuse any legacy report.**
- [ ] **Step 4: After every completed deck, validate 512/512 finished, 0 error, 0 unfinished and atomically refresh the index.**
- [ ] **Step 5: Continue through deck 055 when measured throughput makes the complete run practical; otherwise preserve the resumable prefix and report the measured ETA.**
- [ ] **Step 6: Validate the final aggregate contains 55 reports and 28,160 official-engine games before calling it a complete zero-shot strength evaluation.**

### Task 4: Final Audit

**Files:**
- Validate: all files above.

**Interfaces:**
- Consumes: final pool and evaluation assets.
- Produces: an evidence-backed handoff with clickable report paths and exact completion statistics.

- [ ] **Step 1: Run `git diff --check` and relevant unittest suites.**
- [ ] **Step 2: Verify deck directories are exactly `001_*` through `055_*`, each has exactly 60 valid cards, and manifest deck IDs still match the committed schedule.**
- [ ] **Step 3: Verify the legacy tree still has its original 25 reports and incomplete manifest.**
- [ ] **Step 4: Verify the new index/report embedded JSON records seed, schedule, runtime, per-opponent results, errors, unfinished games, and timing.**

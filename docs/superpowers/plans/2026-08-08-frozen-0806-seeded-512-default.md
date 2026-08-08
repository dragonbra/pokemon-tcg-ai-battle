# Frozen-0806 Seeded 512 Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make future Frozen-0806 checkpoint evaluation default to two deterministic 256-game units (512 games total) on the seeded official engine while preserving the existing incomplete Policy-0806 256-game evaluation assets unchanged.

**Architecture:** Keep the immutable Frozen-0806 pool distribution as a 256-game sampling unit. Add a small evaluation-contract module that expands it to two units, commits the evaluation seed and derives a contract-specific schedule identity; route the standard Frozen CLI through this contract. The historical 25/55 Policy-0806 run remains at its existing paths and is documented as legacy incomplete evidence, not silently resumed or deleted.

**Tech Stack:** Python 3.11, `unittest`, existing evaluation `BatchConfig`, seeded official engine runtime.

## Global Constraints

- Do not modify `engine/source/`.
- Do not delete, overwrite, or relabel the existing Policy-0806 256-game reports or `.tmp/evaluation/frozen_0806_full` diagnostics.
- The Frozen-0806 pool remains an immutable 256-game unit; the evaluation contract runs exactly two units.
- The evaluation seed must be distinct from RL training seeds and recorded in every report manifest.
- Every 512-game run must contain 256 first-seat and 256 second-seat games, 256 paired engine/Search seeds, zero errors, and zero unfinished games before it is accepted as strength evidence.

---

### Task 1: Define the shared Frozen-0806 evaluation contract

**Files:**
- Create: `evaluation/frozen_0806_contract.py`
- Test: `tests/test_evaluation_frozen_0806_contract.py`

**Interfaces:**
- Produces: `FROZEN_0806_UNIT_GAMES`, `FROZEN_0806_EVALUATION_UNITS`, `FROZEN_0806_EVALUATION_GAMES`, `FROZEN_0806_EVALUATION_SEED`, `evaluation_counts(schedule)`, and `evaluation_schedule_id(base_schedule_sha256)`.

- [ ] Write tests asserting 256 × 2 = 512, doubled per-opponent counts, a stable contract schedule ID, and rejection of a malformed non-256 schedule.
- [ ] Run `python3 -m unittest -v tests.test_evaluation_frozen_0806_contract` and confirm the test fails because the module does not exist.
- [ ] Implement the constants and pure helper functions with strict validation.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Make the standard Frozen CLI use the new contract

**Files:**
- Modify: `evaluation/cli.py`
- Modify: `tests/test_evaluation_cli.py`

**Interfaces:**
- Consumes: the contract constants and helpers from Task 1.
- Produces: a `BatchConfig` with 512 total games, the committed evaluation seed, doubled fixed distribution, seeded runtime enabled, and the contract schedule ID whenever `--pool frozen` is selected.

- [ ] Add a mocked Frozen catalog CLI test that captures the delegated `BatchConfig`.
- [ ] Assert 512 games, a 256/256 seat-compatible even count, fixed seed, seeded engine, doubled counts, and contract schedule identity.
- [ ] Run the focused CLI test and confirm it fails under the old 256-game behavior.
- [ ] Change only the Frozen branch of `_run`; leave legacy `--pool opponents --games N` behavior unchanged.
- [ ] Re-run the CLI tests and confirm they pass.

### Task 3: Align the checkpoint-series evaluator and documentation

**Files:**
- Modify: `train/0034_dragapult_third_large_model_rl/evaluate_seeded_checkpoint_series.py`
- Modify: `evaluation/README.md`

**Interfaces:**
- Consumes: the contract constants and helpers from Task 1.
- Produces: one canonical seed/count definition shared by the completed 0034 comparison and future Frozen CLI runs.

- [ ] Replace local 512 count and evaluation-seed literals in the checkpoint-series evaluator with shared contract imports.
- [ ] Document the Frozen-0806 512-game default, two-unit distribution, seed separation, paired-seat requirement, and report fields.
- [ ] Record that `evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1/` remains a preserved 25/55 legacy 256-game partial run and must not be combined with new 512-game reports.
- [ ] Run the checkpoint-series schedule test and confirm it still commits 512 games, 256 seed pairs, and 256/256 seats.

### Task 4: Verify compatibility and preservation

**Files:**
- Verify only; no additional files.

**Interfaces:**
- Consumes: all preceding changes.
- Produces: evidence that the new default is correct without mutating historical assets.

- [ ] Run the focused contract, CLI, batch scheduling, checkpoint-series, and seeded-runtime tests.
- [ ] Recount the preserved Policy-0806 reports and confirm 25 reports remain present.
- [ ] Confirm `.tmp/evaluation/frozen_0806_full` still exists.
- [ ] Run `git diff --quiet -- engine/source` and confirm the official source is unchanged.

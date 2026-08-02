# 0025 Frozen Evaluation Implementation Plan

> **For agentic workers:** Execute this plan inline; the official-engine evaluation must remain attached to the current session until completion.

**Goal:** Export the globally lowest-loss V3 checkpoint as a self-contained Raging Bolt candidate and evaluate it for 10 games against every enabled Frozen opponent.

**Architecture:** Add a 0025-owned portable legacy runtime that reproduces the training codec and greedy decoder, including the Frozen Arena batched GPU inference interface. Package the exact James Cox deck, model-only checkpoint, runtime source, and physical `cg/`, then validate, smoke, and run the formal 51-by-10 official-engine evaluation resolved from the current catalog.

**Tech Stack:** Python 3.11, PyTorch, official engine runtime, repository Evaluation CLI, Frozen Arena GPU inference service.

## Global Constraints

- Do not modify `engine/source/`.
- Do not import executable code from another numbered training project.
- Use `legacy_default/best_validation_loss.pt`, epoch 6, validation loss `0.3075830024137147`.
- Keep the candidate in `evaluation/arena/candidates/`; do not promote it to the formal opponent pool.
- Formal output is `experiments/0025_semantic_foundation_pretraining/evaluation/V3_james_cox_raging_bolt_restart.html` with an immutable artifact backlink.
- Offline imitation and official-engine evaluation are distinct evidence layers.

---

### Task 1: Portable Runtime And Export

- [x] Add 0025-owned online codec, batched greedy decoder, portable policy, and legal fallback.
- [x] Add a fail-closed exporter for exact deck, model-only checkpoint, physical `cg/`, source files, and manifest.
- [x] Test batched decode parity, package self-containment, and checkpoint rejection rules.

### Task 2: Candidate Validation And Smoke

- [x] Export `evaluation/arena/candidates/0025_v3_legacy_default_best_loss`.
- [x] Run repository package validation.
- [x] Run a 20-game Frozen official-engine smoke with zero candidate/opponent errors and zero unfinished games.

### Task 3: Formal Frozen Evaluation

- [x] Run all 51 enabled Frozen opponents for 10 games each with 8 workers and one CPU thread per worker.
- [x] Write the immutable V3 report and project evaluation index.
- [x] Create the V3 artifact backlink and verify report hash/run identity.

### Task 4: Authority Sync

- [x] Update `DESIGN.md` and `DESIGN.html` with deployment inputs, evaluation contract, and measured outcome.
- [x] State the evidence boundary and next hypothesis based on official-engine results.

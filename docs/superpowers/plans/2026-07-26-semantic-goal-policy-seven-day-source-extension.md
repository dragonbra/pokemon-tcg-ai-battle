# 0013 Eight-Day Source Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add only 2026-07-18 and 2026-07-19 to the approved 2026-07-18..25 Yushin Ito winner source, then publish one deterministic mixed eight-day train/validation dataset.

**Architecture:** The existing daily archive reader validates each immutable ZIP in a separate process and merges daily spools in declared chronological order. The tracked source manifest, frozen protocol source-date boundary, source audit, source-manifest hash binding, split commitments, and DESIGN documents are updated together before a no-clobber dataset rebuild.

**Tech Stack:** Python 3.11, standard-library `unittest`, `ProcessPoolExecutor`, deterministic gzip JSONL shards, SHA-256 commitments.

## Global Constraints

- Include exactly 2026-07-18 through 2026-07-25; do not include 2026-07-10..17.
- Keep expert normalization and winner semantics unchanged: complete unique Yushin Ito winner only.
- Mix all seven days before the deterministic episode-player group split.
- Preserve the declared 2026-07-24 patch precedence.
- Do not modify `engine/source/`, opponent catalog, candidate pool, or Kaggle assets.
- Any source/protocol/hash mismatch stops publication; no overwrite of a completed dataset.
- No commit or push.

---

### Task 1: Extend and freeze the source boundary

**Files:**
- Modify: `experiments/0013_semantic_goal_policy/source_manifest.json`
- Modify: `train/0013_semantic_goal_policy/configs/pre_run_protocol.json`
- Modify: `train/0013_semantic_goal_policy/protocol.py`
- Test: `train/0013_semantic_goal_policy/tests/test_protocol.py`
- Test: `train/0013_semantic_goal_policy/tests/test_source_reader.py`

**Interfaces:**
- Consumes: archive SHA-256 and member counts for 2026-07-18/19.
- Produces: one canonical chronological eight-source declaration and new frozen protocol SHA-256.

- [ ] Add tests asserting exact dates `2026-07-18..2026-07-25`, hashes, counts, and rejection of any eighth date.
- [ ] Run focused tests and observe failure against the eight-day contract.
- [ ] Add the two archive declarations and explicit source date range to the protocol.
- [ ] Recompute the canonical protocol and source-manifest SHA-256; update code constants only from these bytes.
- [ ] Run protocol/source tests and `git diff --check`.

### Task 2: Audit and publish the eight-day dataset

**Files:**
- Modify: `experiments/0013_semantic_goal_policy/data_audit/source_audit.json`
- Generate: `rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1/`
- Generate private: `rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1.private_split.json`

**Interfaces:**
- Consumes: `build_dataset_with_derived_split(..., source_workers=8)`.
- Produces: deterministic train/validation shards, dataset reference, distribution audit, private split commitment.

- [ ] Scan 2026-07-18 and 2026-07-19 with two workers and record exact added winner groups and incomplete counts.
- [ ] Run the eight-source builder with eight daily workers; retain chronological merge order.
- [ ] Validate every shard/hash/count via `iter_dataset()` and compare aggregate source counts with the audit.
- [ ] Persist only aggregate/hash evidence in tracked audit; keep detailed split assignments private under `rl_runs`.
- [ ] Stop if action schema, visibility, conflict, split, or hash gates fail.

### Task 3: Synchronize design and resume downstream gates

**Files:**
- Modify: `experiments/0013_semantic_goal_policy/DESIGN.md`
- Modify: `experiments/0013_semantic_goal_policy/DESIGN.html`
- Modify: `experiments/0013_semantic_goal_policy/README.md`

**Interfaces:**
- Consumes: final dataset reference and exact eight-day counts.
- Produces: auditable current-stage documentation for record-backed training.

- [ ] Replace eight-day source facts with exact eight-day facts and link the final dataset commitment.
- [ ] Run all project tests, compileall, `git diff --check`, and protected-engine diff.
- [ ] Resume record-backed trainer/model integration only after dataset publication validates.

## Self-review

- Spec coverage: exact two added dates, mixed split, patch preservation, parallel scan, immutable publication, and synchronized docs are covered.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Type consistency: `source_workers` remains an integer and `build_dataset_with_derived_split` continues returning `(dataset_reference, SplitManifest)`.

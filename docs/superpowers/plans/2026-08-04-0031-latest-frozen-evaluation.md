# 0031 Latest Frozen Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the 0031 V4 epoch-2 `latest.pt` checkpoint with Dragapult, Marnie, Mega Lopunny, and Raging Bolt exact League decks against all 51 immutable Frozen Arena opponents.

**Architecture:** Add a deployment-only chronological observation adapter around the existing 0031 compiler and causal knowledge state, then export self-contained temporary Arena packages that strict-load the original model-only checkpoint. A dedicated batch runner binds the four exact League deck hashes, runs 510 official-engine games per deck, validates complete zero-error reports, and publishes an auditable aggregate below `evaluation/arena/combat_mat/0031_latest_frozen_test/`.

**Tech Stack:** Python 3.11, PyTorch, official engine runtime, existing `evaluation` CLI, HTML/JSON embedded reports.

## Global Constraints

- Never modify `engine/source/`.
- Use `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V4_lr5e4_no_early_stop_b512/checkpoint/rule_faithful_semantic/latest.pt` with schema `0031_model_only_checkpoint_v1`, epoch 2, global step 33002, and strict state loading.
- Preserve each exact 60-card deck and its League catalog SHA-256.
- Use Frozen pool `0019_foundation_51_exact_decks_v4`, 51 opponents, 10 games per opponent, alternating first/second, eight workers, and one CPU thread per worker.
- Candidate packages must contain physical files, no symlinks, and no runtime imports from the repository.
- Store temporary runs under `.tmp/evaluation/0031_latest_frozen_test/`; store durable source reports and the aggregate index under `evaluation/arena/combat_mat/0031_latest_frozen_test/`.
- Refuse partial reports, identity drift, errors, and overwrite of a different report.
- Do not modify the 0031 model, feature, action, loss, or training contract; deployment-only additions do not require DESIGN changes.
- Preserve unrelated user changes. Commits are omitted because the shared worktree is already dirty and the user did not request commits.

---

### Task 1: Online Inference And Self-Contained Export

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/deployment/__init__.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/deployment/online_runtime.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/deployment/inference.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/export_candidate.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_deployment.py`

**Interfaces:**
- Consumes: 0031 model-only checkpoint, exact deck, public/full-engine prototypes, and official `cg/`.
- Produces: `PortableSemanticPolicy.from_checkpoint(...)`, `OnlineCausalEncoder.encode(...)`, and `export_candidate(...) -> dict[str, object]`.

- [ ] Write a test that creates a small 0031 checkpoint, encodes a chronological projected observation, and asserts greedy output obeys `minCount/maxCount`.
- [ ] Run `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_deployment` and verify missing deployment modules fail.
- [ ] Implement `OnlineCausalEncoder` using `CausalKnowledge.consume`, `compile_canonical_row`, and `collate_canonical_records` with a structural dummy target used only to satisfy collation.
- [ ] Implement strict schema/model/arm loading and `SemanticPolicy.greedy_action`; reject illegal decode output rather than silently substituting a policy action.
- [ ] Implement atomic self-contained export with `main.py`, `deck.csv`, physical `cg/`, model, contracts, domain, features, knowledge, prototypes, manifest, and checkpoint provenance.
- [ ] Run the focused deployment test and existing 0031 model/feature tests.

### Task 2: Four Package Smokes

**Files:**
- Generate ignored packages: `.tmp/model_loading/0031_latest/candidates/<deck_id>/`
- Generate ignored deck files: `.tmp/model_loading/0031_latest/decks/<deck_id>.csv`

**Interfaces:**
- Consumes: the four exact records from the immutable 0022 V11 League catalog.
- Produces: four validated Arena candidates using one checkpoint hash.

- [ ] Materialize `dragapult_ex_001`, `marnies_grimmsnarl_ex_froslass_001`, `mega_lopunny_ex_mega_froslass_ex_001`, and `raging_bolt_ex_james_cox_henry_chao_001` and verify their deck hashes.
- [ ] Run `python3 -m evaluation validate <package>` for every package.
- [ ] Run one official-engine game for the first package and verify it completes with model inference and no worker error.

### Task 3: Frozen Batch Reports

**Files:**
- Create: `evaluation/rule_faithful_frozen_batch.py`
- Create: `tests/test_rule_faithful_frozen_batch.py`
- Generate: `evaluation/arena/combat_mat/0031_latest_frozen_test/manifest.json`
- Generate: `evaluation/arena/combat_mat/0031_latest_frozen_test/index.html`
- Generate: `evaluation/arena/combat_mat/0031_latest_frozen_test/reports/<deck_id>.html`

**Interfaces:**
- Consumes: the four validated candidates and completed evaluation reports.
- Produces: four immutable 510-game source reports and an aggregate manifest/index.

- [ ] Add tests for exact checkpoint/deck/pool identity, 510/510 completion, zero errors, deterministic deck order, and overwrite refusal.
- [ ] Implement the batch CLI using `--pool frozen --opponents all --games 10 --workers 8 --worker-cpu-threads 1` and shared CUDA inference when memory permits.
- [ ] Run the four decks in requested order, refreshing the aggregate after each completed report.
- [ ] Audit 2,040 total games, all four deck hashes, one checkpoint hash, zero errors/unfinished games, and valid report links.
- [ ] Run focused tests plus `python3 -m unittest -v tests.test_evaluation_assets` and report the official-engine results without treating offline BC metrics as strength evidence.

## Self-Review

- Spec coverage: the four named decks, the latest 0031 node, immutable Frozen pool, official engine, exact deck identity, per-deck reports, and aggregate output are each assigned to a task.
- Placeholder scan: no deferred implementation or unspecified error handling remains.
- Type consistency: deployment consumes `Mapping[str, Any]`, returns collated tensors and `list[int]`; export returns a JSON-compatible manifest; batch validation consumes embedded HTML report data and returns one aggregate record per deck.

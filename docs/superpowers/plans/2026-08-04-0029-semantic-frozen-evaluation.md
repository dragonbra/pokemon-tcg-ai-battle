# 0029 Semantic Frozen Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the 0029 multi-memory semantic checkpoint in a self-contained Arena package and evaluate every 0022 V11 League deck against the immutable 51-deck Frozen pool, publishing one durable HTML report per deck.

**Architecture:** Preserve the checkpoint's actual 0025 `SemanticFoundationPolicy` tensor contract and add deployment-only greedy decoding plus a causal online observation adapter. Export temporary self-contained packages from the immutable checkpoint and exact League catalog decks, run the unmodified official engine through the existing Frozen evaluation CLI, then publish complete reports and an aggregate index below `evaluation/arena/combat_mat/0029_frozen_test/`.

**Tech Stack:** Python 3.11, PyTorch model-only checkpoints, existing evaluation runner, official engine runtime, HTML/JSON audit assets.

## Global Constraints

- Never modify `engine/source/`.
- Use checkpoint `best_greedy_exact.pt`, schema `0025_model_only_checkpoint_v1`, model `SemanticFoundationPolicy`, and strict state loading.
- Preserve the exact 0022 V11 League catalog deck multiset and SHA-256 for every candidate.
- Use Frozen pool `0019_foundation_51_exact_decks_v4`, all 51 opponents, 10 games per opponent, alternating first/second, eight workers, one CPU thread per worker, and shared CUDA inference.
- Treat offline exact-action metrics only as imitation evidence; report strength only from official-engine games.
- Candidate packages must contain no symlinks or repository runtime imports.
- Do not overwrite an existing per-deck report; fail closed on identity mismatch or partial runs.
- Keep unrelated user changes untouched. Commits are intentionally omitted because the shared worktree is already dirty with unrelated 0031 work.

---

### Task 1: Semantic Deployment Contract

**Files:**
- Create: `train/0025_semantic_foundation_pretraining/deployment/semantic_online_runtime.py`
- Create: `train/0025_semantic_foundation_pretraining/deployment/semantic_inference.py`
- Modify: `train/0025_semantic_foundation_pretraining/model/multi_memory.py`
- Test: `train/0025_semantic_foundation_pretraining/tests/test_semantic_deployment.py`

**Interfaces:**
- Consumes: model-only payload containing `metadata.arm == "semantic"` and nested `model_config.config`.
- Produces: `PortableSemanticPolicy.from_checkpoint(path, deck)` and `OnlineCausalEncoder.encode(observation) -> dict[str, Tensor]` compatible with `evaluation.runner.inference_server.PolicyServer`.

- [ ] Add a failing test that strict-loads a small semantic checkpoint, compiles a chronological online observation, and returns a legal ordered option sequence.
- [ ] Run `python3 -m unittest -v train.0025_semantic_foundation_pretraining.tests.test_semantic_deployment` and confirm the missing deployment modules fail.
- [ ] Implement deployment-only batched greedy decoding with explicit STOP and `minCount/maxCount` bounds.
- [ ] Implement the online adapter using the training-time `CausalKnowledge`, `compile_row`, prototype index, and `collate` functions.
- [ ] Make inference errors fail closed in the shared server and rerun the focused test.

### Task 2: Self-Contained Semantic Export

**Files:**
- Modify: `train/0025_semantic_foundation_pretraining/export_candidate.py`
- Modify: `train/0025_semantic_foundation_pretraining/tests/test_semantic_deployment.py`

**Interfaces:**
- Consumes: checkpoint path, exact 60-line deck, official `cg/`, output path, and Frozen deck ID.
- Produces: a standard package with `main.py`, `deck.csv`, physical `cg/`, semantic runtime source, prototypes, `model.bin`, and provenance manifest.

- [ ] Add a failing export test asserting semantic arm acceptance, no symlinks, 60 cards, correct project ID from checkpoint metadata, and isolated package import.
- [ ] Extend `_main_source` and runtime copying for `arm == "semantic"` without changing existing legacy/canonical output.
- [ ] Copy only the semantic compiler/model/knowledge/runtime files and both prototype sidecars.
- [ ] Run semantic deployment tests and the existing 0025 deployment/canonical deployment suites.

### Task 3: Priority Package Smoke

**Files:**
- Generate ignored assets: `.tmp/model_loading/0029_large/candidates/<deck_id>/`
- Generate ignored decks: `.tmp/model_loading/0029_large/decks/<deck_id>.csv`

**Interfaces:**
- Consumes: V11 `league_catalog.json`, extracted `best_greedy_exact.pt`, and `evaluation/arena/frozen/_policy/cg/`.
- Produces: four validated packages for `dragapult_ex_001`, `mega_lucario_ex_solrock_001`, `mega_lopunny_ex_mega_froslass_ex_001`, and `raging_bolt_ex_james_cox_henry_chao_001`.

- [ ] Materialize exact deck CSV files from the structured catalog and verify the catalog SHA for each.
- [ ] Export all four packages and run `python3 -m evaluation validate` on each.
- [ ] Run a one-game direct official-engine smoke outside the research report path and confirm model inference is used with no worker/inference errors.

### Task 4: Durable Frozen Report Publisher

**Files:**
- Create: `evaluation/semantic_frozen_batch.py`
- Create: `tests/test_semantic_frozen_batch.py`
- Generate: `evaluation/arena/combat_mat/0029_frozen_test/index.html`
- Generate: `evaluation/arena/combat_mat/0029_frozen_test/manifest.json`
- Generate: `evaluation/arena/combat_mat/0029_frozen_test/reports/<deck_id>.html`

**Interfaces:**
- Consumes: completed `.tmp/evaluation/0029_frozen_test/<deck_id>/run-*/report.html` and embedded report payload.
- Produces: immutable per-deck HTML, a machine-readable manifest, and an index ordered with the four requested priority decks first.

- [ ] Add tests for report completion validation, pool/checkpoint/deck identity, refusal to overwrite conflicting reports, and deterministic index ordering.
- [ ] Implement a CLI that exports one deck, invokes the existing evaluation CLI, validates 510 completed games and zero errors, publishes the report atomically, and refreshes manifest/index after every deck.
- [ ] Include deck name/hash, checkpoint hash, Frozen pool ID, run ID, W-L-D, first/second results, completion, errors, duration, and report link in the index.
- [ ] Run the publisher tests and a priority-deck report publication smoke.

### Task 5: Full League Evaluation

**Files:**
- Generate: all remaining per-deck HTML files and finalized manifest/index under `evaluation/arena/combat_mat/0029_frozen_test/`.

**Interfaces:**
- Consumes: all 48 exact decks from the immutable V11 catalog.
- Produces: 48 reports, each containing 51 opponents x 10 official-engine games, plus a complete aggregate entry point.

- [ ] Run the four priority decks in requested order with `--workers 8 --worker-cpu-threads 1 --candidate-device cuda:0 --opponent-device cuda:0`.
- [ ] After each run, verify `510/510` completed, zero errors, immutable pool/checkpoint/catalog identities, and openable HTML.
- [ ] Run the remaining 44 decks in catalog order, resuming only by skipping already validated published reports.
- [ ] Validate the final manifest has 48 unique deck IDs, 48 reports, 24,480 games, exact deck hashes, and no failed or partial runs.
- [ ] Run focused unit tests, evaluation asset validation, HTML link checks, and report payload audits; record any residual stochastic-comparison limitation in the index.

## Self-Review

- Spec coverage: all League decks, four requested priorities, per-deck Frozen reports, durable combat-matrix subdirectory, exact checkpoint/deck/pool identities, and official-engine evidence are assigned to tasks.
- Placeholder scan: no deferred implementation steps or unspecified tests remain.
- Type consistency: online encoding returns one-observation tensor batches; the existing inference server combines them; publisher inputs and output identities are explicitly defined.

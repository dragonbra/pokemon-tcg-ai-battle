# 0038 Release-Blocking Semantic Parity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove or falsify semantic parity between the 0038 CUDA RL/Frozen path and the Kaggle-compatible official CPU submission path, locating the first divergence before any new RL training or submission.

**Architecture:** Add an isolated, read-only audit package under `train/0038_action_boundary_rl/semantic_parity/` that inventories immutable runtime identities, canonicalizes public state/legal/action data, compares fixed rule scripts and macro unfolding, and replays fixed snapshots through CUDA, training-reference, and packaged inference. Store large traces under ignored `.tmp/evaluation/0038_semantic_parity_audit/`; keep the final concise evidence ledger in root `SEMANTIC_PARITY_AUDIT.md`.

**Tech Stack:** Python 3.11, PyTorch, existing official seeded CPU runtime, existing resident CUDA engine/bindings, unittest/property tests, JSON/JSONL manifests, SHA-256.

## Global Constraints

- Do not modify `engine/source/`, official ABI, observation schema, checkpoint tensors, or existing checkpoint files.
- Do not start RL training, upload/submit to Kaggle, add model structures, or tune reward/learning rate.
- Run deterministic greedy inference with dropout disabled and explicit dtype/device metadata.
- Stop comparison of a game after its first semantic divergence; later differences are consequences, not root-cause evidence.
- Missing critical model keys, schema mismatch, silent fallback, timeout, or lifecycle reset fail closed.
- Gate order is A (rules) → B (macro) → C (snapshot inference) → D (paired games).
- New diagnostic artifacts live under `.tmp/evaluation/0038_semantic_parity_audit/` and remain untracked.

---

### Task 1: Runtime Path and Immutable Identity Inventory

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/__init__.py`
- Create: `train/0038_action_boundary_rl/semantic_parity/inventory.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_inventory.py`
- Create: `.tmp/evaluation/0038_semantic_parity_audit/runtime_inventory.json`

**Interfaces:**
- Consumes: U230 checkpoint, extracted submission, CUDA rules pack/extension, official seeded CPU library.
- Produces: `build_runtime_inventory(checkpoint, package_root) -> dict[str, Any]` and a fail-closed hash ledger used by every later Gate.

- [ ] Write tests requiring all entrypoints, schema versions, hashes, precision flags, and model component inventories.
- [ ] Verify the tests fail before the inventory module exists.
- [ ] Implement path discovery and SHA-256 hashing without importing or mutating engine state.
- [ ] Audit all `load_state_dict`, fallback, `try/except`, `.eval()`, dtype, LoRA merge, allocation/meta loading, deck routing, and pending lifecycle code paths.
- [ ] Generate `runtime_inventory.json` and run the inventory tests.

### Task 2: Canonical Comparison Records

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/canonical.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_canonical.py`

**Interfaces:**
- Consumes: official observations, CUDA semantic tensors/state snapshots, primitive selections.
- Produces: `canonical_public_observation`, `canonical_legal_set`, `canonical_macro_action`, `stable_hash`, and `FirstDivergence` records.

- [ ] Write tests for stable serial identity, option-order-independent legal signatures, mask preservation, hidden-field exclusion, and deterministic hashing.
- [ ] Verify malformed observations and missing required fields fail closed.
- [ ] Implement the smallest canonicalizers required by Gates A–D.
- [ ] Run focused canonicalization tests.

### Task 3: Gate A — Model-Free Rule Differential

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/gate_a_rules.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_gate_a.py`
- Reuse read-only: `engine_cuda/tools/run_0022_deck40_official_parity.py`
- Create: `.tmp/evaluation/0038_semantic_parity_audit/gate_a/summary.json`

**Interfaces:**
- Consumes: fixed deck pairs, fixed engine seeds, deterministic primitive policies/scripts.
- Produces: per-step official/CUDA canonical state, legal set, terminal, winner, public-event and reward-hook comparison with first-divergence classification.

- [ ] Add tests for script serialization, deterministic selection validation, and first-divergence truncation.
- [ ] Run existing official CPU/POD/CUDA setup and battle parity tools on the 007 focal deck and representative opponents.
- [ ] Cover attack turn-end, evolution, attach/retreat, Ability/Item/Supporter/Stadium order, gust, KO/prize, Munkidori, Phantom Dive, and Area Zero Bench expansion/contraction where reachable fixtures exist.
- [ ] Record unsupported effects explicitly as coverage gaps rather than PASS.
- [ ] Emit Gate A PASS/FAIL/INCOMPLETE with the first differing field and state hashes.

### Task 4: Gate B — Phantom Dive Macro Compiler and Transaction Parity

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/gate_b_macro.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_gate_b.py`
- Create: `.tmp/evaluation/0038_semantic_parity_audit/gate_b/summary.json`

**Interfaces:**
- Consumes: canonical allocations, official callback fixtures, CUDA adapter semantic fixtures.
- Produces: primitive sequence, target-universe hash, per-callback stable-serial relocation records, transaction lifecycle and final public-state comparisons.

- [ ] Write failing tests for `{A:3,B:2,C:1}`, order aliases, target identity drift, empty masks, terminal-in-transaction, and actor/context drift.
- [ ] Add explicit `n=1..8` coverage; `n>5` must remain one strategic macro even if the internal allocation algorithm changes.
- [ ] Prove one pending transaction, one strategic forward/value/transition, six primitive selects, and no internal discount for normal cases.
- [ ] Compare CUDA execution with official primitive unfolding on fixed fixtures; fail closed on state/legal/KO/prize/terminal mismatch.
- [ ] Verify joint old-logprob and PPO replay use identical root-plus-allocation definition.

### Task 5: Gate C — Fixed Snapshot Preprocessing and Model Parity

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/gate_c_snapshot.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_gate_c.py`
- Create: `.tmp/evaluation/0038_semantic_parity_audit/gate_c/snapshots.jsonl`
- Create: `.tmp/evaluation/0038_semantic_parity_audit/gate_c/summary.json`

**Interfaces:**
- Consumes: representative official traces and CUDA decision tensors for U230.
- Produces: per-layer hashes/errors for observation → legal set → mask → gate → features → logits → macro → primitive expansion → Value.

- [ ] Select deterministic snapshots spanning both seats, root decisions, forced callbacks, Phantom callbacks, KO/prize, and extended Bench states.
- [ ] Compare official/package and training-reference preprocessing exactly before running neural comparison.
- [ ] Compare CUDA semantic tensors to official/package tensors field by field and stop at the first mismatch.
- [ ] For matching tensors, report state/option representation error, root/allocation logit max/mean absolute error, top-k agreement, margin, greedy top-1, macro and Value parity.
- [ ] Run CPU FP32, CUDA FP32 with TF32 disabled, and packaged FP16-storage→FP32-runtime paths under deterministic greedy mode.

### Task 6: Gate D — End-to-End Paired Replay/Game Localization

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/gate_d_paired.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_gate_d.py`
- Create: `.tmp/evaluation/0038_semantic_parity_audit/gate_d/summary.json`

**Interfaces:**
- Consumes: the same U230 checkpoint/package, opponent snapshot, seats and canonical seed schedule.
- Produces: one first-divergence record per game plus timeout/fallback/macro reset/forward counters.

- [ ] Write tests proving comparison stops after the first divergence and records both sides' hashes/top-k/Value/action intent.
- [ ] Run a small deterministic paired smoke only after Gates A–C have usable evidence.
- [ ] Separate RNG/setup divergence from observation/legal/gate/logit/macro/primitive/lifecycle divergence.
- [ ] Report outcomes only as downstream context; never use aggregate win rate as the divergence detector.

### Task 7: Fail-Closed Submission Manifest

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/submission_manifest.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_submission_manifest.py`
- Create: `.tmp/evaluation/0038_semantic_parity_audit/u230_submission_manifest.json`

**Interfaces:**
- Consumes: inventory and component state dicts.
- Produces: versioned manifest covering base, LoRA, decoder, Value, allocation, schemas, database/rules, precision, greedy config and git identity.

- [ ] Add tests that mutate/remove every critical field and require startup rejection.
- [ ] Compute separate component hashes rather than only a monolithic model-file hash.
- [ ] Verify critical missing/unexpected keys are exactly zero.
- [ ] Verify no broad exception path can silently switch to legacy/random/fallback inference.

### Task 8: Release Decision and Evidence Report

**Files:**
- Create: `SEMANTIC_PARITY_AUDIT.md`
- Update only if semantics actually change: `experiments/0038_action_boundary_rl/DESIGN.md`
- Update only if semantics actually change: `experiments/0038_action_boundary_rl/DESIGN.html`

**Interfaces:**
- Consumes: runtime inventory and Gate A–D summaries.
- Produces: release-blocking PASS/FAIL matrix, first divergence evidence, exact commands, modified-file ledger, residual risk and submission-ready verdict.

- [ ] Cross-check every requested acceptance criterion against a concrete artifact.
- [ ] Mark uncovered or unavailable official fixtures as INCOMPLETE, never PASS.
- [ ] Explain any minimal bug fix with before/after evidence and regression test.
- [ ] End with an explicit `submission_ready: yes/no` and the blockers required to change it.
- [ ] Run all new tests plus relevant existing 0038/CUDA parity tests; record exact results and hashes.

## Self-Review

- Spec coverage: Tasks 1–8 map directly to path inventory, Gates A–D, packaging audit, manifest and release verdict.
- Placeholder scan: no deferred implementation placeholders are used; unsupported runtime coverage is represented as evidence-backed INCOMPLETE.
- Type consistency: all Gates consume immutable inventory/canonical records and write independent JSON evidence under the same audit root.
- Execution mode: inline in this session because the user requested immediate audit execution; no subagents are used.

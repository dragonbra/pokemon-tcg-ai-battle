# 0038 Kaggle Parity and Update 230 Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove and enforce semantic parity between the 0038 Frozen evaluator and the self-contained Kaggle runtime, preserve the audited feature-compiler acceleration, and export update 230 as a submission-ready archive.

**Architecture:** Treat the accelerated Frozen policy as the reference implementation and replay identical official observations through both runtimes. Export the production runtime from the same 0038 source modules instead of maintaining an unchecked parallel inference path, then gate packaging on logits/action/history parity and an extracted-package official-engine smoke.

**Tech Stack:** Python 3.11, PyTorch, official seeded engine runtime, unittest, tar/gzip.

## Global Constraints

- Do not modify `engine/source/`, the official ABI, observation schema, or primitive `select` return shape.
- Preserve DecisionGate, forced observe-only, Phantom Dive macro, opponent-meta conditioning, and stable target identity.
- Preserve exact actor semantics; performance optimization may only change caching/compilation mechanics.
- The final archive must be `archive/submission/dist/0038_dragapult_ex_rl_update230.tar.gz` with `main.py`, `deck.csv`, `cg/`, and `strategy/` at extraction root.
- Do not upload or submit to Kaggle without a separate explicit authorization.

---

### Task 1: Establish the failing semantic and compiler parity evidence

**Files:**
- Modify: `train/0038_action_boundary_rl/tests/test_export_full_semantic_candidate.py`
- Inspect: `train/0038_action_boundary_rl/training/run_full_semantic.py`
- Inspect: `train/0038_action_boundary_rl/rollout/cuda_collector.py`
- Inspect: `train/0038_action_boundary_rl/kaggle_runtime/compound_inference.py`

**Interfaces:**
- Consumes: fixed official observations and update-230 `0038_model_only_checkpoint_v1`.
- Produces: a failing test that compares reference root logits, allocation logits, primitive selects, history state, and compiler implementation identity.

- [ ] **Step 1: Write the failing parity test**

```python
def test_exported_runtime_matches_reference_on_fixed_trace(self):
    reference = load_reference_policy(UPDATE_230)
    exported = load_exported_policy(UPDATE_230)
    for observation in fixed_official_trace():
        expected = reference.debug_decision(observation)
        actual = exported.debug_decision(observation)
        torch.testing.assert_close(actual.root_logits, expected.root_logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual.allocation_logits, expected.allocation_logits, atol=1e-5, rtol=1e-5)
        self.assertEqual(actual.select, expected.select)
        self.assertEqual(actual.history_digest, expected.history_digest)
```

- [ ] **Step 2: Run the test to verify the current export fails**

Run: `python3 -m unittest -v train.0038_action_boundary_rl.tests.test_export_full_semantic_candidate`

Expected: FAIL at the first concrete semantic or compiler-runtime mismatch.

- [ ] **Step 3: Record the first divergence and trace it to its source**

Run: `git diff --no-index <reference-runtime-file> <exported-runtime-file>`

Expected: an exact implementation/configuration difference that explains the failed assertion.

### Task 2: Make the Kaggle runtime share the audited production semantics

**Files:**
- Modify: `train/0038_action_boundary_rl/export_full_semantic_candidate.py`
- Modify: `train/0038_action_boundary_rl/kaggle_runtime/compound_inference.py`
- Modify: `train/0038_action_boundary_rl/kaggle_runtime/main.py`
- Modify: `train/0038_action_boundary_rl/tests/test_export_full_semantic_candidate.py`

**Interfaces:**
- Consumes: the diagnosed divergence from Task 1.
- Produces: `export_candidate(...)` whose runtime uses the same state compiler, history update, root/meta conditioning, canonical allocation scorer, and primitive macro executor as Frozen evaluation.

- [ ] **Step 1: Add a regression assertion for every diagnosed divergence**

```python
self.assertEqual(exported.action_index, reference.action_index)
self.assertEqual(exported.pending_macro, reference.pending_macro)
self.assertEqual(exported.compiler_contract, reference.compiler_contract)
```

- [ ] **Step 2: Run the focused tests and retain the failure**

Run: `python3 -m unittest -v train.0038_action_boundary_rl.tests.test_export_full_semantic_candidate`

Expected: FAIL before the implementation change.

- [ ] **Step 3: Replace the divergent export path with the reference contract**

```python
portable_metadata["inference_contract"] = "0038_frozen_compound_greedy_v1"
portable_metadata["feature_compiler_contract"] = reference_feature_compiler_contract
```

- [ ] **Step 4: Run semantic and disabled-path parity tests**

Run: `python3 -m unittest -v train.0038_action_boundary_rl.tests.test_export_full_semantic_candidate train.0038_action_boundary_rl.tests.test_observe_only_replay_parity train.0038_action_boundary_rl.tests.test_worker_feature_parity`

Expected: PASS with exact primitive actions and numerically bounded logits.

### Task 3: Export and validate update 230

**Files:**
- Create: `archive/submission/0038_dragapult_ex_rl_update230/`
- Create: `archive/submission/dist/0038_dragapult_ex_rl_update230.tar.gz`

**Interfaces:**
- Consumes: `checkpoint/update-000230.pt`, its SHA sidecar, and `core-update-000230.json`.
- Produces: a self-contained Kaggle archive and manifest with checkpoint, runtime, compiler, and Frozen provenance hashes.

- [ ] **Step 1: Verify checkpoint and Frozen evaluation contracts**

Run: `sha256sum rl_runs/0038_action_boundary_rl/versions/V10_complete_512_rollout_fresh_rl/checkpoint/update-000230.pt`

Expected: digest equals `update-000230.pt.sha256`, 2048 valid Frozen games, zero errors.

- [ ] **Step 2: Export the package**

Run: `python3 -m train.0038_action_boundary_rl.export_full_semantic_candidate --checkpoint rl_runs/0038_action_boundary_rl/versions/V10_complete_512_rollout_fresh_rl/checkpoint/update-000230.pt --output archive/submission/0038_dragapult_ex_rl_update230`

Expected: strict state inventory, Frozen provenance, and compiler contract checks pass.

- [ ] **Step 3: Run extracted-package and official-engine tests**

Run: `python3 -m evaluation validate archive/submission/0038_dragapult_ex_rl_update230`

Expected: exact 60-card deck and matching official `cg` hash.

- [ ] **Step 4: Build and inspect the archive**

Run: `tar -czf archive/submission/dist/0038_dragapult_ex_rl_update230.tar.gz -C archive/submission/0038_dragapult_ex_rl_update230 .`

Expected: no top-level wrapper, symlink, bytecode cache, optimizer, rollout buffer, replay, or absolute local path.

- [ ] **Step 5: Report the artifact without submitting it**

Run: `sha256sum archive/submission/dist/0038_dragapult_ex_rl_update230.tar.gz`

Expected: report archive path, size, SHA-256, Frozen result, parity evidence, compiler contract, and official-engine smoke result.


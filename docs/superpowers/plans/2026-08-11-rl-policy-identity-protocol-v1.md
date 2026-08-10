# RL Policy Identity Protocol V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a repository-level immutable policy identity protocol and a hard runtime gate that prevents focal-policy weights from contaminating frozen opponents in training, evaluation, or CUDA routing.

**Architecture:** Keep policy metadata and reconstruction in the self-contained `0040` project, using existing immutable 0806/0809 archives as registered bases. A single resolver returns a fully materialized model plus a content-addressed effective identity audit; CPU rollout, CUDA rollout, frozen evaluation, and parity tools consume that resolver. CUDA groups work only after identity resolution and uses a complete `Semantic0031DeviceAdapter` for each distinct opponent policy.

**Tech Stack:** Python 3.11, PyTorch, existing semantic0031 model/archive loaders, unittest/pytest-compatible tests, JSON manifests, Markdown.

## Global Constraints

- Do not modify `engine/source/`.
- Do not change PPO, rewards, model structure, or training hyperparameters.
- Preserve all historical outputs; label Hybrid evidence invalid as Frozen-0806 rather than deleting it.
- Policy identity resolution precedes routing or batching; mismatches raise a fatal exception.
- Frozen-0806 and Frozen-0809 CUDA-2048 remain separate benchmarks and promotion is manual.
- Do not run CUDA-2048 until the four-seed lane-1 CPU/CUDA Full-0806 parity gate passes.
- Preserve unrelated dirty-worktree changes in CUDA kernels, parity tools, and docs.

---

### Task 1: Canonical Protocol And Agent Contract

**Files:**
- Modify: `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: the user-specified V1 invariants and existing Frozen-0806 v3 evaluation contract.
- Produces: the only canonical protocol and a root-level mandatory read/link for future RL work.

- [ ] Verify the existing untracked protocol covers all 12 canonical rules, manifest fields, Promote Champion workflow, result validity, and historical Hybrid labeling.
- [ ] Add missing normative language and explicit CPU/CUDA/training/evaluation identity semantics.
- [ ] Add an `RL Policy Identity / Promote Champion - NON-NEGOTIABLE` section to root `AGENTS.md` containing the short invariant list and canonical link.
- [ ] Search for a second canonical copy and ensure none exists.

### Task 2: Policy Registry, Resolver, And Hard Gate

**Files:**
- Create: `train/0040_dragapult_0809_action_boundary_rl/policy_identity.py`
- Create: `train/0040_dragapult_0809_action_boundary_rl/policy_registry.json`
- Test: `train/0040_dragapult_0809_action_boundary_rl/tests/test_policy_identity.py`

**Interfaces:**
- Produces: `resolve_policy(policy_id, device) -> ResolvedPolicy`, `audit_materialized_policy(...) -> PolicyIdentityAudit`, `PolicyIdentityViolation`, and stable component/effective hashes.
- Consumes: immutable 0806/0809 archive manifests, checkpoints, schemas, and archive loaders.

- [ ] Write failing tests for Full-0806, Full-0809, focal-switch stability, deliberate Hybrid rejection, promoted composition round-trip, and training/evaluation resolver equality.
- [ ] Register Policy-0806 and Policy-0809 by immutable archive/checkpoint identity without tensor duplication.
- [ ] Hash every effective model component (`prototype_encoder`, `state_encoder`, option input/layers/norm, LoRA where present, decoder) from state tensors using canonical names, dtype, shape, and bytes.
- [ ] Compute the effective-policy hash from policy ID, schema identities, component hashes, checkpoint hashes, and immutable composition metadata.
- [ ] Resolve and fully materialize the requested archive model; compare actual component/effective hashes to the registry and raise `PolicyIdentityViolation` on any mismatch.
- [ ] Add promoted-snapshot manifest validation that reconstructs immutable base plus declared delta and rejects changed bases or deltas.
- [ ] Run `python3 -m unittest -v train.0040_dragapult_0809_action_boundary_rl.tests.test_policy_identity`.

### Task 3: Full-Policy CUDA Routing

**Files:**
- Modify: `engine_cuda/python/ptcg_cuda_engine/semantic0031_router.py`
- Modify: `train/0040_dragapult_0809_action_boundary_rl/rollout/cuda_collector.py`
- Modify: `train/0040_dragapult_0809_action_boundary_rl/semantic_parity/run_gate_d_lockstep.py`
- Test: `train/0040_dragapult_0809_action_boundary_rl/tests/test_cuda_collector.py`
- Test: `train/0040_dragapult_0809_action_boundary_rl/tests/test_policy_identity.py`

**Interfaces:**
- Consumes: `ResolvedPolicy.model`, `ResolvedPolicy.audit`, and complete `Semantic0031DeviceAdapter` instances.
- Produces: a router that either uses identical full adapters or fails before executing; no partial frozen-head route may claim a registered policy ID.

- [ ] Write a failing grouped-vs-independent fixed-input inference test.
- [ ] Require an opponent full-policy adapter when focal/opponent effective hashes differ.
- [ ] Remove the CUDA collector construction that combines focal trunk with opponent layer 1/norm/decoder.
- [ ] Attach requested/materialized identity audits to collector diagnostics and expose them for manifests.
- [ ] Update lockstep parity to use the same full-policy adapter path.
- [ ] Run the focused router/collector/identity tests.

### Task 4: Shared Training And Evaluation Resolution

**Files:**
- Modify: `train/0040_dragapult_0809_action_boundary_rl/training/run_full_semantic.py`
- Modify: `train/0040_dragapult_0809_action_boundary_rl/evaluation/run_update0_frozen.py`
- Modify: `train/0040_dragapult_0809_action_boundary_rl/evaluation/diagnose_frozen_failures.py`
- Modify: `train/0040_dragapult_0809_action_boundary_rl/semantic_parity/run_gate_d_lockstep.py`
- Test: `train/0040_dragapult_0809_action_boundary_rl/tests/test_policy_identity.py`
- Test: `train/0040_dragapult_0809_action_boundary_rl/tests/test_full_semantic_policy.py`

**Interfaces:**
- Consumes: the single `resolve_policy` API and immutable policy IDs.
- Produces: identical effective opponent identity for PPO rollout and frozen evaluation, independent of sample/greedy selection mode.

- [ ] Replace direct Frozen-0806 loading with policy-ID resolution shared by training, evaluation, diagnostics, and parity.
- [ ] Add preflight fatal audit before rollout/evaluation collector construction.
- [ ] Keep action selection mode outside policy materialization so sample and greedy use identical weights.
- [ ] Run focused training/evaluation construction tests without starting a formal run.

### Task 5: Formal Manifest And Historical Evidence Boundary

**Files:**
- Modify: `train/0040_dragapult_0809_action_boundary_rl/training/run_full_semantic.py`
- Modify: `train/0040_dragapult_0809_action_boundary_rl/evaluation/run_update0_frozen.py`
- Modify: `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`
- Create or modify: a scoped audit note under `experiments/0040_dragapult_0809_action_boundary_rl/`

**Interfaces:**
- Consumes: `PolicyIdentityAudit.to_manifest()`.
- Produces: formal result manifests containing candidate/checkpoint identity, opponent policy/effective hash, execution path, lanes, selection mode, seeds/distribution, repository commit/config, and `policy_identity_audit: PASS`.

- [ ] Add the identity audit payload to new formal evaluation and RL run artifacts.
- [ ] Make report validity fail closed unless the audit status is `PASS`.
- [ ] Record known historical shared-trunk results as `Hybrid-opponent / invalid-as-Frozen0806` without modifying or deleting old result files.
- [ ] Assert Frozen-0806 and Frozen-0809 outputs use distinct benchmark IDs and cannot be auto-aggregated.

### Task 6: Four-Seed CPU/CUDA Full-0806 Gate

**Files:**
- Modify: `train/0040_dragapult_0809_action_boundary_rl/semantic_parity/run_gate_d_lockstep.py` only if needed for reusable CLI/report fields.
- Create: `.tmp/evaluation/policy_identity_full0806_four_seed/<run-id>/report.json` (ignored temporary evidence).

**Interfaces:**
- Consumes: seeds `2078498379`, `1145059092`, `1142670790`, `899381985`, lane 1, greedy, one focal checkpoint, canonical CPU Full-0806 and resident CUDA Full-0806.
- Produces: identity PASS, previous-divergence disappearance count, and full trajectory exact-match count with actor/state/observation/legal/action/next-state comparisons.

- [ ] Run the identity preflight alone and require PASS.
- [ ] Run each seed through CPU canonical and CUDA resident lockstep with the same focal checkpoint.
- [ ] Persist a temporary report showing whether decision `1/24/6/6` divergences disappeared and whether all trajectories exactly match.
- [ ] Do not run CUDA-2048 if any seed fails.

### Task 7: Final Verification And Audit Report

**Files:**
- Verify all modified files and repository status.

**Interfaces:**
- Produces: evidence for the user's 12 requested delivery items and final PASS/PARTIAL/FAIL conclusion.

- [ ] Run all new regression tests plus relevant existing CUDA collector, frozen policy, checkpoint, and parity tests.
- [ ] Run `git diff --check`.
- [ ] Search executable paths for partial opponent construction and classify every remaining match.
- [ ] List exact modified files without claiming unrelated user changes.
- [ ] Report blockers explicitly; only declare PASS if no silent Hybrid path remains and the four-seed gate is 4/4.

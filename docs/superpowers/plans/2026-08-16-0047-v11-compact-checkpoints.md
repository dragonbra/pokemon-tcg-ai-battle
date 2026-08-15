# 0047 V11 Compact Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restart 0047 from immutable Policy-0814 with exact focal deck 007 while reducing per-update FP32 checkpoints to reconstructable deltas and retaining only successfully evaluated five-update nodes.

**Architecture:** A project-local checkpoint module owns delta selection, atomic persistence, immutable-base identity validation, strict reconstruction, and post-evaluation pruning. The PPO runner always saves before evaluation; evaluation reloads that saved checkpoint into a fresh audited Policy-0814 model and hard-fails on any live/reloaded effective-state mismatch. V10 remains a stopped historical version, while the new semantics use V11.

**Tech Stack:** Python 3.11, PyTorch, pytest, official CUDA engine runtime, W&B/TensorBoard training telemetry.

## Global Constraints

- Do not modify `engine/source/`.
- V10 cannot be reused because it already contains metrics and checkpoints; allocate strict next version V11.
- PPO/model-only checkpoints remain FP32 and contain no optimizer, scheduler, RNG, replay, or rollout state.
- Policy-0814 immutable actor/value hashes are validated before reconstruction.
- Checkpoint deletion is limited to non-evaluated V11 nodes and occurs only after a successful U5/U10/... evaluation; the user explicitly authorized this retention exception on 2026-08-16.
- U0 and every multiple-of-five checkpoint remain available as source FP32 policy evidence.
- Every Kaggle-facing evaluation continues to materialize full FP16 storage and strict-load FP32 runtime weights.

---

### Task 1: Preserve the stopped V10 audit record

**Files:**
- Modify: `rl_runs/0047_meta_routed_moe_rl/versions/V10_deck070_policy0814_moe7_win_only_two_pool/artifact/status.json`

**Interfaces:**
- Consumes: the final durable V10 update number and explicit user deletion authorization.
- Produces: a fail-closed historical status that does not claim resumability or valid retained candidate evidence.

- [ ] Stop the process and verify its CUDA context is gone.
- [ ] Record `stopped_by_user`, update 4, and the deleted-weight scope.
- [ ] Delete only `checkpoint/update-000000.pt` through `update-000004.pt` and `artifact/evaluation/update-000000/candidate_fp16.pt`.
- [ ] Verify V10 logs, metrics, router tables, report JSON, TensorBoard, and W&B metadata remain.

### Task 2: Add the compact checkpoint contract

**Files:**
- Create: `train/0047_meta_routed_moe_rl/training/moe_checkpoint.py`
- Modify: `train/0047_meta_routed_moe_rl/tests/test_moe_contract.py`

**Interfaces:**
- Produces: `build_compact_checkpoint(model, update, metadata) -> dict`, `load_compact_checkpoint(model, payload) -> None`, `atomic_save_compact_checkpoint(...) -> Path`, and `prune_non_eval_checkpoints(...) -> tuple[Path, ...]`.

- [ ] Write a failing round-trip test that perturbs expert/value/router state, persists only non-`actor.*` FP32 tensors, reconstructs from a fresh audited base, and requires exact tensor equality.
- [ ] Write rejection tests for a changed Policy-0814 base hash, unexpected/missing delta keys, non-FP32 floating tensors, and forbidden optimizer/replay fields.
- [ ] Write a retention test proving U1-U4 are removed only after U5 PASS while U0/U5 and router JSON remain.
- [ ] Implement exact delta inventory validation and atomic save.
- [ ] Run the focused checkpoint tests.

### Task 3: Bind V11 training and evaluation to saved policy identity

**Files:**
- Modify: `train/0047_meta_routed_moe_rl/training/run_v1_moe.py`
- Modify: `train/0047_meta_routed_moe_rl/evaluation/moe_three_pool.py`
- Modify: `train/0047_meta_routed_moe_rl/tests/test_moe_contract.py`

**Interfaces:**
- Consumes: compact checkpoint functions from Task 2.
- Produces: V11 exact-deck-007 launch and source-checkpoint-bound U0/U5/... evaluation.

- [ ] Change the formal version to `V11_deck007_policy0814_moe7_compact_eval5_retention` and focal deck/taxonomy to `007`/`0`.
- [ ] Save U0 and every update as compact FP32 model-only checkpoints before downstream telemetry.
- [ ] Reload each due evaluation checkpoint into a fresh Policy-0814 model and compare deployment-effective tensor hashes against the live model before games begin.
- [ ] Record source checkpoint path/SHA-256, portable FP16 SHA-256, deployment-effective SHA-256, and PASS in the evaluation report.
- [ ] After a successful multiple-of-five evaluation, remove only compact checkpoints whose updates are not divisible by five.
- [ ] Record `evaluated_nodes_only_after_successful_eval` and the explicit retention authorization in training config/status.

### Task 4: Synchronize authoritative design documentation

**Files:**
- Modify: `experiments/0047_meta_routed_moe_rl/DESIGN.md`
- Modify: `experiments/0047_meta_routed_moe_rl/DESIGN.html`

**Interfaces:**
- Consumes: final V11 constants and checkpoint schema.
- Produces: matching Markdown/HTML facts for focal deck, taxonomy, policy reconstruction, retention, evaluation identity, and current phase.

- [ ] Replace V10/deck-070 current-run statements with V11/deck-007 facts while retaining the V10 stop history.
- [ ] Document compact FP32 delta contents, immutable Policy-0814 dependencies, strict reconstruction, and approximate byte reduction.
- [ ] Document the user-authorized eval-node retention exception and fail-safe ordering.
- [ ] Cross-check both documents against code constants and training config fields.

### Task 5: Verify and launch

**Files:**
- Verify: `train/0047_meta_routed_moe_rl/tests/`
- Create at runtime: `rl_runs/0047_meta_routed_moe_rl/versions/V11_deck007_policy0814_moe7_compact_eval5_retention/`

**Interfaces:**
- Consumes: all preceding implementation.
- Produces: a live formal V11 W&B-online PPO process.

- [ ] Run focused MoE/checkpoint tests and the broader 0047 test subset affected by checkpoint/evaluation loading.
- [ ] Run launch readiness without creating V11 artifacts and confirm all four version directories are unused.
- [ ] Start formal V11 in a new tmux session with W&B online.
- [ ] Confirm U0 compact checkpoint size/inventory, deployment evaluation identity PASS, process health, GPU residency, and log progression.
- [ ] Do not report success until the first V11 artifacts demonstrate the compact schema and exact deck-007 identity.

# 0045 V16 Complete-Delta Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue deck-007 Policy-0814 PPO from V14 U105 while reinitializing the unrecoverable StateEncoder LoRA and guaranteeing every future model-only checkpoint exactly reconstructs the in-memory inference model from immutable Policy-0814 plus a complete delta.

**Architecture:** Introduce a V2 complete-delta checkpoint schema whose saved keys are derived from `model.named_parameters()` before reading `state_dict()`, not from detached tensor `requires_grad`. Build V16 U105 by loading the preserved V14 U105 delta into a deterministically initialized V14 architecture, retain the newly initialized StateEncoder LoRA, save all 115 trainable tensors, reconstruct from immutable Policy-0814, and require full model-state equality before formal launch. V16 uses a fresh optimizer and fresh on-policy rollout, keeps the V14 exact-deck Policy-0814 schedule, and evaluates every five updates.

**Tech Stack:** Python 3.11, PyTorch, project-local PPO/CUDA Engine 2.0, pytest, TensorBoard, W&B online.

## Global Constraints

- Do not modify `engine/source/` or CUDA engine sources.
- V14 remains immutable and is labeled checkpoint-incomplete; do not rewrite its checkpoints or results.
- Parent is exact V14 U105 SHA-256 and source update 105.
- StateEncoder LoRA uses the unchanged V14 rank-16/alpha-16 Q/V/O and final family-fusion architecture, but starts from a newly initialized zero-effective LoRA state because the trained tensors are unrecoverable. Its V16 learning rate is `4e-5`, exactly twice V14; every other optimizer group remains at the V14 rate.
- Optimizer is fresh; rollout is newly collected on-policy data.
- Opponent is complete immutable Policy-0814 and the exact-deck quota schedule is unchanged.
- Checkpoint retention is all; no optimizer, scheduler, RNG, rollout buffer, or replay state is stored.
- Every V2 checkpoint must contain all 115 trainable parameter tensors and reconstruct the entire model state exactly from immutable Policy-0814.
- Periodic evaluation remains greedy Policy-0814 exact-deck CUDA-512 every five updates under `kaggle_fp16_storage_fp32_runtime_v1`.

---

### Task 1: Complete-Delta Checkpoint Contract

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/training/checkpointing.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/run_v1.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/evaluation/candidate.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_complete_delta_checkpoint.py`

**Interfaces:**
- Produces: `build_complete_delta_checkpoint(model, update, metadata)`, `load_complete_delta(model, payload)`, and `audit_reconstruction(source_model, reconstructed_model, payload)`.

- [ ] Write a failing regression proving the legacy filter loses 16 StateEncoder LoRA tensors.
- [ ] Write a failing round-trip test requiring all 115 trainable names in the V2 payload and full `state_dict()` equality after reconstructing from Policy-0814.
- [ ] Implement the V2 schema with explicit trainable-name inventory, per-tensor hashes, immutable-base identity, and forbidden optimizer/RNG fields.
- [ ] Replace future `_checkpoint` output with V2 and hard-load it immediately after every atomic write.
- [ ] Make candidate materialization accept V1 historical inputs and require complete V2 coverage for new inputs.
- [ ] Run focused checkpoint/candidate tests.

### Task 2: V16 U105 Handoff

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/training/run_v16_complete_delta_u105.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_v16_complete_delta_u105.py`

**Interfaces:**
- Consumes: V14 U105 checkpoint, immutable Policy-0814 Actor/Value, V14 adaptation and rollout contract.
- Produces: V16 `checkpoint/update-000105.pt`, deterministic complete reference artifact, readiness and reconstruction audits.

- [ ] Write a failing readiness test requiring exact V14 parent hash/status and an unused V16 version path.
- [ ] Build the U105 model with a recorded initialization seed, load every preserved V14 tensor, and assert the only absent V14 trainables are the known 16 StateEncoder LoRA tensors.
- [ ] Save U105 under the complete-delta V2 schema and reconstruct it from immutable Policy-0814.
- [ ] Require full tensor inventory/value equality, identical deterministic greedy probe logits/actions, and 115/115 trainable coverage.
- [ ] Build a deterministic complete Policy-0814 reference anchor with zero-effective StateEncoder LoRA.
- [ ] Run V16 readiness and focused tests without creating formal version directories.

### Task 3: Formal Continuation and Evaluation

**Files:**
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`

**Interfaces:**
- Produces: formal V16 training artifacts, W&B run, every-update complete checkpoints, and U105/U110/... Policy-0814 eval512 reports.

- [ ] Run the complete 0045 test suite.
- [ ] Materialize V16 U105 and run its baseline Policy-0814 eval512; require identity PASS, 512 terminal, zero errors/unfinished.
- [ ] Launch V16 with fresh optimizer/on-policy data from update 105 and no automatic update limit.
- [ ] Confirm U106 checkpoint is complete-delta V2, reconstructs exactly, and its canonical PPO metric/W&B row is durable.
- [ ] Confirm the process continues toward U110 and that the next automatic eval cadence is U110.
- [ ] Update DESIGN/decisions/status with the V14 defect boundary, V16 parent/init/reference identities, reconstruction hard gate, baseline result, and active W&B run.

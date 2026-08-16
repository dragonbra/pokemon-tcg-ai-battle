# 0045 V13 Policy-0814 Shared-Encoder LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start 0045 V13 from immutable Policy-0814 with focal deck 007, one shared ActionDecoder, r=16 Q/V/O LoRA on the final Option/board/event blocks, and r=16 LoRA on the final state family-fusion MLP.

**Architecture:** Keep the existing 0045 single Actor/Critic and full-action contract. Materialize independent focal and opponent Policy-0814 instances; expand only the requested shared encoder locations while preserving the existing ActionDecoder, Allocation, Value, Value-adapter, and Prize trainable set. Use an exact-deck base-quota schedule plus a seeded random remainder, all against Policy-0814.

**Tech Stack:** Python 3, PyTorch parametrizations, CUDA Engine 2.0, official engine runtime, pytest, W&B/TensorBoard.

## Global Constraints

- V13 is a fresh Policy-0814 initialization, not a continuation from V8 U299.
- Focal deck is exact deck 007; all 512 rollout opponents use Policy-0814.
- Fixed deck quotas are 001=143, 002=68, 003=48, 007=64, 071=30, 008=14, 009=16, 011=7.
- The remaining 122 lanes sample only 071/008/009/011 from deterministic per-update seeds.
- Focal and opponent effective weights and tensor storage remain independent.
- No official engine source is modified.
- Model-only checkpoints remain FP32 with checkpoint retention `all`.

---

### Task 1: Register Policy-0814 in 0045

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/assets/policies/definitions/policy_0814/model.pt`
- Create: `train/0045_single_deck_expert_minimal_lora/assets/policies/manifests/policy_0814.json`
- Modify: `train/0045_single_deck_expert_minimal_lora/assets/policies/registry.json`
- Modify: `train/0045_single_deck_expert_minimal_lora/assets.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/policy_identity.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/runtime.py`

**Interfaces:**
- Consumes: immutable archive actor SHA-256 `d7921f...897b`.
- Produces: `materialize_policy_bundle(..., "Policy-0814")` and `load_policy("Policy-0814", deck_id=...)`.

- [x] Verify the copied actor file hash and the eight effective component hashes.
- [x] Make registry validation admit Policy-0814 without removing historical immutable policies.
- [x] Strict-load Policy-0814 through the project-local runtime and reject mismatched identity.
- [x] Add a regression test that materialization returns effective SHA-256 `476d57...2580`.

### Task 2: Implement the V13 LoRA boundary

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/policy/minimal_lora_actor_critic.py`
- Modify: project-local LoRA helper modules selected after source audit.
- Create: `train/0045_single_deck_expert_minimal_lora/tests/test_v13_shared_encoder_lora.py`

**Interfaces:**
- Consumes: complete Policy-0814 state dict.
- Produces: a single shared Actor with r=16 Q/V/O LoRA at the requested final Option, board, event, and family-fusion locations.

- [x] Write a tensor audit test for exact trainable names, ranks, and frozen early layers.
- [x] Generalize LoRA injection to Q/V/O with rank 16 while preserving zero-residual initialization.
- [x] Attach rank-16 LoRA to every major final family-fusion Linear and nowhere earlier.
- [x] Keep ActionDecoder fully trainable and preserve the existing non-Actor RL trainable tensors.
- [x] Strict-load Policy-0814 base weights before injecting trainable LoRA and verify focal/opponent storage disjointness.

### Task 3: Implement exact-deck rollout scheduling

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/league/exact_deck_quota.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/run_v1.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/telemetry.py`
- Create: `train/0045_single_deck_expert_minimal_lora/tests/test_v13_exact_deck_quota.py`

**Interfaces:**
- Produces: `exact_deck_quota_schedule(random_seed, shuffle_seed, mappings, lanes, fixed_deck_quotas, random_deck_ids)`.

- [x] Test exact 512 accounting, fixed lower bounds, random-pool exclusivity, and same-seed reproducibility.
- [x] Add `exact_deck_quota_training_pool` to `_jobs` and formal validation.
- [x] Log realized per-deck games and the single Policy-0814 distribution.

### Task 4: Define and smoke V13

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/training/run_v13_policy0814_shared_encoder_lora.py`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`

**Interfaces:**
- Produces: pristine `V13_dragapult_007_policy0814_shared_encoder_lora_r16` formal run and trainable tensor audit JSON.

- [x] Require a pristine V13 artifact/checkpoint/TensorBoard/W&B boundary.
- [x] Generate the first 512-job schedule and assert every job uses Policy-0814.
- [x] Run candidate FP16-storage/FP32-runtime materialization smoke without strength claims.
- [x] Emit every trainable tensor name, shape, parameter count, module, LoRA rank, and total.
- [x] Update both authoritative design formats with the exact measured audit.
- [ ] Launch W&B-online formal training only after all gates pass.

### Task 5: Add the V13 Policy-0814 eval512

**Files:**
- Create: `evaluation/policy0814_exact_deck_schedule.py`
- Create: `evaluation/run_policy0814_exact_deck_cuda512.py`
- Modify: `training/periodic_evaluation.py`
- Modify: `training/run_v1.py`

- [x] Freeze one common-random-number 512-game schedule using the rollout quota model.
- [x] Materialize the candidate as FP16 storage and strict-load FP32 runtime before scoring.
- [x] Use only immutable Policy-0814 and assert focal/opponent storage disjointness.
- [x] Emit aggregate and per-deck W/L/D/win-rate summaries; retain game rows only for audit.
- [x] Dispatch this profile synchronously every five completed updates.

## Self-Review

- Spec coverage: single decoder, all four requested r=16 injection families, frozen early layers, unchanged Critic boundary, deck 007, Policy-0814-only, 512-game distribution, audit, smoke, and launch are covered.
- Placeholder scan: no deferred implementation placeholders remain.
- Type consistency: the scheduler and V13 runner consume the same explicit quota/rand-pool arguments added to `run_v1.run` and `_jobs`.

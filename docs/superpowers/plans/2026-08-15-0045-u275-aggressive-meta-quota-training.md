# 0045 U275 Aggressive Meta-Quota Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue deck-007 from exact V4 U275 with an exact 512-game rollout quota of Meta 03=200, Meta 05=200, Meta 02=50, and 62 games uniformly spread across every other active Meta, while running the unchanged three-pool Policy-0809 evaluation after every update.

**Architecture:** Add a focused exact-quota rollout scheduler rather than overloading proportional weights. Start a new V6 repository version with a fresh optimizer and fresh on-policy collection from V4 U275, retaining Frozen-0045-Init as the reference and complete immutable Policy-0809 as the sampled rollout opponent. V5 is already occupied by the formal deck-023 U105 evaluation identity and may not be reused. Reuse the existing one-materialization, three-pool Policy-0809 evaluation, but parameterize its cadence to every update.

**Tech Stack:** Python 3.11, PyTorch PPO, CUDA Engine 2.0, official engine runtime, pytest, TensorBoard, W&B.

## Global Constraints

- Do not mutate or append to V4; V4 is stopped after complete U275 evaluation.
- The V6 parent is exact V4 `update-000275.pt`; optimizer and on-policy data are fresh.
- Rollout contains exactly 512 games: Meta 03=200, Meta 05=200, Meta 02=50, remaining active Meta classes total=62.
- The 62 remaining games are balanced by Meta first, then by exact deck within each Meta.
- Rollout opponent is locked to complete immutable Policy-0809.
- Evaluation remains complete immutable Policy-0809 with the existing three disjoint 512-game pools.
- Evaluation runs after every completed update, synchronously and before the next rollout.
- LR, entropy coefficient, PPO objective, rollout size, reference anchor, and model architecture remain unchanged.
- Checkpoints remain model-only with retention `all`; no cleanup or deletion is authorized.
- Every candidate evaluation must pass `kaggle_fp16_storage_fp32_runtime_v1`, official-engine terminal, routing, and identity gates.

---

### Task 1: Exact aggressive rollout schedule

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/league/aggressive_meta_quota.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_aggressive_meta_quota.py`

**Interfaces:**
- Consumes: `Sequence[DeckMapping]`, quota/shuffle seeds, fixed quota mapping.
- Produces: `aggressive_meta_quota_schedule(...) -> tuple[MetaDeckSlot, ...]`.

- [ ] Write a failing test requiring exact counts `{2: 50, 3: 200, 5: 200}`, exactly 62 remaining lanes, per-remaining-Meta counts differing by at most one, and per-Meta exact-deck counts differing by at most one.
- [ ] Run `python3 -m pytest train/0045_single_deck_expert_minimal_lora/tests/test_aggressive_meta_quota.py -q` and verify the missing module fails.
- [ ] Implement deterministic exact-quota allocation with fatal validation for inactive fixed classes, non-positive totals, quota sum overflow, or non-contiguous deck mappings.
- [ ] Rerun the focused test and require PASS.

### Task 2: Generic rollout schedule and evaluation cadence hooks

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/training/run_v1.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/periodic_evaluation.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_aggressive_meta_quota.py`

**Interfaces:**
- Consumes: `opponent_sampling_mode="aggressive_meta_quota_training_pool"`, `opponent_policy_ids=("Policy-0809",)`, `opponent_meta_quotas={2: 50, 3: 200, 5: 200}`, and `periodic_evaluation_interval_updates=1`.
- Produces: exact scheduled rollout jobs, canonical config metadata, and `is_due(checkpoint_update, interval_updates=...)` cadence checks.

- [ ] Add failing tests that the generic runner accepts only the explicit aggressive mode/quota combination and that interval 1 marks every positive update due.
- [ ] Parameterize cadence without changing the historical five-update default.
- [ ] Route the aggressive mode through the exact-quota scheduler and record both requested and realized Meta counts in training config/rollout metrics.
- [ ] Require the synchronous Policy-0809 three-pool evaluation after every durable V5 checkpoint.
- [ ] Run focused tests and require PASS.

### Task 3: V6 U275 entry point and readiness audit

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/training/run_v6_u275_aggressive_meta_quota.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_aggressive_meta_quota.py`

**Interfaces:**
- Consumes: exact V4 U275 checkpoint and Frozen-0045-Init checkpoint.
- Produces: `readiness()` and `launch(wandb_mode="online", updates=None)`.

- [ ] Add a failing contract test for start update 275, exact quotas, Policy-0809 rollout and three-pool eval, interval 1, unchanged cold-start LR, and unlimited updates.
- [ ] Implement hash-locked parent/reference readiness, unused-version gates, first-schedule realized counts, and W&B identity.
- [ ] Run readiness and focused tests; require exact quota audit and PASS.

### Task 4: Authoritative design synchronization

**Files:**
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`

**Interfaces:**
- Consumes: tested runtime constants and exact V4 U275 results.
- Produces: authoritative V5 architecture/training/evaluation record.

- [ ] Record V4 stop boundary and U275 three-pool results.
- [ ] Document V6 parent/reference split, exact quotas, remaining-Meta balancing, unchanged model/PPO settings, and every-update evaluation cost/semantics.
- [ ] Cross-check HTML and Markdown against the runtime readiness payload.

### Task 5: Verification and launch

**Files:**
- Verify: `train/0045_single_deck_expert_minimal_lora/tests/`
- Generate: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V6_dragapult_007_u275_aggressive_meta_quota_policy0809/`

**Interfaces:**
- Consumes: V5 entry point and all hard gates.
- Produces: running formal V6 with the exact existing V4 U275 baseline provenance and U276 every-update evaluation.

- [ ] Run focused tests, then `python3 -m pytest train/0045_single_deck_expert_minimal_lora/tests -q`.
- [ ] Run readiness and verify GPU availability, storage, immutable identities, and unused output paths.
- [ ] Launch V6 online without an update limit; record the already-complete V4 U275 three-pool report as immutable baseline provenance without rerunning or duplicating its materialization.
- [ ] Verify the first Policy-0809 rollout realizes exactly 200/200/50/62 and U276 evaluation is written before any U276-source rollout begins.

### Task 6: Preserve V6 telemetry failure and continue as V7

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/telemetry.py`
- Create: `train/0045_single_deck_expert_minimal_lora/training/run_v7_u276_aggressive_meta_quota.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_aggressive_meta_quota.py`

**Interfaces:**
- Consumes: durable V6 U276 model-only checkpoint after the telemetry-only failure.
- Produces: telemetry acceptance for the explicit aggressive mode and a fresh-optimizer V7 continuation that runs U276 baseline eval before new rollout.

- [ ] Preserve V6 as failed after durable U276: zero canonical metric rows and no V6 eval report.
- [ ] Add a failing telemetry test reproducing the unsupported-mode exception.
- [ ] Extend the telemetry mode contract once, recording both Meta-balanced and aggressive-quota flags; run focused and full regression tests.
- [ ] Bind V7 to exact V6 U276, rerun the missing U276 three-pool baseline, then continue with fresh optimizer/on-policy data and every-update eval.

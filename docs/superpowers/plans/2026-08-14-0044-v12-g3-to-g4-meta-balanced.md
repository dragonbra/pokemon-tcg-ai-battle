# 0044 V12 G3-to-G4 Meta-Balanced Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the next 0044 PPO version from G3 with one immutable complete Champion-G3 opponent policy, balanced active Meta Archetypes on both focal and opponent decks, and small policy-only Meta residual capacity.

**Architecture:** Each 512-game rollout assigns the 28 non-empty Own Archetype V2 classes 18 or 19 lanes, then frequency-balances member exact decks with seeded shuffles. Focal and opponent schedules use independent namespaces so their marginals match without artificial pairing. The actor adds 29 resident rank-4 Meta residual experts after the shared policy strategy adapter; Value and frozen opponent-Meta prediction remain unchanged. Formal launch hard-fails until Champion-G3 has a promoted immutable manifest.

**Tech Stack:** Python 3.12, PyTorch, PPO Protocol V2, CUDA Engine 2.0 resident routing, pytest, W&B.

## Global Constraints

- Formal repository version is V12 because V11 is already the immutable Full-67 G2/G3 evaluation version.
- Do not mutate V10 checkpoints, V11 reports, Champion-G2 or Policy-0809.
- Champion-G3 opponent and Reference-G3 must resolve independently through a complete immutable policy manifest.
- Focal PPO/master/checkpoint weights remain FP32; deployment evidence remains FP16 storage to FP32 runtime.
- Schedule controls only coin-winner seeds; the winning complete Agent chooses at official context 41.
- Rollout is 512 games, logical minibatch 4096, physical minibatch 1024, gradient accumulation 4, PPO epochs 3.
- Actor decoder/strategy/allocation LR is `5e-6`, Option LoRA LR is `1e-5`, Meta residual LR is `5e-6`, and all Value groups use `2e-5`.
- Update `DESIGN.md` and `DESIGN.html` in the same change.

---

### Task 1: Balanced hierarchical deck schedules

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/league/meta_balanced.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_v12_meta_balanced_schedule.py`

**Interfaces:**
- Produces: `balanced_meta_deck_schedule(seed: int, lanes: int, mappings: Sequence[DeckMapping]) -> tuple[MetaDeckSlot, ...]`.

- [ ] Test exact 512 lanes, 28 active classes, 18/19 classes, member-deck count spread at most one, deterministic replay and full deck coverage.
- [ ] Test independent focal/opponent namespaces preserve identical Meta counts but do not force pair equality.
- [ ] Implement deterministic rotating remainder allocation and final seeded lane shuffle.
- [ ] Run the focused schedule tests.

### Task 2: Policy-only Meta Actor Residual

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy/strategy_adapters.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy/actor_critic.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/semantic_runtime/deployment/compound_inference.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_meta_actor_residual.py`

**Interfaces:**
- Produces: `MetaActorResidual(width=320, classes=29, rank=4)` and a policy-only residual application keyed by own archetype ID.

- [ ] Test exactly 74,240 parameters, parent/zero-compatible initialization and bit-identical U0 actor forward.
- [ ] Test actor loss reaches selected residual B tensors while Value loss reaches none of them.
- [ ] Test batched mixed-Meta routing equals per-row inference and does not fragment policy identity.
- [ ] Implement resident gather/batched low-rank residual math and deployment materialization inventory.
- [ ] Run focused model, candidate round-trip and policy-isolation tests.

### Task 3: Champion-G3 immutable gate

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/assets.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy_identity.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/promote/champion_g3.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_champion_g3_gate.py`

**Interfaces:**
- Consumes: completed V11 Full-67 paired evidence and explicit human PROMOTE decision.
- Produces: immutable `Champion-G3` complete effective policy manifest, portable artifact and effective hash.

- [ ] Test missing/incomplete Full-67 evidence fails closed.
- [ ] Test an unpromoted raw U110 checkpoint cannot resolve as Champion-G3.
- [ ] Test focal changes never change Champion-G3 effective hash or tensor storage.
- [ ] Implement promotion preparation without making the human PROMOTE decision automatically.
- [ ] Run identity and package round-trip tests.

### Task 4: V12 formal training entrypoint

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/training/run_v12.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/run_v1.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/ppo_full_semantic.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_v12_g4_training.py`

**Interfaces:**
- Consumes: promoted Champion-G3 policy, source FP32 G3 U110 checkpoint and both Meta-balanced schedules.
- Produces: formal V12 readiness and launch with G3 reference/opponent isolation.

- [ ] Test V12 cannot launch before Champion-G3 promotion.
- [ ] Test 512 focal and 512 opponent assignments, 512 terminal trajectories and one resident opponent policy load.
- [ ] Test logical minibatch 4096, physical 1024, three epochs and approximately unchanged optimizer-step budget.
- [ ] Test exact optimizer group LRs including Meta residual `5e-6` and Value `2e-5`.
- [ ] Implement fresh optimizer, local U0, periodic U0 evaluation and model-only checkpoint retention.
- [ ] Run readiness and focused training tests.

### Task 5: Documentation and verification

**Files:**
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md`
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html`
- Modify: `train/0044_g2_dragapult_policy_option_lora/README.md`

**Interfaces:**
- Produces: authoritative G3-to-G4 model, schedule, optimizer and promotion boundaries.

- [ ] Document field/tensor shapes, Meta residual parameter inventory and Value isolation.
- [ ] Document exact focal/opponent hierarchical distributions and independent seeds.
- [ ] Document V11 evaluation versus V12 training identity.
- [ ] Run all 0044 tests, CUDA context-41 smoke, mixed-Meta forward parity and `git diff --check`.

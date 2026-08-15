# 0047 Win-Only PPO and Two-Pool Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restart 0047 as a new version whose Actor learns only from terminal win/loss advantage and whose fixed evaluation consists of a U0 baseline plus two CUDA-512 physical pools every five updates.

**Architecture:** Keep the Policy0814 seven-expert MoE, sparse E0-to-specialist routing, Frozen0814 opponent, rollout schedule, critic topology, and effective-policy PPO implementation unchanged. Set the Actor Prize Advantage coefficient to zero while retaining prize telemetry/critic diagnostics, and replace the three physical eval pools with one 512-game focus pool split 256/256 between the old-three and new-four exact decks plus one 512-game remain pool.

**Tech Stack:** Python 3.11, PyTorch PPO, official CUDA engine, pytest, W&B, model-only checkpoints.

## Global Constraints

- Do not mutate or append to V8/V9; create a fresh V10 repository version and W&B run after V9's pre-training heatmap dependency failure.
- Actor advantage is terminal win/loss GAE only; Prize Advantage must have exactly zero Actor coefficient.
- Preserve directional prize telemetry and the Critic-side Prize auxiliary objective as diagnostics/scaffolding only.
- Evaluate U0 before collecting any PPO rollout, then U5/U10/etc.
- The focus CUDA-512 schedule is exactly 256 old-three games and 256 new-four games.
- Old-three exact decks are `001/002/011`; new-four exact decks are `007/003/009/023`.
- Remain-meta is CUDA-512 and excludes all seven focus exact decks.
- Report focus, old-three, new-four, remain-meta, every evaluated exact deck, and per-meta aggregates.
- Generate and upload `router/meta_expert_heatmap` at U0 and every later eval point.
- Do not modify `engine/source/` or any V8 artifact.

---

### Task 1: Lock the new reward and schedule contracts with tests

**Files:**
- Modify: `train/0047_meta_routed_moe_rl/tests/test_moe_contract.py`

**Interfaces:**
- Consumes: `evaluation.moe_three_pool_schedule.materialize(...)`, `training.run_v1_moe` constants.
- Produces: regression assertions for exact 256/256 focus allocation, remain exclusion, U0 cadence, and zero Actor Prize Advantage.

- [ ] Add a test asserting the physical pools are `focus_seven` and `remain_meta`, with focus subgroup counts 256/256 and exact per-deck balancing.
- [ ] Add a test asserting `PRIZE_AUX_ACTOR_WEIGHT == 0.0`, `EVAL_INTERVAL == 5`, and `EVAL_AT_U0 is True`.
- [ ] Run `python3 -m pytest -q train/0047_meta_routed_moe_rl/tests/test_moe_contract.py` and confirm the new assertions fail against V8 behavior.

### Task 2: Materialize and aggregate the two physical eval pools

**Files:**
- Modify: `train/0047_meta_routed_moe_rl/evaluation/moe_three_pool_schedule.py`
- Modify: `train/0047_meta_routed_moe_rl/evaluation/moe_three_pool.py`

**Interfaces:**
- Produces: `materialize(...)` with `focus_seven` and `remain_meta`; reports containing `pool_summaries`, `focus_group_summaries`, `per_deck`, `per_meta`, and `entries`.
- Produces W&B metrics `eval/focus_seven/*`, `eval/old_three/*`, `eval/new_four/*`, `eval/remain_meta/*`, and `eval/deck_NNN/*`.

- [ ] Build 256 deterministic old-three jobs balanced 85/85/86 and 256 deterministic new-four jobs balanced 64 each, then deterministically shuffle their combined focus schedule.
- [ ] Keep remain-meta at 512 deterministic meta-first/deck-balanced jobs excluding the seven focus decks.
- [ ] Aggregate old-three/new-four independently from the shared focus results and aggregate every exact deck present in both physical pools.
- [ ] Set evaluation lane count to the full 512 jobs and retain official greedy/no-oracle routing semantics.
- [ ] Run the schedule regression test and confirm exact counts, fixed hashes, and exclusions pass.

### Task 3: Add U0 evaluation and pure-win Actor PPO to V9

**Files:**
- Modify: `train/0047_meta_routed_moe_rl/training/run_v1_moe.py`
- Modify: `train/0047_meta_routed_moe_rl/training/ppo_moe.py`

**Interfaces:**
- Consumes: `evaluate_model(...)`, `wandb_metrics(...)`, `PPOConfig.prize_aux_actor_weight`.
- Produces: V9 model-only checkpoints, U0 evaluation evidence, and pure-win Actor optimization.

- [ ] Add `prize_aux_actor_weight: float = 0.0` to `PPOConfig`, validate it is zero for this run, and use it instead of the model preset's historical `0.10` in the Actor advantage expression.
- [ ] Change version/run identity to a new V9 directory and stable W&B run ID.
- [ ] After saving U0 and initializing the logger, run both eval pools, log all eval metrics at step 0, and upload the U0 router heatmap before entering the rollout loop.
- [ ] Preserve the U5/U10 cadence using the same evaluation helper and heatmap helper.
- [ ] Mark V8 interrupted at its last completed checkpoint without deleting or rewriting its checkpoints.

### Task 4: Synchronize design documentation and validate

**Files:**
- Modify: `experiments/0047_meta_routed_moe_rl/DESIGN.md`
- Modify: `experiments/0047_meta_routed_moe_rl/DESIGN.html`

**Interfaces:**
- Documents: pure-win Actor objective, Critic-only Prize auxiliary boundary, two physical eval pools, subgroup/deck metrics, and U0 cadence.

- [ ] Update both design documents with the exact reward equation and evaluation schedule.
- [ ] Run `python3 -m pytest -q train/0047_meta_routed_moe_rl/tests/test_moe_contract.py`.
- [ ] Run `python3 -m compileall -q train/0047_meta_routed_moe_rl`.
- [ ] Run a small schedule/materialization smoke and verify no true meta reaches the focal router.

### Task 5: Launch and observe the formal V9 run

**Files:**
- Runtime output: `rl_runs/0047_meta_routed_moe_rl/versions/V10_deck070_policy0814_moe7_win_only_two_pool/`

**Interfaces:**
- Produces: U0 two-pool report, U0 heatmap, W&B scalars, rollout/PPO progress, and model-only checkpoints.

- [ ] Launch the formal online W&B run in a dedicated tmux session.
- [ ] Confirm U0 produces exactly 1,024 terminal games, with 512 focus and 512 remain.
- [ ] Confirm W&B contains old-three, new-four, remain-meta, per-deck metrics, and the U0 router heatmap.
- [ ] Confirm U1 pre-update effective logprob parity, finite gradients, and checkpoint creation.
- [ ] Leave the indefinite 070/Frozen0814 training process running unless a hard correctness failure occurs.

# 0043 Mixed-Focal Cohort V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume 0043 from V1 checkpoint 21 with focal decks 002 and 007 mixed inside each opponent-policy CUDA cohort, reducing four sequential cohorts to two without changing policy identity, exact-deck conditioning, PPO semantics, or PFSP chronology.

**Architecture:** Keep one mutable focal policy and two immutable resident opponent policies. Resolve and group only by opponent effective-policy identity; carry each lane's focal exact-deck ID/hash/static features and own-archetype ID through the same permutation and audit their association before inference and trajectory writeback. V2 starts from the V1 model-only checkpoint with a fresh optimizer and preserved C002 PFSP snapshot provenance.

**Tech Stack:** Python, PyTorch, CUDA Engine 2.0 resident runtime, pytest, W&B.

## Global Constraints

- Do not modify `engine/source/`.
- Do not share mutable focal tensors with either frozen opponent.
- Requested and materialized effective-policy identity mismatch is fatal.
- Focal deck identity is lane-static routing and conditioning data, not a separate policy identity.
- V1 remains immutable; V2 receives a new repository version and W&B run ID.
- Checkpoints remain model-only and all updates are retained.
- No CPU feature-compiler fallback; CUDA feature D2H remains zero.

---

### Task 1: Freeze the V1 boundary

**Files:**
- Modify: `rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/artifact/status.json`
- Modify: `experiments/0043_champion_league_rl/DECISIONS.md`

**Interfaces:**
- Consumes: V1 `checkpoint/update-000021.pt`, `artifact/pfsp_state.json`, and `training_metrics.jsonl`.
- Produces: immutable V2 parent identity and planned-stop audit.

- [ ] Verify checkpoint 21, metric row 21, `source_policy_update=20`, and `curriculum_version=C002` all agree.
- [ ] Record the planned stop and W&B synced state without rewriting V1 metrics.
- [ ] Hash checkpoint 21 and the C002 PFSP state for V2 provenance.

### Task 2: Specify and test mixed-focal lane identity

**Files:**
- Modify: `train/0043_champion_league_rl/tests/test_run_v1.py`
- Modify: `train/0043_champion_league_rl/tests/test_cuda_engine_2_routing.py`
- Modify: `train/0043_champion_league_rl/tests/test_focal_seed_runtime.py`

**Interfaces:**
- Consumes: `RolloutJob`, exact deck hash, focal own-archetype ID, resolved opponent policy identity.
- Produces: failing tests requiring two opponent-policy cohorts and mixed focal IDs within each cohort.

- [ ] Assert grouping keys contain only `opponent_policy_id`.
- [ ] Assert every cohort contains both focal deck IDs under the deterministic 256-game schedule.
- [ ] Assert lane permutations preserve game ID, focal deck ID/hash, own-archetype ID, opponent deck ID/hash, and opponent policy ID.
- [ ] Assert a deliberately mismatched focal deck hash or own-archetype ID fails closed.
- [ ] Run the focused tests and confirm they fail against the four-cohort implementation.

### Task 3: Implement two-cohort CUDA collection

**Files:**
- Modify: `train/0043_champion_league_rl/training/run_v1.py`
- Modify: `train/0043_champion_league_rl/rollout/cuda_collector.py`
- Modify: `train/0043_champion_league_rl/rollout/cuda_action_boundary.py`
- Modify: `train/0043_champion_league_rl/rollout/deck_routing.py`

**Interfaces:**
- Consumes: a sequence of jobs sharing one resolved opponent policy but carrying heterogeneous focal deck fields.
- Produces: one `ChunkedCudaRolloutCollector.collect()` result per opponent policy with original job ordering and identity audits intact.

- [ ] Group jobs by `opponent_policy_id` only.
- [ ] Build per-lane focal deck-static rows and own-archetype IDs without scalar cohort assumptions.
- [ ] Keep opponent Champion-G1 own-deck IDs per lane and Policy-0809 routing isolated.
- [ ] Audit the complete lane identity tuple before route compaction and after trajectory writeback.
- [ ] Run focused tests and confirm all pass.

### Task 4: Validate semantics and throughput

**Files:**
- Create: `.tmp/evaluation/0043_mixed_focal_cohort/semantic.json`
- Create: `.tmp/evaluation/0043_mixed_focal_cohort/performance.json`

**Interfaces:**
- Consumes: unchanged four-cohort baseline and new two-cohort implementation under identical deterministic schedules.
- Produces: PASS/FAIL semantic and performance gates.

- [ ] Run deterministic sample-mode identity audits for 002 and 007 against both opponent policies.
- [ ] Require 256 terminal games, zero errors/unfinished games, zero routing failures, and zero feature D2H.
- [ ] Confirm exactly three resident effective-policy loads and two execution cohorts.
- [ ] Benchmark at least three 256-game rollout repetitions after warmup.
- [ ] Report games/s, decisions/s, mean/max batch size, hot-loop time, materialization time, peak allocated/reserved memory, and old/new ratio.
- [ ] Run the complete 0043 test suite.

### Task 5: Create and launch V2 continuation

**Files:**
- Create: `rl_runs/0043_champion_league_rl/versions/V2_mixed_focal_cohort/`
- Modify: `experiments/0043_champion_league_rl/DESIGN.md`
- Modify: `experiments/0043_champion_league_rl/DESIGN.html`
- Modify: `experiments/0043_champion_league_rl/DECISIONS.md`
- Modify: `experiments/0043_champion_league_rl/active_training_config.json`

**Interfaces:**
- Consumes: V1 checkpoint 21 and C002 state with immutable hashes.
- Produces: fresh-optimizer V2 formal run using the optimized two-cohort scheduler.

- [ ] Materialize V2 model-only seed from V1 checkpoint 21 and record its parent hash.
- [ ] Copy the C002 PFSP state as provenance and continue logical source update chronology from 21.
- [ ] Use a new stable W&B run ID and retain `rollout/source_policy_update`/`checkpoint/update` chronology.
- [ ] Update both authoritative DESIGN formats with the mixed-focal lane contract and measured performance.
- [ ] Run readiness and one full-update acceptance gate.
- [ ] Launch V2 online only after every semantic and performance gate passes.

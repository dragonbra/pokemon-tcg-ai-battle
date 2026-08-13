# 0043 V3 Generalist Focal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare a new formal 0043 V3 run that inherits V2 checkpoint U207 and randomly trains the focal policy across exact decks 001–067 without changing opponent-policy, PPO, CUDA Engine 2.0, or Frozen evaluation semantics.

**Architecture:** Preserve the V1/V2 training loop and opponent-policy cohort isolation. Add a seeded, frequency-balanced focal scheduler that assigns all 67 decks across every 256-lane rollout, pass it explicitly into the shared runner, and expose V3 through a new immutable version launcher whose parent is V2 U207 with a fresh optimizer and newly collected on-policy data.

**Tech Stack:** Python 3.11, PyTorch FP32 PPO, CUDA Engine 2.0, official engine runtime, W&B online logging, pytest.

## Global Constraints

- V3 parent is exactly `V2_mixed_focal_cohorts/checkpoint/update-000207.pt`; U136 remains evaluation/submission evidence only.
- Focal decks are exact numeric identities `001`–`067`; own archetype is a lane-bound model feature and never a resident cohort key.
- Every 256-game rollout contains each focal deck 3 or 4 times, seeded and shuffled; the 55 decks receiving a fourth lane vary deterministically by update.
- Opponent decks remain 001–067 and opponent policies remain independent complete Policy-0809 / Champion-G1 identities under PFSP 128/64/64.
- PPO hyperparameters, rollout size 256, reference anchor G1/update-0, FP32 optimization, trainable tensors, reward/loss, CUDA Engine 2.0, model-only checkpoint retention and Frozen benchmark composition remain unchanged.
- V3 uses a fresh optimizer, inherited V2 PFSP state, a new W&B run, and never writes into V2.

---

### Task 1: Seeded 67-deck focal schedule

**Files:**
- Modify: `train/0043_champion_league_rl/initial_run.py`
- Modify: `train/0043_champion_league_rl/tests/test_initial_run_acceptance.py`

**Interfaces:**
- Produces: `balanced_focal_schedule(seed: int, deck_ids: tuple[str, ...], lanes: int = 256) -> tuple[FocalLane, ...]`.

- [ ] Write tests proving 256 lanes, IDs 001–067 only, per-deck counts in `{3,4}`, all decks present, deterministic same-seed output, and different-seed ordering/count allocation.
- [ ] Run the focused test and confirm the new symbol is initially absent.
- [ ] Implement quotient/remainder allocation followed by seeded sampling of remainder decks and seeded lane shuffle.
- [ ] Run the focused test and the existing 002/007 schedule regression.

### Task 2: Make the shared PPO runner accept an explicit focal scheduler

**Files:**
- Modify: `train/0043_champion_league_rl/training/run_v1.py`
- Test: `train/0043_champion_league_rl/tests/test_v3_generalist.py`

**Interfaces:**
- Consumes: `balanced_focal_schedule` from Task 1.
- Produces: `_jobs(..., focal_deck_ids=..., focal_seed=...)` and `run(..., focal_deck_ids=...)` while retaining V1/V2 defaults `("002", "007")`.

- [ ] Write a test that builds a V3 job schedule and proves all 67 focal IDs, exact deck bytes, own-archetype IDs and lane IDs agree.
- [ ] Run it and confirm failure before implementation.
- [ ] Parameterize only focal scheduling; preserve opponent league scheduling and policy grouping.
- [ ] Add focal distribution identity to status/config metadata and ensure telemetry continues emitting dynamic `rollout/focal_deck/<ID>/*` metrics.
- [ ] Run focused routing, telemetry and mixed-cohort tests.

### Task 3: Add the immutable V3 launcher and readiness gate

**Files:**
- Create: `train/0043_champion_league_rl/training/run_v3.py`
- Test: `train/0043_champion_league_rl/tests/test_v3_generalist.py`

**Interfaces:**
- Produces: module CLI `python3 -m train.0043_champion_league_rl.training.run_v3` and explicit `--launch-formal` gate.

- [ ] Test exact version `V3_generalist_focal_001_067`, start update 207, parent path/hash, parent PFSP path, new W&B ID, and unused output directories.
- [ ] Implement read-only readiness that validates U207 schema/update/hash, 29×16 own embeddings, frozen 15-way opponent Meta, 67-deck registry and CUDA Engine 2.0 artifacts.
- [ ] Implement formal launch forwarding to the shared runner with `focal_deck_ids=001..067`, fresh optimizer, inherited PFSP and original reference anchor.
- [ ] Run readiness without launching and persist its audit under the V3 artifact directory only at formal launch time; keep pre-launch evidence in `.tmp/evaluation/`.

### Task 4: Distribution, routing and performance acceptance

**Files:**
- Create: `.tmp/evaluation/0043_v3_generalist_readiness/` artifacts (ignored)
- Modify: `experiments/0043_champion_league_rl/DESIGN.md`
- Modify: `experiments/0043_champion_league_rl/DESIGN.html`
- Modify: `experiments/0043_champion_league_rl/DECISIONS.md`

**Interfaces:**
- Consumes: V3 readiness and job materialization from Tasks 1–3.
- Produces: auditable pre-launch evidence and authoritative design state.

- [ ] Audit at least 67 seeded schedules for long-run focal count equality, all-deck per-update coverage, exact deck/hash routing and own-archetype lookup.
- [ ] Run a small mixed 67-focal CUDA Engine 2.0 rollout smoke with both opponent policies, zero routing failure and zero feature D2H; compare throughput against the V2 mixed-cohort boundary.
- [ ] Verify V3 output directories and W&B run ID are unused and no training process is active.
- [ ] Synchronize DESIGN HTML/Markdown and DECISIONS with the V2 U207 parent, fresh optimizer, 67-deck focal semantics and readiness/launch state.
- [ ] Run the 0043 focused suite, `git diff --check`, commit and push without including unrelated session files.

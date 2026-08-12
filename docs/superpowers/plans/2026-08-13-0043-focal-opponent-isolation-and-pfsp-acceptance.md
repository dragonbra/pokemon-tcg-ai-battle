# 0043 Focal/Opponent Isolation and PFSP Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep unpromoted focal weights outside immutable policy assets, pin the first run to Champion-G1 initialization with focal decks 002/007, and prove the 67-deck × 2-policy PFSP routing is identity-safe and fast.

**Architecture:** The immutable asset registry contains only admitted frozen opponents. The mutable focal seed and future updates live under one versioned `rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/` boundary, while a tracked launch contract records its hashes and provenance. CUDA inference loads each effective policy once per run, caches deck-static prototype state separately, groups lanes by exact policy ID, and scatters results back to immutable lane IDs.

**Tech Stack:** Python 3.11, PyTorch CUDA, CUDA Engine 2.0, pytest, JSON manifests.

## Global Constraints

- Opponent policy pool is exactly `Policy-0809` and frozen `Champion-G1` before the first formal run.
- Focal initialization is a mutable copy derived from Champion-G1 and is never a registered Champion or opponent.
- Focal deck pool is exactly `002` and `007`; opponent deck pool is exactly `001` through `067`.
- Policy identity resolves before routing; mutable focal tensors may not alias either frozen opponent.
- Feature tensors remain CUDA-resident with zero feature D2H bytes.
- No formal PPO launch occurs in this work; status remains prepared/blocked until all launch gates pass.

---

### Task 1: Separate mutable focal run state from immutable policy assets

**Files:**
- Modify: `train/0043_champion_league_rl/assets/policies/registry.json`
- Modify: `train/0043_champion_league_rl/assets.py`
- Modify: `train/0043_champion_league_rl/policy_identity.py`
- Modify: `train/0043_champion_league_rl/runtime.py`
- Create: `experiments/0043_champion_league_rl/initial_run_contract.json`
- Create locally: `rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/`

**Interfaces:**
- Produces immutable `active_policy_pool == [Policy-0809, Champion-G1]`.
- Produces tracked focal seed provenance and hashes without registering it as a policy asset.

- [ ] Remove the unpromoted seed from the policy registry and move its checkpoint payload into the V1 run checkpoint/artifact directories.
- [ ] Record `parent_policy_id=Champion-G1`, focal deck IDs 002/007, and model-only semantics in the initial run contract.
- [ ] Make policy asset validation reject non-frozen candidate/run policies.
- [ ] Update tests so policy assets contain exactly the two admitted frozen policies.

### Task 2: Add resident multi-policy isolation and cache gates

**Files:**
- Modify: `train/0043_champion_league_rl/cuda_engine_2/inference.py`
- Create: `train/0043_champion_league_rl/tests/test_cuda_engine_2_policy_pool.py`

**Interfaces:**
- Produces a run-scoped cache keyed by effective policy identity.
- Consumes deck IDs as separate static routing inputs without reloading policy weights per episode.

- [ ] Add a failing test for single-load policy residency, focal/frozen storage non-aliasing, and mutation isolation.
- [ ] Implement the minimal resident policy pool and deck-static cache.
- [ ] Add grouped lane routing checks that reject requested/materialized policy or deck/hash mismatch.
- [ ] Run CUDA tests and record load counts plus device-resident metrics.

### Task 3: Validate the 67-deck × 2-policy PFSP schedule

**Files:**
- Modify: `train/0043_champion_league_rl/preflight.py`
- Create: `train/0043_champion_league_rl/tests/test_initial_run_acceptance.py`

**Interfaces:**
- Produces a deterministic 256-lane acceptance manifest with exact deck/policy/hash bindings.

- [ ] Generate PFSP, uniform, and latest-champion lanes over decks 001–067 and policies Policy-0809/Champion-G1.
- [ ] Assert exact 128/64/64 quotas, both policies, valid deck identities, immutable effective hashes, and no focal-seed admission.
- [ ] Assert focal deck sampling is limited to 002/007 and independent from opponent sampling.
- [ ] Run multiple deterministic curriculum seeds to cover all 134 deck-policy pairs and reject any mismatch.

### Task 4: Performance and documentation closeout

**Files:**
- Modify: `experiments/0043_champion_league_rl/DESIGN.md`
- Modify: `experiments/0043_champion_league_rl/DESIGN.html`
- Modify: `experiments/0043_champion_league_rl/DECISIONS.md`

**Interfaces:**
- Produces a `.tmp/evaluation/0043_initial_run_acceptance/` report.

- [ ] Benchmark warm resident inference separately from one-time initialization.
- [ ] Verify `features_device_resident=true`, `feature_d2h_bytes=0`, and no per-game checkpoint loads.
- [ ] Run the complete 0043 suite and CUDA Engine 2.0 performance audit.
- [ ] Synchronize Markdown/HTML design facts while keeping formal training authorization false.

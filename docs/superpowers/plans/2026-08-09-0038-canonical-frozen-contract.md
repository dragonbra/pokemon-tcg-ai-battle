# 0038 Canonical Frozen Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 0038 rollout sampling preserve the committed 256-slot environment frequency and make update-0/every-5-update evaluation exactly match the 007 `frozen_0806_seeded_2048_v2` contract.

**Architecture:** Evaluation jobs are derived directly from the shared Frozen-0806 contract and the immutable 256-slot catalog, never from a 0038-specific uniform panel. Training rollout constructs independent 256-game frequency units, with deterministic random ordering/seats/seeds, while retaining the existing Agent/action/PPO contract.

**Tech Stack:** Python 3.11, PyTorch PPO, shared `evaluation.frozen_0806_contract`, CUDA resident engine, unittest.

## Global Constraints

- Do not modify `engine/source/`, official ABI, observation schema, or primitive `select` shape.
- Canonical evaluation uses evaluation seed `341512806`, eight 256-game replicas, standard opponent slot weights, and the 007 schedule hash.
- Rollout count must be a multiple of 256 and each consecutive 256-game unit must reproduce exact catalog counts.
- Do not use the GPU or restart formal PPO until the user confirms the GPU is free.
- Preserve the failed V4 update-0 audit; no PPO update was completed.

---

### Task 1: Canonical Frozen evaluation jobs

**Files:**
- Modify: `train/0038_action_boundary_rl/evaluation/frozen_jobs.py`
- Modify: `train/0038_action_boundary_rl/training/run_full_semantic.py`
- Test: `train/0038_action_boundary_rl/tests/test_scaling_and_frozen_panel.py`

**Interfaces:**
- Consumes: `evaluation_game_seed(...)`, immutable `DeckIdentity.games` counts.
- Produces: `build_frozen_jobs(*, focal_deck_id, focal_deck, runtime_root, source_policy_update)` and `canonical_schedule_sha256(...)`.

- [ ] Write a failing test asserting 2,048 jobs, the standard 007 schedule SHA `98b58b...ce9`, 1,024/1,024 seats, exact `games * 8` opponent counts, and no dependency on the uniform panel JSON.
- [ ] Run `python3 -m unittest -v train.0038_action_boundary_rl.tests.test_scaling_and_frozen_panel` and verify the new assertions fail.
- [ ] Build jobs in replica → opponent → slot order with:

```python
engine_seed = evaluation_game_seed(
    focal_identity=focal_deck_id,
    opponent_identity=opponent.deck_id,
    slot=slot,
    replica=replica,
)
focal_first = replica % 2 == 0
```

- [ ] Replace formal update-0/every-5 call sites and schedule validation with the canonical schedule ID/hash.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Frequency-faithful rollout units

**Files:**
- Modify: `train/0038_action_boundary_rl/training/run_full_semantic.py`
- Test: `train/0038_action_boundary_rl/tests/test_project_identity.py`

**Interfaces:**
- Consumes: immutable 256-slot catalog.
- Produces: `build_jobs(...)` where every consecutive block of 256 jobs has exact environment opponent counts.

- [ ] Write a failing test for 2,048 rollout jobs asserting eight independent 256-game units, exact opponent counts per unit, 1,024/1,024 seats, unique engine seeds, and deterministic replay from `(seed, source_policy_update)`.
- [ ] Run the focused test and verify failure under the old paired 512-game scheduler.
- [ ] Implement unit-local shuffle, deterministic balanced seat assignment, and independently sampled engine/search/policy seeds; reject counts not divisible by 256.
- [ ] Re-run the focused test and confirm it passes.

### Task 3: Formal-run failure and audit guards

**Files:**
- Modify: `train/0038_action_boundary_rl/training/run_full_semantic.py`
- Test: `train/0038_action_boundary_rl/tests/test_project_identity.py`

**Interfaces:**
- Consumes: `is_sparse_diagnostic_update` from `training.metric_frequency`.
- Produces: a pre-PPO update-0 resume guard that rejects any real update checkpoint/metric history.

- [ ] Add a failing import/launch-path test that reaches the sparse-diagnostic predicate without a `NameError`.
- [ ] Import the predicate from its defining module and verify the minimal reproduction.
- [ ] Record that the interrupted V4 has only update-0 model weights and preserve its status/config history before a later restart.

### Task 4: Documentation and verification

**Files:**
- Modify: `0038_RL_experiment_setting.md`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.md`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.html`
- Modify: `CUDA_0038_RL_THROUGHPUT_REPORT.md`

**Interfaces:**
- Produces: one documented global evaluation/rollout frequency contract.

- [ ] Document canonical evaluation versus stochastic rollout semantics and remove claims that the uniform `0038_frozen_2048_v1` is the formal panel.
- [ ] Run the complete 0038 tests, shared CUDA adapter tests, `git diff --check`, and a CPU-only schedule audit.
- [ ] Commit code/docs without staging user-owned GPU evaluation outputs or runtime directories.
- [ ] Wait for explicit GPU availability before rerunning update-0 or PPO.

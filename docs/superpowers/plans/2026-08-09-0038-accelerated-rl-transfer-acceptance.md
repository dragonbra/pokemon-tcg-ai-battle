# 0038 Accelerated RL Transfer Acceptance Implementation Plan

> **Execution note:** this plan is implemented sequentially in the primary process. No formal RL run starts until the attested U0 and fixed-snapshot package parity gate pass.

**Goal:** Start a fresh, repaired-schema, CUDA-resident 0038 PPO trajectory from the original Zero-Shot/Pretrain U0 and test whether a controlled higher learning rate learns materially faster than the previous run. The first 50 updates are an observation window, not a stopping limit; the run continues until the user requests a stop.

**Architecture:** Preserve the existing 0038 Action Boundary, hierarchical Phantom Dive allocation, reward, 512-game rollout, and fixed 32-step PPO contract. Add a small acceptance-run controller around the existing trainer for per-group LR scheduling, KL/clip health gates, rollback/retry, CUDA Frozen-2048 champion tracking, and auditable provenance. CPU evaluation is excluded from U0 and the training loop. No CPU evaluation is scheduled automatically; it requires the user to select a checkpoint and authorize the comparison.

**Stack:** Python 3.11, PyTorch PPO, repaired CUDA resident engine, official CPU evaluation runtime, TensorBoard, W&B.

---

### Task 1: Freeze semantic-parity prerequisites

**Files:** existing semantic parity and CUDA diagnostic changes; `SEMANTIC_PARITY_AUDIT.md`.

1. Run the focused parity suite and `git diff --check`.
2. Confirm no `engine/source/` mutation.
3. Commit the audited parity baseline so training manifests reference a clean immutable commit.

### Task 2: Add the accelerated LR and safety contract

**Files:**
- Create: `train/0038_action_boundary_rl/training/accelerated_transfer.py`
- Modify: `train/0038_action_boundary_rl/training/ppo_full_semantic.py`
- Test: `train/0038_action_boundary_rl/tests/test_accelerated_transfer.py`

1. Record actual optimizer group names, base LR, trainable count, and gradient ownership.
2. Apply Actor/Allocation/LoRA multipliers 3x (U1-2), 5x (U3-5), and at most 10x (U6+); keep Win/Prize Value at 2x.
3. Add optimizer-state reset, trainable-state snapshot/restore, and auditable LR changes.
4. Classify KL/clip/NaN/parity health; retry a rolled-back update at a lower Actor multiplier where allowed.

### Task 3: Add acceptance-run orchestration

**Files:**
- Modify: `train/0038_action_boundary_rl/training/run_full_semantic.py`
- Test: existing formal-run tests plus new controller tests.

1. Allocate fresh `V11_accelerated_transfer_acceptance` paths.
2. Force an unbounded/manual-stop run, `PRIZE` preset, CUDA resident backend, 512 rollout games, 512 trajectory games, and 32 optimizer steps.
3. Build strict U0 from the common Zero-Shot/Pretrain update-0 checkpoint with zero-delta LoRA and no RL checkpoint/state.
4. Require U0 model-load report and CUDA/package fixed-snapshot parity before rollout.
5. Run CUDA Frozen-2048 at U0 and every fifth update; persist all per-game results.
6. Fail closed on invalid macro, fallback, unsupported, replay parity, NaN/Inf, or representation drift.
7. Save every model-only checkpoint, current CUDA champion, degradation warnings, status, and summary.
8. Do not run U0 official CPU evaluation.

### Task 4: Documentation and provenance

**Files:**
- Modify: `experiments/0038_action_boundary_rl/DESIGN.md`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.html`
- Create: `ACCELERATED_RL_TRANSFER_VALIDATION.md`

1. Document the acceptance phase, LR groups/schedule, safety gates, and CUDA-only selection contract.
2. Record initialization/checkpoint/runtime/feature/action/reward/git hashes.
3. State explicitly that CPU U0-2048 was omitted by user direction and that no CPU evaluation is scheduled without a later explicit request.

### Task 5: Verify, launch, and monitor

1. Run unit/property/parity tests and a tiny isolated PPO smoke.
2. Run U0 CUDA/package snapshot parity; stop if it fails.
3. Run CUDA Frozen-2048 U0.
4. Launch without an update limit with W&B online and monitor update-boundary health. Treat U1-U50 as the first acceptance observation window.
5. Stop automatically only on an unrecoverable safety failure, or cleanly at an update boundary after `STOP_REQUESTED`.

### Task 6: User-directed post-training transfer check

1. Continue recording CUDA candidates; do not autonomously declare the run finished at U50.
2. Wait for the user to select one or more updates and explicitly request CPU/GPU transfer analysis.
3. Only then strictly export packages and schedule the requested official-CPU comparison.
4. Report any later CPU result separately from the CUDA U0 paired deltas; do not claim a CPU U0 paired effect without a CPU U0 run.
5. Keep `ACCELERATED_RL_TRANSFER_VALIDATION.md` current during training. Do not submit to Kaggle or begin any follow-on model experiment.

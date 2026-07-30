# 0022 Worker Scaling Implementation Plan

> **For agentic workers:** Execute this plan inline task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the valid V1 update-1 asset, make formal PPO stoppable at update boundaries, and select a higher-throughput V2 worker/coalescing configuration from official-engine evidence.

**Architecture:** PPO semantics, deck catalog, Foundation, reward, and update size stay unchanged. Only rollout supply concurrency and request coalescing are treated as a new versioned experiment variable; benchmarks use the same `LeagueRolloutCollector` and official engine as training.

**Tech Stack:** Python 3.11, PyTorch/CUDA, multiprocessing official-engine workers, unittest, W&B, TensorBoard.

## Global Constraints

- Never modify `engine/source/`.
- Preserve V1 metrics and `update-000001.pt`; do not append a changed concurrency contract to V1.
- Keep 0021 Epoch 15 and its dataset; the dataset is only a future low-disk cleanup candidate.
- Keep model-only bounded checkpoints and the existing 100/80 GiB SSD guards.
- Allocate a new `V<n>_<tag>` and W&B run for changed rollout concurrency.

---

### Task 1: Auditable Stop Contract

**Files:**
- Modify: `train/0022_league_training/training/run.py`
- Test: `tests/test_0022_league_training.py`
- Create: `experiments/0022_league_training/decisions/003_v1_throughput_and_v2_scaling.md`

- [x] Add a process-local stop flag set by SIGINT/SIGTERM and checked before starting each update.
- [x] Finish an in-flight update before a requested graceful stop, then write `completed_stop_requested` with the last complete checkpoint.
- [x] Preserve exception-to-`failed` behavior for genuine errors.
- [x] Record the measured V1 update-1 facts and why V1 stopped.
- [x] Run the focused tests.

### Task 2: Versioned Coalescing Configuration

**Files:**
- Modify: `train/0022_league_training/cli.py`
- Modify: `train/0022_league_training/training/run.py`
- Test: `tests/test_0022_league_training.py`

- [x] Add `--coalesce-ms` and persist its exact value in `training_config.json`.
- [x] Validate that the value is non-negative.
- [x] Add unit coverage for CLI/config propagation.
- [x] Run the focused tests.

### Task 3: Official-Engine Scaling Gate

**Files:**
- Create temporary results under `.tmp/evaluation/0022_worker_scaling/`
- Update: `experiments/0022_league_training/decisions/003_v1_throughput_and_v2_scaling.md`
- Update: `experiments/0022_league_training/DESIGN.md`
- Update: `experiments/0022_league_training/DESIGN.html`

- [x] Run equal-size collector benchmarks with 64, 128, and 256 workers at 5 ms coalescing.
- [x] Require 100% finished, zero errors, record games/s, requests/batches, mean/max batch, RAM, GPU peak, and disk.
- [x] Choose the smallest configuration within 10% of the best throughput to control RAM.
- [x] Synchronize both authoritative design documents with the selected formal contract.

### Task 4: Immutable V2 Launch

**Files:**
- Create: `rl_runs/0022_league_training/versions/V2_<tag>/` via the project initializer
- Update tracked lightweight status/config/metrics only as the run proceeds

- [ ] Run all 0022 tests and repository contract tests.
- [ ] Initialize V2 and verify all 48 update-0 decoder/value assets and Foundation hashes.
- [ ] Launch the 20-hour formal run with 512 games/update and the selected concurrency.
- [ ] Guard the update-0 Frozen gate and first PPO update through checkpoint creation and W&B sync.
- [ ] Commit and push code, design, decision, and lightweight V1 artifacts; never commit checkpoints, W&B staging, or TensorBoard events.

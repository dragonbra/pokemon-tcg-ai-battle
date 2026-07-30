# 0022 League Training Implementation Plan

> **For agentic workers:** Execute this plan task-by-task. Each task must pass its listed gate before the next task starts.

**Goal:** Train `dragapult_ex_001` with decoder/value-only PPO against the 48-deck League on the official engine, preserve all League assets, and run for approximately 20 guarded GPU hours without unbounded SSD growth.

**Architecture:** The immutable 0019 epoch-13 foundation supplies the frozen encoder and initial decoder. A self-contained 0022 runtime collects official-engine trajectories while routing Frozen and Live opponent decisions through resident GPU inference. The first formal run updates only the focal decoder/value; Live-opponent trajectory updates remain an explicit later stage so the focal experiment has one interpretable variable.

**Tech Stack:** Python 3.11, PyTorch/CUDA, official `cg` engine runtime, PPO/GAE, TensorBoard, W&B online, atomic model-only checkpoints.

## Global Constraints

- Do not modify `engine/source/`.
- Do not import executable code from another numbered `train/` project at runtime.
- Foundation identity is 0019 epoch 13, SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.
- Actor-visible source identity is neutral `source_id=0`; deck identity comes only from exact deck features.
- Encoder parameters remain frozen. Only focal decoder and value parameters update in the first formal run.
- Checkpoints are atomic model-only decoder/value payloads. No optimizer, RNG, rollout buffer, replay, or full foundation copy is saved.
- Canonical metrics are flushed to `training_metrics.jsonl`, then TensorBoard, then W&B `dragon_bra/pokemon-tcg-policy-learning`.
- Rollout diagnostics must record `rollout/source_policy_update`; checkpoints record their resulting `checkpoint/update`.
- Fixed Frozen greedy evaluation is separate from sampled rollout reward.
- SSD guard fails closed before launch below the configured free-space threshold and stops cleanly at the runtime low-water mark.

---

### Task 1: Freeze the self-contained PPO runtime boundary

**Files:**
- Create: `train/0022_league_training/policy/`
- Create: `train/0022_league_training/rollout/`
- Create: `train/0022_league_training/training/`
- Create: `train/0022_league_training/tests/test_policy.py`
- Create: `train/0022_league_training/tests/test_rollout.py`
- Create: `train/0022_league_training/tests/test_training.py`

**Interfaces:**
- Produces `LeagueActorCritic`, `ActionSample`, `TrajectoryDecision`, `EpisodeTrajectory`, and `PPOTrainer`.
- All copied source is physically owned by 0022 and imports only 0022 or shared infrastructure.

- [ ] Add import-boundary tests rejecting `train.0017`, `train.0020`, or other numbered runtime imports.
- [ ] Copy the stable 0020 action distribution, online encoder, rollout record, and PPO math into focused 0022 modules.
- [ ] Replace the 0020 checkpoint loader with `load_foundation()` plus 0022 decoder/value composition.
- [ ] Verify encoder gradients remain absent while decoder/value gradients are finite.

### Task 2: Build official-engine League rollout collection

**Files:**
- Create: `train/0022_league_training/rollout/collector.py`
- Create: `train/0022_league_training/rollout/worker.py`
- Create: `train/0022_league_training/opponents.py`
- Modify: `train/0022_league_training/tests/test_rollout.py`

**Interfaces:**
- Consumes the immutable deck catalog and focal checkpoint update.
- Produces completed `EpisodeTrajectory` objects tagged with focal deck, opponent deck, Frozen/Live view, seat, seed, and source policy update.

- [ ] Write tests for balanced seats, deterministic opponent scheduling, terminal rewards, and policy-version tagging.
- [ ] Run every game in an isolated official-engine worker process.
- [ ] Route Frozen views to the foundation decoder and Live views to their named decoder checkpoint.
- [ ] Keep trajectory tensors in memory only until their PPO update completes.
- [ ] Smoke 2 opponents x 2 games with 4/4 finished and zero errors.

### Task 3: Implement focal decoder/value PPO

**Files:**
- Create: `train/0022_league_training/training/ppo.py`
- Create: `train/0022_league_training/training/advantages.py`
- Modify: `train/0022_league_training/tests/test_training.py`

**Interfaces:**
- Consumes on-policy focal trajectories from exactly one `source_policy_update`.
- Produces decoder/value update metrics and a new atomic decoder-only checkpoint.

- [ ] Test GAE/terminal return math for win `+1`, loss `-1`, draw `0`, gamma `1`.
- [ ] Test clipped policy loss, clipped value loss, entropy, KL-to-foundation, gradient clipping, and minibatch iteration.
- [ ] Reject mixed policy updates in one PPO batch.
- [ ] Assert every trainable parameter belongs to decoder or value head.
- [ ] Prove save-reload recomposition gives greedy-identical actions.

### Task 4: Add formal run logging and SSD guard

**Files:**
- Create: `train/0022_league_training/training/run.py`
- Create: `train/0022_league_training/storage.py`
- Create: `train/0022_league_training/tests/test_storage.py`
- Modify: `train/0022_league_training/cli.py`

**Interfaces:**
- Produces a new immutable `rl_runs/0022_league_training/versions/V<n>_<tag>/` with canonical metrics, TensorBoard, W&B metadata, and bounded checkpoints.

- [ ] Fail before allocation if any target version path is occupied.
- [ ] Fail before launch below 100 GiB free; stop cleanly below 80 GiB free.
- [ ] Retain latest 2, best Frozen-eval 2, and explicit branch checkpoints; never delete a referenced best checkpoint.
- [ ] Record per-update free bytes, run-directory bytes, CUDA allocated/reserved/peak bytes, rollout throughput, and policy provenance.
- [ ] Verify W&B initialization without logging secrets or uploading checkpoints.

### Task 5: Synchronize the authoritative design

**Files:**
- Modify: `experiments/0022_league_training/DESIGN.md`
- Modify: `experiments/0022_league_training/DESIGN.html`
- Modify: `experiments/0022_league_training/manifest.json`

- [ ] Document exact tensors, trainable parameter names/counts, PPO losses, reward timing, rollout provenance, checkpoint retention, and focal-only first-run boundary.
- [ ] Mark Live-opponent gradient updates as a later version, not a hidden part of the first formal run.
- [ ] Cross-check all paths and hashes against current code and catalog.

### Task 6: Run gates and launch

**Files:**
- Runtime only: `rl_runs/0022_league_training/versions/V<n>_<tag>/`
- Formal evaluation: `experiments/0022_league_training/evaluation/V<n>_<tag>.html`

- [ ] Run unit and import-boundary tests.
- [ ] Measure resident VRAM for foundation plus all Live decoder/value assets.
- [ ] Run official-engine rollout smoke.
- [ ] Run a short PPO canary, confirm finite losses, nonzero decoder delta, unchanged encoder hash, reload parity, bounded disk, and zero worker errors.
- [ ] Run balanced Frozen greedy evaluation before considering the canary healthy.
- [ ] Commit and push the implementation after all preflight gates pass.
- [ ] Start the approximately 20-hour formal run in a persistent monitored process.
- [ ] Poll metrics, worker health, GPU utilization, W&B sync, disk free space, and checkpoint retention; repair only with a new version when training semantics change.

# 0044 V23 Final Entropy Fine-Tune Implementation Plan

> **For agentic workers:** Execute this plan inline in the current session. Subagent execution is not enabled for this task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop V22 at its latest durable checkpoint and launch a new V23 final fine-tune with entropy coefficient reduced from 0.003 to 0.0015.

**Architecture:** Send SIGINT only to the exact V22 process, preserve its latest completed model-only checkpoint, and explicitly record any partial next update as discarded. Add a self-contained V23 runner that strict-pins that checkpoint, starts a fresh optimizer and fresh on-policy collection, and changes only the entropy coefficient while retaining V22's 512-game rollout, six 3.0x Meta weights, half-standard learning rates, Champion-G3 opponent identity, and memory controls.

**Tech Stack:** Python 3, PyTorch PPO, pytest, CUDA Engine 2.0, TensorBoard, W&B online.

## Global Constraints

- Do not modify `engine/source/`.
- V22 remains immutable after stop; no metrics, checkpoints, or W&B curves are appended manually.
- Resolve and hash the actual latest durable V22 checkpoint after the process exits; V23 must hard-pin that exact identity.
- V23 starts at local U0 with a fresh optimizer and fresh on-policy rollout data.
- Change only `entropy_coefficient: 0.003 -> 0.0015`; keep half-standard LR and all other PPO semantics unchanged.
- Keep 512 games/update and Meta weights `{0:3,1:3,2:3,3:3,5:3,27:3}`.
- Keep the independently resolved complete immutable Champion-G3 opponent, PFSP disabled, and periodic evaluation disabled.
- Keep model-only checkpoint retention set to all and launch W&B online in `dragon_bra/pokemon-tcg-policy-learning`.

---

### Task 1: Stop and seal V22

**Files:**
- Modify: `rl_runs/0044_g2_dragapult_policy_option_lora/versions/V22_v21_u1_rollout512_meta6_x3/artifact/status.json`

**Interfaces:**
- Consumes: exact live PID selected by command line `training.run_v22` and its existing canonical status/checkpoints.
- Produces: an exited V22 process, synchronized W&B run, latest durable checkpoint identity, and explicit partial-update disposition.

- [x] Send SIGINT to the exact V22 PID and wait for process exit without SIGKILL.
- [x] Verify no checkpoint newer than the last matching canonical metrics row exists.
- [x] Record `state=stopped`, the user-requested reason, last durable update, discarded partial update, and W&B synchronization result atomically.

### Task 2: Implement and test V23 entropy-only continuation

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/training/run_v23.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/tests/test_v23_final_entropy_finetune.py`

**Interfaces:**
- Consumes: exact sealed V22 checkpoint, V22 rollout/sampling/memory constants, and `PPOConfig` via a version-specific learning configuration.
- Produces: `run_v23.readiness()` and `run_v23.launch()` with `entropy_coefficient=0.0015` and every other requested semantic pinned.

- [x] Write tests for parent hash/metadata, fresh optimizer/data, entropy-only delta, 512 games, six Meta weights, half-standard LR, Champion-G3, and V22 memory controls.
- [x] Run the focused test and confirm it fails before `run_v23.py` exists.
- [x] Implement the minimal V23 runner and entropy override plumbing without changing project defaults.
- [x] Run focused and adjacent identity/V22 tests and confirm they pass.

### Task 3: Synchronize design and launch V23

**Files:**
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md`
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html`
- Create at runtime: `rl_runs/0044_g2_dragapult_policy_option_lora/versions/V23_*/` standard artifact/checkpoint/TensorBoard/W&B paths.

**Interfaces:**
- Consumes: tested V23 runner and sealed V22 identity.
- Produces: synchronized authoritative design records and a running formal V23 online training process.

- [x] Document V22 stop boundary and V23 entropy-only final fine-tune in both authoritative design files.
- [x] Run readiness, CUDA, policy-identity, path-unused, and W&B import gates.
- [x] Launch V23 online and verify U0/U1 checkpoints, config, running status, PID, GPU residency, and W&B URL.

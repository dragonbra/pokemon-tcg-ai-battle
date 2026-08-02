# 0025 V3 Training Restart Implementation Plan

> **For agentic workers:** Execute this plan inline in the current foreground session; do not delegate the training monitor.

**Goal:** Preserve the interrupted V2 evidence and run a fresh, fully monitored V3 James Cox Raging Bolt three-arm BC experiment without loading V2 state.

**Architecture:** Keep the frozen paired dataset, split, three model arms, optimizer family, and validation contract unchanged. Allocate a new immutable version, construct all models and optimizers from the configured seed, and keep the repository watchdog attached in the foreground until training reaches a terminal state.

**Tech Stack:** Python 3.11, PyTorch CUDA/bfloat16, W&B online, TensorBoard, repository `TrainingLogger`, JSONL canonical metrics.

## Global Constraints

- Do not modify `engine/source/`.
- Do not load, overwrite, delete, or append to V2 checkpoints or metrics.
- Source/team provenance must remain absent from actor-visible inputs.
- Each active arm gets exactly one training update pass and one complete validation pass per epoch.
- Checkpoints remain model-only with finite retention.
- Offline imitation metrics are not gameplay-strength evidence.

---

### Task 1: Seal V2

**Files:**
- Modify: `rl_runs/0025_semantic_foundation_pretraining/versions/V2_james_cox_raging_bolt_ablation/artifact/status.json`
- Create: `experiments/0025_semantic_foundation_pretraining/decisions/2026-08-02-v2-host-reboot-v3-restart.md`

- [x] Record `state=interrupted`, `last_complete_epoch=2`, the host-reboot reason, and the no-resume decision.
- [x] Preserve the original V2 metrics, W&B identity, and model-only checkpoints unchanged.
- [x] Record the reboot-created NUL-byte anomaly in the V2 metrics file as an audit warning.

### Task 2: Preflight The Fresh Version

**Files:**
- Verify: `train/0025_semantic_foundation_pretraining/`
- Verify: `rl_runs/0025_semantic_foundation_pretraining/dataset/V1_james_cox_raging_bolt_semantic/`

- [x] Confirm `V3_james_cox_raging_bolt_restart` and its evaluation report path are unused.
- [x] Run the 0025 tests and validate dataset split/hash invariants.
- [x] Confirm CUDA, RAM, swap, disk, and W&B online prerequisites.

### Task 3: Start And Monitor V3

**Files:**
- Create at runtime: `rl_runs/0025_semantic_foundation_pretraining/versions/V3_james_cox_raging_bolt_restart/`

- [x] Start `monitor_training` with 40 epochs, batch size 256, validation batch size 256, patience 6, and 30-second heartbeats.
- [x] Verify the training config records seed `20260802` and contains no resume/checkpoint input.
- [x] Keep the watchdog in the foreground and inspect every heartbeat for process, status, canonical metrics, checkpoints, W&B, GPU, RAM, swap, disk, OOM, timeout, worker EOF, and engine errors.
- [x] Continue until `TRAINING_MONITOR_COMPLETE` or immediately diagnose any alert.

### Task 4: Synchronize Authority And Report

**Files:**
- Modify: `experiments/0025_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0025_semantic_foundation_pretraining/DESIGN.html`

- [x] Record V2 as interrupted after two complete epochs and V3 as a from-scratch rerun.
- [x] Record the V3 W&B run identity and final training state.
- [x] Summarize per-arm best validation checkpoints while preserving the official-engine evaluation boundary.

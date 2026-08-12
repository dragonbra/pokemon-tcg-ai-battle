# 0042 V8 002 Alakazam Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start an unbounded formal V8 PPO branch from V7 update 270 using exact focal deck 002 (`Alakazam / Dudunsparce`) while preserving every other V7 training and Frozen-0809 setting.

**Architecture:** Strict-load the V7 U270 model-only checkpoint into the existing 0042 full model after the trainer freezes the unchanged Policy-0809/U0 reference policy. Create a fresh optimizer, RNG stream, on-policy rollout stream, W&B run, and version directory; only the exact focal deck and its runtime-derived own-archetype class change.

**Tech Stack:** Python 3, PyTorch PPO Protocol V2, resident official CUDA engine, W&B online, unittest.

## Global Constraints

- Version is `V8_alakazam_dudunsparce_002`; no earlier version is overwritten or appended.
- Source is exactly V7 `checkpoint/update-000270.pt`, SHA-256 `0ce3b0920d4fd09674d06270c100c500e7bea627ee58933f0ecb02d69a5a5198`.
- Focal deck is catalog 002, ID `alakazam_dudunsparce_3f4515092dc5`, exact-deck SHA-256 `3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf`, own-archetype class 3 (`alakazam`).
- Training remains 256 games/update, 256 trajectories, 256 CUDA lanes, PPO minibatch 2048, forward microbatch 1024, 3 epochs, decoder/allocation LR `2e-5`, Policy Adapter LR `4e-5`, Value groups LR `1e-4`, and Frozen evaluation every 10 updates.
- Reference KL remains anchored to the immutable original Policy-0809/U0 snapshot created before branch checkpoint load; only the trainable focal starts at V7 U270.
- Opponents remain independent full immutable `Policy-0809` over the same neutral 55-deck schedule and per-lane exact-deck routing contract.
- Periodic evaluation remains identity-bound Frozen-0809 CUDA-2048 under `kaggle_fp16_storage_fp32_runtime_v1`.
- V8 has no update limit and stops only through `artifact/STOP_REQUESTED` after a complete update.
- All model-only checkpoints are retained. No optimizer, RNG, rollout, or replay state is imported from V7.
- Official `engine/source/` remains read-only. Promotion remains human-only.

---

### Task 1: Bind exact focal deck 002

**Files:**
- Create: `train/0042_full_model_design/focal_decks/alakazam_dudunsparce_002/deck.csv`
- Create: `train/0042_full_model_design/focal_decks/alakazam_dudunsparce_002/manifest.json`
- Modify: `train/0042_full_model_design/tests/test_project_identity.py`

**Interfaces:**
- Consumes: catalog-002 immutable 60-card payload and `RunConfig` custom focal interface.
- Produces: exact-deck/hash/routing and own-archetype class-3 regression proof.

- [ ] Copy the exact catalog payload and record immutable provenance and physical-file hash.
- [ ] Add a V8 config test that validates the exact deck, focal jobs, and class 3 `alakazam` conditioning.
- [ ] Run the project identity test and require PASS.

### Task 2: Synchronize the authoritative design

**Files:**
- Modify: `experiments/0042_full_model_design/DESIGN.md`
- Modify: `experiments/0042_full_model_design/DESIGN.html`

**Interfaces:**
- Consumes: verified V7 U270 lineage and V8 focal identity.
- Produces: current model/training-stage documentation matching runtime truth.

- [ ] Record V8 exact deck, hash, class 3, and V7 U270 source identity.
- [ ] Record fresh optimizer/on-policy state, unchanged original Policy-0809 reference-KL anchor, unchanged hyperparameters, opponent pool, evaluation cadence, and unbounded duration.
- [ ] Preserve V7 results as provenance rather than relabeling them as V8 evidence.

### Task 3: Pass formal launch gates and start V8

**Files:**
- Create through runner: `rl_runs/0042_full_model_design/versions/V8_alakazam_dudunsparce_002/{artifact,checkpoint,tensorboard,wandb}`
- Create through monitor: `.tmp/training_monitor/0042_full_model_design/V8_alakazam_dudunsparce_002/`

**Interfaces:**
- Consumes: V7 U270 model-only checkpoint, exact deck 002, full Policy-0809 resolver, existing formal runner and watchdog.
- Produces: healthy, unbounded V8 formal PPO run.

- [ ] Require unused V8 paths, matching U270 sidecar, available CUDA artifacts/GPU/disk, and W&B online authentication.
- [ ] Run focused custom-deck, initialization, candidate-deployment, Policy-0809, routing, and monitor tests.
- [ ] Launch via `monitor_training` with V7's explicit hyperparameters, exact deck-002 flags, `--launch-formal`, and no `--updates` argument.
- [ ] Require update-0 checkpoint/evaluation and update 1 to complete with zero identity/routing/fallback errors, finite PPO metrics, W&B online recording, and all-checkpoint retention.
- [ ] Leave trainer and watchdog running and report PIDs, log, W&B URL, current update, and next Frozen milestone.

### Task 4: Preserve failed V8 and launch V9 with an explicit FP16 numeric waiver

**Files:**
- Preserve: `rl_runs/0042_full_model_design/versions/V8_alakazam_dudunsparce_002/`
- Modify: `train/0042_full_model_design/training/run_full_semantic.py`
- Modify: `train/0042_full_model_design/tests/test_0040_long_run_contract.py`
- Create through runner: `rl_runs/0042_full_model_design/versions/V9_alakazam_dudunsparce_002_fp16_drift_allowed/`

**Interfaces:**
- Consumes: V8 U0 failure showing only training-FP32 versus FP16-storage deployment Value drift above the 0.003 diagnostic threshold.
- Produces: a per-run waiver that leaves CPU/CUDA semantic parity, deployment identity, opponent identity, official-engine health, and routing gates strict.

- [ ] Keep V8 as an immutable failed formal attempt; do not resume or overwrite it.
- [ ] Add a default-false `allow_fp16_deployment_numeric_drift` config/CLI field and regression test its exact scope.
- [ ] Persist the raw FP16 comparison and an explicit waiver record instead of relabeling numeric parity as PASS.
- [ ] Launch V9 from the same V7 U270 model-only checkpoint and exact deck 002 with all other parameters unchanged.

# 0023 Limitless Focal 24h RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue project 0023 with the Limitless Mega Lopunny ex deck (including Abra) as the focal policy, inherit every deck decoder from V6 update 13, and supervise a foreground RL run for at least 24 GPU hours.

**Architecture:** Change only the project focal-deck identity while preserving the existing 50-deck League catalog and isolated decoder/value heads. Initialize a new immutable V7 version from every matching V6 live checkpoint, then run the existing League PPO runner with a foreground watchdog whose structured stdout is observed by the main agent.

**Tech Stack:** Python 3.11, PyTorch/CUDA, official engine runtime, project 0023 League PPO runner, JSONL/TensorBoard/W&B logging, `monitor_training.py`.

## Global Constraints

- Never modify `engine/source/`; official engine runtime remains the only gameplay authority.
- Do not overwrite V1–V6; allocate a strictly newer `V7_<tag>` version with isolated artifact, checkpoint, TensorBoard, and W&B paths.
- Use the exact repository deck `mega_lopunny_ex_001` and record its 60-card hash and Limitless provenance.
- Inherit all 50 decoder/value model-only checkpoints from V6 `update-000013.pt`; do not initialize any deck from Foundation.
- Keep W&B online logging in private project `dragon_bra/pokemon-tcg-policy-learning`.
- Launch training and watchdog in one long-lived foreground terminal session; retain and repeatedly wait on that session, handling alerts in the main agent.
- Do not package or run unrelated GPU work while this training session is active.

### Task 1: Switch the focal deck and record the new experiment contract

**Files:**
- Modify: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/__init__.py`
- Test: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/tests/test_league.py`
- Modify: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/manifest.json`
- Create: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/decisions/002_limitless_focal_24h_inheritance.md`

**Interfaces:** `FOCAL_DECK_ID` becomes `mega_lopunny_ex_001`; deck loading must expose exactly one focal Live plugin and keep the 50-deck catalog unchanged.

- [ ] **Step 1: Write the failing focal-identity assertion**
- [ ] **Step 2: Run the focused test and confirm the old focal ID fails**
- [ ] **Step 3: Change `FOCAL_DECK_ID` and update the experiment manifest/decision**
- [ ] **Step 4: Run the focused catalog tests and validate that `mega_lopunny_ex_001` is exact 60 cards with Abra**

### Task 2: Preflight and materialize V7 from V6

**Files:**
- Create at runtime: `rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/V7_from_v6_limitless_focal_24h/`
- Verify: `rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/V6_from_v2_update3_20h_gpu_resume/checkpoint/live/*/update-000013.pt`

**Interfaces:** `resolve_project_version_checkpoint_map(..., source_version="V6_from_v2_update3_20h_gpu_resume")` must resolve 50 matching deck checkpoints and `initialize` must write `inherited_0023_version` for all 50 decks with no Foundation count.

- [ ] **Step 1: Run `validate-decks` and verify the focal deck/hash/card count**
- [ ] **Step 2: Run the source-version checkpoint audit for all 50 deck IDs and assert update 13**
- [ ] **Step 3: Initialize V7 and audit its catalog, initialization audit, and model-only checkpoint fields**
- [ ] **Step 4: Run the project unit tests that cover initialization and decoder loading**

### Task 3: Launch and supervise the 24-hour foreground League run

**Files:**
- Use: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/monitor_training.py`
- Write runtime evidence: `rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/V7_from_v6_limitless_focal_24h/`

**Interfaces:** Training emits canonical metrics/status/checkpoints; the watchdog emits flushed `TRAINING_MONITOR_HEARTBEAT`, `TRAINING_MONITOR_ALERT`, or `TRAINING_MONITOR_COMPLETE` records and exits nonzero on anomalies.

- [ ] **Step 1: Start the League runner with V7, CUDA, the existing worker contract, and no packaging/evaluation contention**
- [ ] **Step 2: Attach the watchdog to the same foreground exec session with `--min-hours 24`**
- [ ] **Step 3: Retain the exec session ID and poll it at intervals below 60 seconds**
- [ ] **Step 4: On any alert or nonzero exit, inspect status/metrics/checkpoints/GPU/error evidence and handle the run before reporting**
- [ ] **Step 5: After at least 24 hours, record the final status and checkpoint/evaluation evidence without claiming strength from sampled rollout metrics**

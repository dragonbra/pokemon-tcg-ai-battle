# 0044 Standard LR And V19 Half-LR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Name and enforce the 0044 standard learning-rate profile, select the best late V18/EMA checkpoint by common-seed Benchmark V2, then continue exact deck 069 in one run-scoped experiment using exactly one-half of every standard learning rate and 1,024 rollout games per update.

**Architecture:** A project-local LR profile module is the single source of truth for all seven optimizer-group rates. Generic 0044 configuration defaults to the standard profile independently of deck identity; V19 explicitly selects a `0.5× standard` run override and records both the base profile and scale in readiness, checkpoint provenance, local config, metrics, and W&B config.

**Tech Stack:** Python 3.11, dataclasses, PyTorch PPO, CUDA Engine 2.0, pytest, W&B online.

## Global Constraints

- `0044_standard_lr_v1` is the project default for every deck and every new training version unless that version explicitly overrides it.
- Standard rates are decoder `5e-6`, policy strategy adapter `5e-6`, allocation `5e-6`, Option LoRA `1e-5`, Meta Actor Residual `5e-6`, Value `2e-5`, and Prize `2e-5`.
- V19 alone uses run-scoped scale `0.5`: `2.5e-6`, `2.5e-6`, `2.5e-6`, `5e-6`, `2.5e-6`, `1e-5`, and `1e-5` respectively.
- The half-rate override is not attached to deck 069, its registry entry, own-archetype mapping, checkpoint schema, or future default.
- V18 stops at durable U17; partial U18 is discarded. V19 starts at local U0 from the best same-contract candidate among U14/U15/U16/U17 and the declared U14–U17 EMA, with a fresh optimizer and fresh on-policy data.
- V19 alone uses 1,024 rollout games per update; this is a run-level variance-reduction variable and is not attached to deck 069 or the project default.
- PPO epochs remain 3, logical minibatch remains 4,096, and physical forward microbatch remains 768; the larger rollout is not offset by enlarging the minibatch.
- Focal remains exact deck 069; opponent policy remains immutable Champion-G3; opponent decks remain Meta-first balanced over live pool 001–069; PFSP and evaluate remain disabled.
- No official engine source file may be modified.

---

### Task 1: Project-local standard LR source of truth

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/training/lr_profiles.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/run_v1.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_lr_profiles.py`

**Interfaces:**
- Consumes: seven `PPOConfig` optimizer-group learning-rate fields.
- Produces: immutable `LearningRateProfile`, `STANDARD_LR_PROFILE`, `scaled_standard_profile(scale, override_id)`, and default `_config()` behavior.

- [ ] **Step 1: Write failing tests for exact standard and half profiles**

Assert all seven exact values, `scaled_standard_profile(0.5)`, rejection of non-positive scale, and that `_config()` with no LR arguments equals the standard profile.

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_lr_profiles.py`
Expected: FAIL because `lr_profiles.py` does not exist.

- [ ] **Step 3: Implement the immutable profiles and default wiring**

Create the dataclass/profile helpers. Change generic `_config()` and `run()` scalar defaults to the standard values, and record a normalized LR profile payload in `training_config.json`; do not inspect `focal_deck_id` when choosing rates.

- [ ] **Step 4: Run profile and PPO configuration tests**

Run: `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_lr_profiles.py train/0044_g2_dragapult_policy_option_lora/tests/test_v12_g3_to_g4.py`
Expected: PASS.

### Task 2: Late-checkpoint and EMA Benchmark V2 selection

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/evaluation/build_v18_u14_u17_ema.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/evaluation/benchmark_v2_v18_late_checkpoint_selection.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/evaluation/render_v18_late_checkpoint_selection.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_v18_late_checkpoint_ema.py`

**Interfaces:**
- Consumes: V18 U14/U15/U16/U17 model-only checkpoints and existing deck-069 U0/G3 reports.
- Produces: an audited FP32 EMA derived candidate and a common-seed CUDA-2048 ranking over U14/U15/U16/U17/EMA.

- [ ] **Step 1: Write failing EMA identity and coefficient tests**

Assert exact source hashes, identical keys/shapes/schema, normalized decay-0.5 weights `1/15, 2/15, 4/15, 8/15`, FP32 averaging for floating tensors, U17 values for non-floating tensors, and derived-candidate metadata that cannot be mistaken for a true update.

- [ ] **Step 2: Implement and materialize the audited EMA artifact**

Write the model-only derived checkpoint atomically under V18 evaluation artifacts and record source/update/hash/weight provenance beside it.

- [ ] **Step 3: Run common-seed Benchmark V2 CUDA-2048**

Evaluate focal 069 for U14, U15, U16, U17, and EMA against Policy-0809 using the identical frozen Benchmark V2 schedule.

- [ ] **Step 4: Render and select**

Publish an HTML/JSON comparison with W/L/D, overall and seat splits, per-Meta/per-deck changes, confidence intervals, and a deterministic best-candidate rule based on highest 2,048-game win rate with no automatic champion promotion.

### Task 3: V19 run-scoped half-rate continuation

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/training/run_v19.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_v19_half_lr_restart.py`

**Interfaces:**
- Consumes: exact V18 U17 checkpoint and `scaled_standard_profile(0.5)`.
- Produces: V19 readiness/launch contract with local U0, fresh optimizer, focal 069, frozen Champion-G3, and run-scoped half LR.

- [ ] **Step 1: Write the failing V19 identity and LR test**

Assert exact parent version/update/hash, local U0, all seven half rates, base profile `0044_standard_lr_v1`, scale `0.5`, scope `run_only`, default-after-run `0044_standard_lr_v1`, 1,024 rollout games, 3 PPO epochs, logical minibatch 4,096, physical microbatch 768, and unchanged opponent/focal sampling.

- [ ] **Step 2: Run the test and verify the wrapper is missing**

Run: `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_v19_half_lr_restart.py`
Expected: FAIL because `training.run_v19` does not exist.

- [ ] **Step 3: Implement readiness and launch**

Hard-check V18 U17 identity and the seven rates, reject used version paths, set a new stable W&B run ID, and pass only the V19 run-level half-rate values plus 1,024 rollout games into `run()`.

- [ ] **Step 4: Run readiness and focused tests**

Run: `python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v19`
Run: `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_v19_half_lr_restart.py train/0044_g2_dragapult_policy_option_lora/tests/test_lr_profiles.py`
Expected: readiness `READY_AWAITING_USER_LAUNCH`; tests PASS.

### Task 4: Documentation, stopped-run audit, and launch

**Files:**
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md`
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html`
- Modify: `rl_runs/0044_g2_dragapult_policy_option_lora/versions/V18_g4_u57_deck069_telemetry_fix/artifact/status.json`

**Interfaces:**
- Consumes: V18 durable boundary and V19 readiness payload.
- Produces: authoritative LR contract, stopped V18 audit, and active V19 W&B run.

- [ ] **Step 1: Mark V18 stopped at U17**

Record partial U18 discard, W&B synced state, and the user-directed LR transition without deleting any checkpoint.

- [ ] **Step 2: Synchronize both design documents**

Document the seven standard values, project-default precedence, V19 run-only `0.5×` and 1,024-game overrides, and explicit non-binding to deck 069.

- [ ] **Step 3: Run the focused regression suite**

Run: `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_lr_profiles.py train/0044_g2_dragapult_policy_option_lora/tests/test_v19_half_lr_restart.py train/0044_g2_dragapult_policy_option_lora/tests/test_telemetry.py train/0044_g2_dragapult_policy_option_lora/tests/test_assets.py`
Expected: PASS.

- [ ] **Step 4: Launch and audit V19 U1**

Run: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v19 --launch-formal`
Expected: U0 and U1 model-only checkpoints, canonical metrics and W&B config list every half-rate group, 1,024 terminal games, exact focal 069, opponent pool 001–069, no routing/error/unfinished/D2H failures, then the long run remains active.

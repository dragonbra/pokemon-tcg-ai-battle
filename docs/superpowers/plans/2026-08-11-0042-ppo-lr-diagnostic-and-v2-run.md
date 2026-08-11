# 0042 PPO LR Diagnostic and V2 Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan task-by-task with a test-first workflow and review each gate before proceeding.

**Goal:** Compare three Actor learning-rate partitions on one immutable U250 on-policy rollout, select a faster healthy configuration, and launch a new unbounded 0042 formal PPO version that should show materially measurable policy movement by roughly update 50.

**Architecture:** Extend the project-local PPO trainer with explicit decoder/Strategy-Adapter/allocation learning rates and diagnostic-only component-gradient telemetry. Add a U250 branch loader and a standalone same-rollout probe which snapshots model state and frozen PPO targets before replaying identical epoch orders for every arm. Formal training remains a fresh numbered version with a new optimizer, Full0809 opponents, exact-deck lane routing, 256 games/update, three complete data epochs, FP32 PPO, and periodic FP16-storage/FP32-runtime Frozen-0809 evaluation.

**Tech Stack:** Python 3.11, PyTorch/AdamW, CUDA-resident official engine, unittest, JSON diagnostics, TensorBoard, W&B.

## Global Constraints

- Do not modify `engine/source/`.
- U250 of `V1_ppo_protocol_v2_baseline` is the immutable branch checkpoint; U251-U255 remain preserved history and are not deleted.
- Diagnostic artifacts go under `.tmp/evaluation/0042_u250_lr_probe/` and never enter formal W&B or candidate records.
- Every arm uses the same 256 completed games, identical `PreparedBatch`, frozen old log-prob/old Value/advantage/return tensors, and identical epoch permutations.
- Opponents remain independently materialized full immutable `Policy-0809`; focal exact deck remains `007_dragapult_ex`.
- Formal PPO remains 256 games/update, minibatch 2,048 decisions, three maximum full shuffle-without-replacement epochs, FP32, AdamW, no scheduler, zero weight decay, and unbounded updates.
- Formal Frozen evaluation remains every 10 updates through FP16 storage and strict FP32 runtime under the canonical 0042 Frozen-0809 contract.
- A new formal run must use a fresh `V<n>_<tag>` directory and W&B run ID; never append altered semantics to V1.

---

### Task 1: Seal V1 and pin U250 provenance

**Files:**
- Modify runtime artifact: `rl_runs/0042_full_model_design/versions/V1_ppo_protocol_v2_baseline/artifact/status.json`
- Create runtime artifact: `.tmp/evaluation/0042_u250_lr_probe/v1-seal.json`

**Interfaces:**
- Consumes: `checkpoint/update-000250.pt`, its SHA-256 sidecar, U250 Frozen result, and the stopped V1 process state.
- Produces: an auditable branch manifest containing source version/update/checkpoint hash, final observed V1 update, stop reason, and U250 Frozen result.

- [ ] **Step 1: Verify no V1 training or monitor process remains.**

Run: `ps -eo pid,stat,args | rg 'V1_ppo_protocol_v2_baseline|run_full_semantic'`

Expected: no active V1 trainer/monitor.

- [ ] **Step 2: Verify U250 checkpoint and sidecar.**

Run: `sha256sum -c <(sed 's|$|  rl_runs/0042_full_model_design/versions/V1_ppo_protocol_v2_baseline/checkpoint/update-000250.pt|' rl_runs/0042_full_model_design/versions/V1_ppo_protocol_v2_baseline/checkpoint/update-000250.pt.sha256)`

Expected: `OK`.

- [ ] **Step 3: Record V1 as manually stopped without deleting U251-U255.**

Set `state=stopped_manual`, `branch_checkpoint_update=250`, `final_completed_update=255`, and preserve W&B URL/sync facts.

- [ ] **Step 4: Write and validate the branch manifest.**

The JSON must include `source_version`, `source_update`, `checkpoint_path`, `checkpoint_sha256`, `frozen_0809_wins=1251`, `losses=797`, `draws=0`, and `status=PASS`.

### Task 2: Add explicit Actor LR partitions and source-checkpoint branching

**Files:**
- Modify: `train/0042_full_model_design/training/ppo_full_semantic.py`
- Modify: `train/0042_full_model_design/training/run_full_semantic.py`
- Modify: `train/0042_full_model_design/tests/test_ppo_protocol_v2.py`
- Modify: `train/0042_full_model_design/tests/test_0040_long_run_contract.py`

**Interfaces:**
- Consumes: `PPOConfig`, `PPOTrainer`, `load_adapted_model_only(model, path)`.
- Produces: `PPOConfig.decoder_learning_rate`, `policy_adapter_learning_rate`, and `allocation_learning_rate`; `RunConfig.initial_model_checkpoint`; CLI flags with exact persisted provenance.

- [ ] **Step 1: Write failing optimizer-group tests.**

Test that defaults remain `5e-6` for all three Actor groups and explicit values map exactly to `action_decoder`, `policy_strategy_adapter`, and `allocation_head` without changing Value groups.

- [ ] **Step 2: Write failing branch-loader tests.**

Test that a fresh version can strict-load a valid 0042 model-only checkpoint, records the source update/hash, rejects a non-0042 schema, and still initializes a fresh optimizer.

- [ ] **Step 3: Implement LR fields and optimizer mapping.**

Use explicit positive floats; do not infer one group from another at runtime. Persist every effective LR in `training_config.json` and `trainable_parameters.json`.

- [ ] **Step 4: Implement `--initial-model-checkpoint`.**

Build the canonical model first, strict-load with `load_adapted_model_only`, verify representation hash and action contracts, save the loaded weights as the new version's `update-000000.pt`, and record original `source_update=250` separately from new `trainer/update=0`.

- [ ] **Step 5: Run focused tests.**

Run: `python3 -m unittest train.0042_full_model_design.tests.test_ppo_protocol_v2 train.0042_full_model_design.tests.test_0040_long_run_contract -v`

Expected: PASS.

### Task 3: Add non-mutating gradient and clipping telemetry

**Files:**
- Modify: `train/0042_full_model_design/training/ppo_full_semantic.py`
- Test: `train/0042_full_model_design/tests/test_ppo_protocol_v2.py`

**Interfaces:**
- Consumes: scalar loss components from `_backward_microbatch` and optimizer parameter groups.
- Produces: diagnostic methods for component gradient norms/cosines and per-group pre-clip norms; production scalar telemetry for clip scale/trigger without changing the optimizer step.

- [ ] **Step 1: Write failing tests for group norms and clip scale.**

Cover no-gradient groups, finite norms, global norm below/above the limit, and verify the returned clip scale equals `min(1, max_grad_norm/global_norm)`.

- [ ] **Step 2: Implement per-step clipping telemetry.**

Record `gradient/preclip/{action_decoder,policy_adapter,allocation,value_win,value_adapter,value_prize}`, `gradient/global_preclip`, `gradient/global_clip_scale`, and `gradient/global_clip_triggered` before the existing single global clip.

- [ ] **Step 3: Implement isolated Actor loss-source diagnostics.**

On a small fixed diagnostic slice, use `torch.autograd.grad` without optimizer mutation to report raw and weighted norms for PPO-win, Prize actor, entropy, and reference KL, plus pairwise cosine against PPO-win. Separate root and allocation parameters.

- [ ] **Step 4: Add root/macro behavior KL telemetry.**

Keep the existing macro-enriched guard for safety, but report root-only, macro-only, and combined KL/sample counts so the enrichment is explicit.

- [ ] **Step 5: Run focused tests.**

Run: `python3 -m unittest train.0042_full_model_design.tests.test_ppo_protocol_v2 train.0042_full_model_design.tests.test_action_boundary train.0042_full_model_design.tests.test_compound_evaluation_batching -v`

Expected: PASS.

### Task 4: Build and run the identical-rollout U250 LR probe

**Files:**
- Create: `train/0042_full_model_design/diagnostics/u250_lr_probe.py`
- Create runtime outputs: `.tmp/evaluation/0042_u250_lr_probe/report.json`

**Interfaces:**
- Consumes: U250 model-only checkpoint, one 256-game Full0809 rollout, one `PreparedBatch`, and three LR partitions.
- Produces: arm-by-arm epoch telemetry, frozen-target hashes, parameter deltas, component gradients, and a machine-readable recommendation.

- [ ] **Step 1: Add a deterministic dry-run test for arm isolation.**

Verify every arm begins from tensor-identical model weights, receives identical frozen targets and epoch orders, and one arm cannot mutate another arm's baseline snapshot.

- [ ] **Step 2: Collect exactly one rollout.**

Use U250 focal weights, 256 lanes/games, focal deck 007, full Policy-0809 opponents, all 55 exact decks, and require all lane/policy identity gates to pass.

- [ ] **Step 3: Execute the three arms.**

Arm A: `5e-6/5e-6/5e-6`; Arm B: `1e-5/1e-5/1e-5`; Arm C: `1e-5/5e-6/5e-6` for decoder/adapter/allocation. Reset model and optimizer for every arm and reuse identical permutations.

- [ ] **Step 4: Select the arm from evidence.**

Reject any arm with nonfinite values, target refresh, incomplete epoch-1 coverage, hard KL guard, severe clip growth, or rapidly dominating Adapter residual. Prefer the smallest change that produces clearly larger finite epoch-end KL and parameter movement without instability.

- [ ] **Step 5: Persist the complete report.**

Include exact config, rollout identity, valid decisions, all three epoch metrics, source-gradient diagnostics, clipping telemetry, and the selected arm/reason.

### Task 5: Synchronize design and operations documentation

**Files:**
- Modify: `experiments/0042_full_model_design/DESIGN.md`
- Modify: `experiments/0042_full_model_design/DESIGN.html`
- Modify: `docs/rl/0042_full_model_design.md`
- Modify: `docs/rl/0042_full_model_design_operations_manual.md`

**Interfaces:**
- Consumes: selected probe report and final effective config.
- Produces: authoritative V2 training semantics and clone-to-run instructions.

- [ ] **Step 1: Document V1 outcome and evidence boundary.**

Record U250 branch provenance, V1's conservative KL behavior, and that U251-U255 are preserved but not the new branch source.

- [ ] **Step 2: Document selected LR partition and telemetry.**

State exact group LRs, unchanged PPO/reward/Frozen semantics, component-gradient metrics, and global clipping behavior.

- [ ] **Step 3: Update the operations manual.**

Include required 0809 checkpoint/materialization, CUDA build/runtime, U250 source checkpoint, launch command, W&B destination, update-10 evaluation cadence, FP16-storage/FP32-runtime evaluation, monitoring, stop sentinel, and non-candidate rules.

### Task 6: Preflight, launch, and verify the new unbounded formal version

**Files:**
- Create runtime version: `rl_runs/0042_full_model_design/versions/V2_<selected_tag>/`
- Create runtime monitor: `.tmp/training_monitor/0042_full_model_design/V2_<selected_tag>/`

**Interfaces:**
- Consumes: selected LR arm, U250 source checkpoint, synchronized code/docs, canonical Policy-0809 resolver.
- Produces: an online W&B formal run with no update cap and an initial health report.

- [ ] **Step 1: Run the complete focused/full test gates.**

Run the 0042 project tests, Policy-0809 identity tests, mixed-deck CUDA parity, model-only checkpoint tests, and candidate FP16-storage/FP32-runtime tests.

- [ ] **Step 2: Run a short diagnostic smoke from U250.**

Use 2 updates without W&B formal logging; require 256 games/update, 100% epoch-1 coverage, finite three-epoch telemetry, unchanged frozen modules, zero identity/routing failures, and expected faster policy movement.

- [ ] **Step 3: Verify fresh formal paths.**

Ensure artifact/checkpoint/tensorboard/wandb directories and the corresponding W&B run ID are unused.

- [ ] **Step 4: Launch unbounded formal training under the monitor.**

Use no `--updates`, online W&B, `eval-every=10`, Full0809 opponents, U250 initial checkpoint, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

- [ ] **Step 5: Observe at least two complete updates.**

Confirm rollout throughput, 33 or KL-shortened logical optimizer steps, all frozen hashes, per-group LRs, behavior/reference KL, clip telemetry, Adapter dynamics, checkpoint writes, W&B mirroring, and absence of NaN/Inf.

- [ ] **Step 6: Record the update-50 inspection contract.**

The formal run remains unbounded. At U10/U20/U30/U40/U50, report Frozen-0809 results and fixed-reference policy diagnostics; do not require a fixed win-rate threshold, but require measurable divergence from the U250 branch point and healthy PPO boundaries.

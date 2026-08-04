# 0031 Training Throughput Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase 0031 epoch throughput without changing the model, dataset order, loss, optimizer semantics, exact epoch-2 resume boundary, checkpoint contract, or W&B run identity.

**Architecture:** Profile the existing PyTorch training step as a pipeline, optimize only the measured bottleneck, and admit a change only after deterministic correctness and repeated throughput benchmarks. Keep all executable changes self-contained under `train/0031_rule_faithful_semantic_foundation_pretraining/`; synchronize the 0031 DESIGN documents only if the admitted execution path changes their recorded training contract.

**Tech Stack:** Python 3.11, PyTorch CUDA/bfloat16, AdamW, PyTorch Profiler/NSight tools, unittest, W&B.

## Global Constraints

- Do not modify `engine/source/`.
- Resume the existing `V4_lr5e4_no_early_stop_b512` run from exact epoch 2/update 33002.
- Keep batch size 512, learning rate 0.0005, weight decay 0.02, seed 20260802, full train/validation passes, and all model/data/action contracts unchanged.
- Continue the existing W&B run with `WANDB_RESUME=must`.
- Preserve every completed epoch/update model-only checkpoint and the exact epoch resume checkpoint.
- Do not stop automatically for convergence; train until the user explicitly requests a stop.

---

### Task 1: Reproducible Pipeline Baseline

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/benchmark_training.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_training_pipeline.py`
- Create: `.tmp/0031_training_optimization/run-*/baseline.json` (generated, ignored)

**Interfaces:**
- Consumes: existing `benchmark(...)` and `_train_epoch(...)` training path.
- Produces: per-stage CPU/CUDA timings, decisions/s, data wait fraction, peak memory, and exact benchmark configuration.

- [x] Establish bounded warmup and measured-batch windows without changing formal training metrics.
- [x] Capture the original formal batch-512 reference under `.tmp/0031_training_optimization/`.
- [x] Profile the representative exact epoch-2 path with PyTorch profiler and explicit stage timings.
- [x] Save repeatable reference and candidate measurements under `.tmp/0031_training_optimization/`.

### Task 2: Root-Cause Isolation

**Files:**
- Read: `train/0031_rule_faithful_semantic_foundation_pretraining/training/trainer.py`
- Read: `train/0031_rule_faithful_semantic_foundation_pretraining/training/prefetch.py`
- Read: `train/0031_rule_faithful_semantic_foundation_pretraining/features/collate.py`
- Read: `train/0031_rule_faithful_semantic_foundation_pretraining/model/policy.py`
- Create: `.tmp/0031_training_optimization/run-*/profile.json` (generated, ignored)

**Interfaces:**
- Consumes: Task 1 timings and profiler trace.
- Produces: one evidence-backed primary bottleneck hypothesis and ranked secondary costs.

- [x] Profile representative post-warmup steps and attribute time to input collation, H2D transfer, forward, backward, gradient clipping, and AdamW.
- [x] Compare GPU activity, kernel launch count, tensor shapes, and host wait with the existing epoch-2 `data_wait_fraction` evidence.
- [x] Record the primary root cause and reject optimizations that would change effective batch, sample order, loss, or model math.

### Task 3: Minimal Throughput Optimization

**Files:**
- Modify only the measured owner among `training/trainer.py`, `training/prefetch.py`, `features/collate.py`, `model/policy.py`, and `model/action_decoder.py`.
- Modify: `run_bc.py` only when an execution-only option must be wired into the exact resume path.
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_training.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_training_pipeline.py`

**Interfaces:**
- Consumes: unchanged canonical batches and exact restored model/optimizer/RNG state.
- Produces: the same loss and optimizer update contract through a faster execution path.

- [x] Add focused equivalence/backward regressions for the selected optimization.
- [x] Implement one optimization axis at a time, reverting candidates that do not exceed benchmark noise.
- [x] Verify finite gradients, parameter updates, teacher logits/loss tolerance, sample coverage/order, and model-only/resume checkpoint rejection rules.
- [x] Run the complete 0031 project test suite and relevant repository training/W&B tests.

### Task 4: Admission Benchmark And Documentation

**Files:**
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md` if the recorded execution path changes.
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.html` if the recorded execution path changes.
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DECISIONS.md`
- Create: `.tmp/0031_training_optimization/run-*/optimized.json` (generated, ignored)

**Interfaces:**
- Consumes: baseline and optimized benchmark artifacts.
- Produces: measured speedup, correctness gate result, resource envelope, and rollback boundary.

- [x] Repeat baseline and optimized runs after warmup at 150- and 300-batch windows.
- [x] Require a stable improvement beyond run-to-run noise and no correctness/checkpoint/W&B regression.
- [x] Record admitted settings, measured hardware-specific speedup, and evidence paths in 0031 decisions/design documents.
- [ ] Commit only the scoped implementation, tests, and documentation without unrelated user changes.

### Task 5: Exact Resume And Continuous Monitoring

**Files:**
- Update by runtime: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V4_lr5e4_no_early_stop_b512/artifact/status.json`
- Append by runtime: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V4_lr5e4_no_early_stop_b512/artifact/training_metrics.jsonl`
- Append by runtime: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V4_lr5e4_no_early_stop_b512/artifact/training_monitor.jsonl`

**Interfaces:**
- Consumes: exact epoch-2 resume checkpoint and admitted faster execution path.
- Produces: epoch 3 onward on the existing W&B curve, with foreground watchdog heartbeats.

- [x] Verify checkpoint epoch 2/update 33002, compatibility hashes, and canonical metrics boundary immediately before launch.
- [x] Launch `monitor_training --resume` in a foreground terminal with the unchanged V4 semantic configuration and admitted execution-only settings.
- [x] Confirm epoch 3 starts, W&B resumes the stable run ID, and initial resource/watchdog health remains normal; checkpoint advancement remains monitored until the epoch boundary.
- [ ] Keep polling the foreground watchdog at intervals no longer than 60 seconds until the user explicitly requests a stop.

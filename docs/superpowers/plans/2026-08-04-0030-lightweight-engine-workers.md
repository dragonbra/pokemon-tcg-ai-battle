# 0030 Lightweight Engine Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Torch/CUDA from isolated official-engine workers, calibrate the highest zero-error concurrency that fits the 23 GiB host, and restart 0030 decoder-only PPO as a strict new version.

**Architecture:** The parent collector remains the only process that compiles semantic features and runs the shared GPU encoder/decoders. Spawned workers import only the rollout protocol, evaluation loader, and official `libcg`; the rollout package exposes the collector lazily. A benchmark records worker PSS, imported modules, official-engine throughput, errors, timeouts, and GPU memory before formal training chooses 64, 96, or 128 workers.

**Tech Stack:** Python 3.11, `multiprocessing` spawn, PyTorch parent inference, official engine runtime, `unittest`, Linux `/proc`, W&B online logging.

## Global Constraints

- Do not modify `engine/source/`.
- Keep 0030 self-contained and do not import executable code from another numbered project.
- Preserve V2 artifacts and create strictly new `V3_lightweight_engine_workers` paths.
- Resume from a V2 model-only checkpoint with a fresh optimizer and newly collected on-policy data.
- Keep the semantic encoder, option encoder, opponent decoder, feature schema, action contract, reward, and PPO objective unchanged.
- Formal training uses W&B project `dragon_bra/pokemon-tcg-policy-learning` and a foreground watchdog session.
- Select worker count only from zero-error official-engine benchmarks; do not weaken timeout or isolation contracts.

---

### Task 1: Close V2 At The Update 5 Evaluation Boundary

**Files:**
- Modify: `rl_runs/0030_dragapult_shared_encoder_decoder_rl/versions/V2_trimmed_fp16_prototype_cache/artifact/status.json`
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/DECISIONS.md`

**Interfaces:**
- Consumes: `training_metrics.jsonl` record with `checkpoint/update == 5` and `eval/checkpoint_update == 5`.
- Produces: preserved `update-0005.pt`, recorded frozen evaluation, and explicit intentional-stop provenance.

- [ ] **Step 1: Poll the foreground watchdog**

Run: poll session `97799` at intervals no greater than 60 seconds.
Expected: heartbeat remains healthy until update 5 metrics are flushed.

- [ ] **Step 2: Verify the update 5 record**

Run: `tail -n 2 rl_runs/0030_dragapult_shared_encoder_decoder_rl/versions/V2_trimmed_fp16_prototype_cache/artifact/training_metrics.jsonl`
Expected: one record contains both `"checkpoint/update": 5` and `"eval/checkpoint_update": 5.0` with 102 evaluation episodes.

- [ ] **Step 3: Stop the foreground launcher**

Send `Ctrl-C` to watchdog session `97799` and wait for both launcher and child to exit. Verify no descendant process remains under the recorded training PID.

- [ ] **Step 4: Record the intentional transition**

Write V2 status/decision evidence stating that update 5 is the last accepted checkpoint, any incomplete update 6 rollout was discarded, and the stop reason is worker import memory amplification.

### Task 2: Add A Failing Torch-Light Worker Contract

**Files:**
- Create: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_worker_imports.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/rollout/__init__.py`

**Interfaces:**
- Consumes: subprocess import of `train.0030_dragapult_shared_encoder_decoder_rl.rollout.worker`.
- Produces: a regression test that fails if `torch` is present in `sys.modules` after worker import.

- [ ] **Step 1: Write the subprocess test**

```python
def test_worker_import_does_not_load_torch(self):
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import train.0030_dragapult_shared_encoder_decoder_rl.rollout.worker; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)"
        ),
    ]
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    self.assertEqual(result.returncode, 0)
```

- [ ] **Step 2: Run the test before the fix**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_worker_imports`
Expected: FAIL because `rollout/__init__.py` eagerly imports `collector`, which imports `torch`.

- [ ] **Step 3: Implement the minimal lazy export**

Keep protocol dataclasses eager and resolve only `HeterogeneousRolloutCollector` through module `__getattr__`, matching the established 0022/0023 pattern without importing those projects.

- [ ] **Step 4: Run focused and project tests**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_worker_imports train.0030_dragapult_shared_encoder_decoder_rl.tests.test_contract`
Expected: PASS, including the existing public import contract.

### Task 3: Add Worker Memory Diagnostics

**Files:**
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/benchmark.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_worker_imports.py`

**Interfaces:**
- Produces: benchmark JSON keys `worker_count_peak`, `worker_pss_total_bytes`, `worker_pss_mean_bytes`, `worker_torch_import_count`, `worker_cuda_mapping_count`, and existing engine/inference metrics.

- [ ] **Step 1: Test `/proc` aggregation with fixture text**

Add a pure parser test for `Pss:` and mapped library names, including missing/exited PIDs.

- [ ] **Step 2: Run the parser test and confirm failure**

Run the focused unittest; expected failure is a missing parser/aggregator.

- [ ] **Step 3: Implement bounded sampling**

Sample only collector-owned live worker PIDs during the official-engine benchmark. Record PSS and forbidden Torch/CUDA imports without retaining process handles after cleanup.

- [ ] **Step 4: Verify diagnostics**

Run a 4-game, 2-worker CUDA benchmark. Expected: 4/4 valid games, zero Torch workers, zero worker CUDA mappings, positive worker PSS, and no leaked child processes.

### Task 4: Calibrate Safe Concurrency

**Files:**
- Create: `.tmp/evaluation/0030_performance/V3_workers_64.json`
- Create: `.tmp/evaluation/0030_performance/V3_workers_96.json`
- Create: `.tmp/evaluation/0030_performance/V3_workers_128.json`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/INFERENCE_PERFORMANCE.md`

**Interfaces:**
- Consumes: fixed checkpoint update 5, fixed seeds, 128 official-engine games per configuration.
- Produces: selected formal worker count based on zero errors, no timeout/OOM, PSS headroom, and games/s.

- [ ] **Step 1: Run 64 workers**

Run the benchmark with 128 games, 64 workers, `cuda:0`, and fixed seed `20260804`.

- [ ] **Step 2: Run 96 workers**

Use the identical game/seed contract and record the same fields.

- [ ] **Step 3: Run 128 workers**

Use the identical contract; a timeout or engine error disqualifies the setting rather than relaxing safeguards.

- [ ] **Step 4: Select concurrency**

Choose the fastest zero-error setting that retains at least 2 GiB available RAM and does not increase swap during steady rollout. Record rejected settings and exact evidence.

### Task 5: Create V3 And Synchronize Authority Documents

**Files:**
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/training/run.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/__main__.py`
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/DESIGN.md`
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/DESIGN.html`
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/DECISIONS.md`
- Create: `rl_runs/0030_dragapult_shared_encoder_decoder_rl/versions/V3_lightweight_engine_workers/artifact/training_config.json`

**Interfaces:**
- Consumes: V2 update-5 model-only checkpoint and selected worker count.
- Produces: an explicit initialization-checkpoint argument, hash-verified V3 model loading, fresh PPO optimizer, and synchronized project records.

- [ ] **Step 1: Test initialization checkpoint validation**

Add tests that accept V2 update 5 only as model weights, reject optimizer/RNG/rollout fields, and preserve the frozen representation hash.

- [ ] **Step 2: Implement explicit V3 initialization**

Add an initialization checkpoint CLI/config field. Load model weights before constructing `PPOTrainer`; do not restore optimizer or append V2 metrics.

- [ ] **Step 3: Update both DESIGN formats and decisions**

Record that model schema and objective are unchanged, only worker process imports/concurrency changed, and V3 starts from the exact V2 checkpoint SHA-256 with a fresh optimizer.

- [ ] **Step 4: Run the complete 0030 test suite and smoke gate**

Run all project unittests, then a fresh-version official-engine smoke update with online logging disabled. Expected: zero failures/errors, unchanged representation hash, and model-only checkpoint validation.

### Task 6: Restart Formal V3 With Foreground Monitoring

**Files:**
- Create: `rl_runs/0030_dragapult_shared_encoder_decoder_rl/versions/V3_lightweight_engine_workers/{artifact,checkpoint,tensorboard,wandb}/`
- Create: `.tmp/training_monitor/0030_dragapult_shared_encoder_decoder_rl/V3_lightweight_engine_workers/`

**Interfaces:**
- Produces: formal online W&B run, foreground watchdog session, and continuous health records.

- [ ] **Step 1: Assert all V3 paths are fresh**

Verify artifact, checkpoint, TensorBoard, W&B, monitor, and authoritative evaluation paths do not already contain a run.

- [ ] **Step 2: Launch formal training**

Use update-5 weights, 512 games/update, the selected worker count, `cuda:0`, seed `20260804`, evaluation every 5 updates, and W&B online mode.

- [ ] **Step 3: Keep the foreground watchdog attached**

Poll at intervals no greater than 60 seconds. Treat engine errors, worker EOF, timeout, OOM, W&B failure, low RAM/disk, or a nonzero child exit as immediate alerts.

- [ ] **Step 4: Report the first V3 evidence**

After the initial frozen evaluation and first completed update, report worker PSS, swap behavior, games/s, sampled rollout separately from frozen greedy evaluation, and the W&B URL.

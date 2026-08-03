# 0028 Training Throughput Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the complete 0028 actor/model semantics unchanged while preventing the RTX 5080 training loop from serially waiting for gzip/JSON/collation work.

**Architecture:** Preserve the immutable canonical JSONL dataset as the auditable source of truth. Add deterministic length-aware batch planning inside each shuffled shard window and a bounded producer thread that prepares CPU batches while the GPU executes the previous batch. Measure producer wait separately from GPU compute; only add a second mmap tensor dataset if the lightweight pipeline cannot reach 85% of measured GPU-only throughput.

**Tech Stack:** Python 3.11, PyTorch 2.11, gzip JSONL, `queue.Queue`, `threading.Thread`, unittest, CUDA/bfloat16, W&B.

## Global Constraints

- Project ID remains exactly `0028_universal_semantic_foundation_pretraining`; no `0027` path may be created.
- Do not modify `engine/source/` or change any actor-visible field, target, model parameter, loss, split, or source/persona visibility rule.
- Preserve canonical JSONL shard hashes and treat any optimized representation as derived, replaceable data.
- Batch planning must be deterministic for a fixed split/seed and must cover every decision exactly once per epoch.
- Provenance may accompany validation batches for audit but must never enter actor tensors or model forward.
- Formal versions are immutable. The stopped `V1_mid_universal_semantic_foundation` is not reusable; retraining must allocate `V2_<tag>`.
- Formal training keeps one train pass and one full teacher-forced plus greedy validation pass per epoch, model-only checkpoints, JSONL then TensorBoard then W&B, and a foreground watchdog.
- Acceptance requires at least 85% of matched-shape GPU-only decisions/s, no producer exception loss, exact record coverage, unchanged tensors for the same records, 27 existing tests plus new performance-pipeline tests passing, and no CUDA OOM.

---

### Task 1: Preserve The Stopped V1 And Add Performance Diagnostics

**Files:**
- Modify: `rl_runs/0028_universal_semantic_foundation_pretraining/versions/V1_mid_universal_semantic_foundation/artifact/status.json`
- Create: `rl_runs/0028_universal_semantic_foundation_pretraining/versions/V1_mid_universal_semantic_foundation/artifact/training_summary.json`
- Create: `experiments/0028_universal_semantic_foundation_pretraining/decisions/2026-08-03-v1-throughput-stop.md`
- Create: `train/0028_universal_semantic_foundation_pretraining/benchmark_training.py`

**Interfaces:**
- Consumes: `CanonicalDecisionDataset`, `SemanticPolicy`, and `teacher_batch`.
- Produces: one JSON benchmark record with `load_seconds`, `collate_seconds`, `data_wait_seconds`, `step_seconds`, end-to-end decisions/s, GPU-only decisions/s, peak CUDA memory, and padded/actual token ratios.

- [ ] **Step 1: Record V1 as stopped by user**

Set `state=stopped_by_user`, `epochs_completed=0`, `checkpoint_count=0`, and record that graceful W&B close was not observed after watchdog interruption.

- [ ] **Step 2: Write a benchmark-output contract test**

Add a test that calls the benchmark on two fixture batches and asserts all timing/count keys are finite, nonnegative, and JSON serializable.

- [ ] **Step 3: Run the focused test and observe failure**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_training_pipeline`

Expected: FAIL because `benchmark_training` and its result contract do not exist.

- [ ] **Step 4: Implement the benchmark boundary timers**

The benchmark must time gzip/JSON loading, canonical collation, cached CPU-batch GPU steps, and complete iteration independently. CUDA timings must surround `torch.cuda.synchronize()` calls; it must not label asynchronous launch time as GPU compute time.

- [ ] **Step 5: Save the measured V1 baseline**

Run 100 end-to-end batches and 20 cached GPU-only batches at batch 256. Write the JSON result under `.tmp/evaluation/0028_throughput_baseline/`; do not commit the temporary report.

### Task 2: Deterministic Length-Aware Batch Planning

**Files:**
- Modify: `train/0028_universal_semantic_foundation_pretraining/training/dataset.py`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_training_pipeline.py`

**Interfaces:**
- Consumes: canonical records and `(split, batch_size, seed)`.
- Produces: `CanonicalDecisionDataset.iter_record_batches(..., length_bucketed: bool = True)` with exact-once deterministic coverage.

- [ ] **Step 1: Write failing determinism and coverage tests**

Construct records with distinct audit identities and variable card/option/effect lengths. Assert two runs with the same seed have identical batch identity order, different seeds change training batch order, validation order stays fixed, and flattened identities equal the input multiset exactly.

- [ ] **Step 2: Write a failing padding-efficiency test**

For the synthetic variable-length fixture, compare current shuffled batching with length-aware batching and require fewer total padded option/effect cells without changing collated tensor values for any record.

- [ ] **Step 3: Implement bounded-window multi-dimensional bucketing**

Within each loaded shard window, randomize ties first, then stable-sort with finite buckets derived only from actor-visible lengths: state length bucket 32, option length bucket 4, effect length bucket 16, and skill length bucket 8. Shuffle complete batch groups after sorting and carry the final incomplete group into the next shard so `batch_count()` remains exact.

- [ ] **Step 4: Run dataset and pipeline tests**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_dataset train.0028_universal_semantic_foundation_pretraining.tests.test_training_pipeline`

Expected: PASS with exact coverage, deterministic order, and improved padding utilization.

### Task 3: Bounded Background Batch Prefetch

**Files:**
- Create: `train/0028_universal_semantic_foundation_pretraining/training/prefetch.py`
- Modify: `train/0028_universal_semantic_foundation_pretraining/training/trainer.py`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_training_pipeline.py`

**Interfaces:**
- Produces: `PrefetchIterator(iterable, depth: int)` with properties `wait_seconds`, `produced`, and `consumed`.
- Consumes: the dataset's CPU batch iterator.
- Trainer metrics add `data_wait_seconds`, `data_wait_fraction`, `prefetch_depth`, and `producer_batches` under the existing `bc/*/optimization/*` namespace.

- [ ] **Step 1: Write failing ordering, exception, and early-close tests**

Assert bounded prefetch preserves item order; a producer exception is re-raised in the consumer; and closing early joins the producer without leaving a live thread.

- [ ] **Step 2: Implement a bounded queue with an explicit terminal message**

Use one producer thread and `queue.Queue(maxsize=depth)`. Store exception plus traceback in a terminal item, use a cancellation event for early close, and never silently truncate the iterable.

- [ ] **Step 3: Integrate prefetch into train and validation loops**

Default formal prefetch depth to 2 and allow `0` to disable it for matched benchmarks. Wrap iteration in a context manager so exceptions and early limits always close the producer. Change `tqdm.set_postfix(..., refresh=False)` so the PTY is not redrawn for every batch.

- [ ] **Step 4: Run all 0028 unit tests**

Run: `python3 -m unittest discover -s train/0028_universal_semantic_foundation_pretraining/tests -v`

Expected: all existing and new tests pass; actor keys and model outputs are unchanged.

### Task 4: Performance Gate And Optional Tensor Materialization

**Files:**
- Modify: `train/0028_universal_semantic_foundation_pretraining/benchmark_training.py`
- Conditionally create: `train/0028_universal_semantic_foundation_pretraining/training/materialized.py`
- Conditionally modify: `train/0028_universal_semantic_foundation_pretraining/training/dataset.py`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_training_pipeline.py`

**Interfaces:**
- Produces: a matched baseline/optimized comparison and an explicit `cpu_pipeline_gate_passed` boolean.
- Optional materializer produces compact ragged tensor shards plus offsets, immutable hash manifest, and provenance sidecar; actor batches remain exactly equal to canonical JSON collation.

- [ ] **Step 1: Benchmark bucket-only, prefetch-only, and combined paths**

Use the same 100-shard-window batches, batch size 256, seed 20260802, model weights, AMP mode, and warmup. Report end-to-end/GPU-only ratio rather than comparing unrelated batch shapes.

- [ ] **Step 2: Apply the acceptance gate**

Pass when end-to-end throughput is at least 85% of matched-shape GPU-only throughput, producer wait is below 15% of epoch time, every record is covered, and peak reserved CUDA memory remains below 13 GiB excluding the display baseline.

- [ ] **Step 3: Materialize only if the gate fails**

If needed, write compact ragged tensors with per-field offsets using integer widths validated against schema vocabularies. Reload with `torch.load(..., weights_only=True, mmap=True)`, verify every shard SHA-256, and test exact tensor equality against canonical collation. Do not create fixed global-padding shards.

- [ ] **Step 4: Re-run the gate after any materialization**

The same benchmark contract must pass before formal training authorization. Record size, build time, load time, and hash manifest if this branch is used.

### Task 5: Authority Documents And Formal V2 Launch

**Files:**
- Modify: `experiments/0028_universal_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0028_universal_semantic_foundation_pretraining/DESIGN.html`
- Modify: `experiments/0028_universal_semantic_foundation_pretraining/manifest.json`
- Create: `experiments/0028_universal_semantic_foundation_pretraining/decisions/2026-08-03-v2-optimized-pipeline.md`
- Modify: `train/0028_universal_semantic_foundation_pretraining/run_bc.py`
- Modify: `train/0028_universal_semantic_foundation_pretraining/monitor_training.py`

**Interfaces:**
- Formal version: `V2_optimized_mid_universal_semantic_foundation`.
- Training CLI records bucket policy, prefetch depth, performance-gate report SHA-256, and unchanged canonical dataset manifest SHA-256.

- [ ] **Step 1: Synchronize DESIGN and manifest**

Document V1 as a stopped throughput baseline, the measured root cause, the optimized data flow, benchmark evidence, tensor shapes, unchanged model/action contract, and V2 as the next formal stage.

- [ ] **Step 2: Add formal configuration and status fields**

Record `length_bucketed=true`, `prefetch_depth=2`, benchmark commitment, W&B online metadata, and canonical dataset commitment in `training_config.json` and status.

- [ ] **Step 3: Run final preflight**

Run JSON parsing, the full 0028 unit suite, exact dataset hash reload, a 100-batch CUDA benchmark, and confirm `engine/source/` has no changes and the V2 run/evaluation paths are unused.

- [ ] **Step 4: Launch formal V2 only after the gate passes**

Use batch 256, validation batch 256, at most 20 epochs, early stopping patience 5, W&B online, finite model-only checkpoints, and the foreground watchdog. Confirm the first live heartbeat includes active W&B state, healthy RAM/swap/disk, and optimized data-wait metrics.

# Async Collate and H2D Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap CPU canonical collation, pinned-memory H2D transfer, and GPU model execution in the local resident inference service without changing policy semantics or submission packages.

**Architecture:** Keep worker-local feature compilation unchanged. Add an opt-in bounded three-stage pipeline inside each resident GPU policy server: the dispatch thread collates raw records and pins the resulting CPU tensors, a transfer thread copies one prepared batch on a dedicated CUDA stream, and a compute thread waits on the transfer event before running the unchanged deterministic model and decoding actions. Queue capacity one provides double buffering without unbounded latency or memory growth; the existing synchronous path remains the fallback.

**Tech Stack:** Python 3.11, PyTorch CUDA streams/events, pinned host tensors, AF_UNIX resident inference, official-engine Policy-0806 profiler, unittest.

**Measured outcome:** Implemented as an opt-in diagnostic path and deliberately left default-off.
The exact 64-game comparison was 479.58 selections/s for synchronous inference versus 432.80
selections/s for selective-pin async inference; the current Policy-0806 workload has only 0.33
seconds of measured H2D GPU time but 5.93 seconds of collate/pin cost. The optimization does not
enter packaged agents and must be recalibrated only for materially larger future models/records.

## Global Constraints

- Never modify `engine/source/`, frozen policy packages, checkpoint weights, model structure, feature schema, action contract, or official observation JSON.
- The async path is evaluation/RL rollout infrastructure only and must remain opt-in.
- CPU and CUDA paths must preserve tensor keys, shapes, dtypes, values, deterministic actions, legality, session affinity, and request order.
- Pipeline queues have capacity one and errors fail every affected request closed; no silent topology fallback may contaminate a benchmark.
- Non-CUDA devices use the existing synchronous path.
- Temporary benchmark reports belong below `.tmp/evaluation/evaluation_inference_profile/`.

---

### Task 1: Pinned Batch Contract

**Files:**
- Modify: `evaluation/runner/inference_server.py`
- Test: `tests/test_evaluation_inference_server.py`

**Interfaces:**
- Consumes: a canonical CPU tensor dictionary returned by `_collate_raw_records_equivalent`.
- Produces: `_pin_cpu_batch(torch, batch) -> dict[str, Tensor]`, preserving exact values, keys, shapes, and dtypes while pinning every tensor on CUDA-capable hosts.

- [ ] Add a fake-tensor unit test proving every field is copied once with `pin_memory=True` and that values, dtype, and shape are unchanged.
- [ ] Add `_pin_cpu_batch` with a CUDA-unavailable fallback that returns the original CPU batch for unit-test and CPU-server compatibility.
- [ ] Run `python3 -m unittest -v tests.test_evaluation_inference_server` and require all tests to pass.

### Task 2: Bounded Transfer/Compute Pipeline

**Files:**
- Modify: `evaluation/runner/inference_server.py`
- Test: `tests/test_evaluation_inference_server.py`

**Interfaces:**
- Produces `_PreparedInference` carrying active requests, pinned CPU batch, and timing metadata.
- Produces `_TransferredInference` carrying device tensors and a CUDA transfer-complete event.
- `PolicyServer(async_h2d=True)` creates capacity-one prepared/transferred queues, one dedicated transfer stream, and two daemon stage threads.

- [ ] Add queue-stage tests with fake streams/events proving FIFO order, capacity-one backpressure, compute waiting on the matching transfer event, and exception propagation to every request.
- [ ] Split synchronous `_infer` into CPU preparation and unchanged model/decode completion helpers.
- [ ] Implement transfer stage using `with torch.cuda.stream(transfer_stream)`, `tensor.to(device, non_blocking=True)`, and `event.record(transfer_stream)`.
- [ ] Implement compute stage using the current compute stream’s `wait_event`, unchanged dtype/model/action logic, and per-stage profiling without unconditional H2D synchronization.
- [ ] Keep `_infer` as the exact synchronous fallback for CPU or disabled async mode.
- [ ] Run `python3 -m unittest -v tests.test_evaluation_inference_server` and require all tests to pass.

### Task 3: Configuration and Audit Surface

**Files:**
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/performance_profile.py`
- Modify: `evaluation/runner/inference_server.py`
- Test: `tests/test_evaluation_batch.py`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Adds `async_h2d: bool = False` to `BatchConfig` and `--async-h2d` to the inference server and performance profiler.
- Records async mode, pin/collate seconds, H2D enqueue/GPU-event time, model time, queue wait, and stage errors in profile/manifest payloads.

- [ ] Add configuration tests rejecting async H2D without CUDA resident inference and recording the flag in workload/manifest payloads.
- [ ] Thread the flag through server process launch without changing formal evaluation defaults.
- [ ] Ensure `close()` drains/stops stage threads and profile reset remains thread-safe.
- [ ] Run the focused batch, profiler, and inference test suites.

### Task 4: Exactness and Official-Engine Benchmark

**Files:**
- Modify: `evaluation/README.md`
- Create: `.tmp/evaluation/evaluation_inference_profile/async_h2d_n16_e6_b128_w1_g256.json`
- Modify: `.tmp/evaluation/evaluation_inference_profile/summary.md`

**Interfaces:**
- Consumes frozen Policy-0806, N16E6, batch 128, wait 1 ms, FP16, 256 games, seed 8062026, and the fixed schedule hash.
- Produces completion, error, throughput, latency, CPU, RSS, batch, collate/pin, H2D, and model evidence against the 557.45 selections/s synchronous baseline.

- [ ] Run a small official-engine smoke and require identical selection count, 100% completion, and zero errors.
- [ ] Run the 256-game async benchmark and require 256/256 completion and zero errors.
- [ ] Keep async mode only if it is semantically exact and improves throughput or exposes a measured next bottleneck without pathological latency/RSS.
- [ ] Document that the optimization is rollout/evaluation-only and does not enter the final packaged agent.
- [ ] Run 94+ relevant unit tests, `py_compile`, `git diff --check`, and verify `git diff --name-only -- engine/source` is empty.

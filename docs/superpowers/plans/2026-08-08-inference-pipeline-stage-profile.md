# Inference Pipeline Stage Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure where Policy-0806 closed-loop inference time is spent and determine whether throughput is limited by fragmented batch launches, IPC, queueing, CPU collation, H2D, GPU model execution, or response waiting.

**Architecture:** Extend the existing opt-in inference profiler with monotonic cross-process request timestamps, worker-side connection/send/response-wait timers, server queue/coalescing timers, and separately measured CPU collate, GPU H2D, model, and decode stages. Preserve the current request/action contract and run the profiler on the established N16E6/B128/1ms synchronous-H2D official-engine workload.

**Tech Stack:** Python 3.11, `multiprocessing.connection` AF_UNIX transport, `perf_counter_ns`, PyTorch CUDA events, frozen Policy-0806, official engine runtime, unittest.

## Global Constraints

- Never modify `engine/source/`, the frozen Policy-0806 package, checkpoint, feature schema, model, or action contract.
- All new instrumentation is active only when `inference_profile=True`; ordinary evaluation remains unchanged.
- Cross-process durations use the host monotonic clock and are reported as diagnostic wall latency, not CPU time.
- Keep backward-compatible aggregate keys such as `collate_h2d_seconds` and `request_latency_ms`.
- The admission run is N16E6/B128/1ms, FP16, synchronous H2D, worker-local stateless compiler, 256 official-engine games, seed 8062026.

---

### Task 1: Stage-Timing Contract

**Files:**
- Modify: `evaluation/runner/inference_server.py`
- Test: `tests/test_evaluation_inference_server.py`

**Interfaces:**
- Consumes: `_InferenceRequest.enqueued_ns`, `_InferenceProfile`, `PolicyServer._dispatch`, `PolicyServer._infer`.
- Produces: percentile distributions for ingress, queue wait, batch coalescing, dispatch idle, and batch intervals; separate `collate_cpu_seconds`, `h2d_wall_seconds`, `h2d_gpu_seconds`, `dtype_cast_wall_seconds`, model, and decode totals.

- [x] Add failing profile tests for named latency distributions and batch timing summaries.
- [x] Add a generic `record_latency_ms(name, value)` collector with count/total/p50/p95/p99 output.
- [x] Record first-request idle wait, per-request queue wait, coalescing duration, batch-cap/deadline reason, and inter-launch interval in `_dispatch`.
- [x] Split synchronous raw-record collation from H2D using CPU wall timing and CUDA events while retaining `collate_h2d_seconds`.
- [x] Record request ingress latency from an optional worker monotonic timestamp without changing action semantics.
- [x] Run `python3 -m unittest -v tests.test_evaluation_inference_server` and require all tests to pass.

### Task 2: Worker IPC Timing

**Files:**
- Modify: `evaluation/runner/engine_pool_worker.py`
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/performance_profile.py`
- Test: `tests/test_evaluation_engine_pool_worker.py`
- Test: `tests/test_evaluation_batch.py`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Consumes: `BatchConfig.inference_profile` and pooled connection `_exchange`.
- Produces: per-game `ipc_connection_wait_seconds`, `ipc_send_seconds`, `ipc_response_wait_seconds`, `ipc_roundtrip_seconds`, and `ipc_calls`, aggregated in profile JSON.

- [x] Add failing tests proving timing is disabled by default and profile requests carry a monotonic client-send timestamp.
- [x] Thread `inference_profile` into engine-pool subprocess payloads.
- [x] Measure connection acquisition, request send, response wait, and complete round trip per agent only in profile mode.
- [x] Aggregate IPC fields without double-counting shared connection-pool totals.
- [x] Run the focused worker, batch, and performance-profile tests and require all tests to pass.

### Task 3: Official-Engine Diagnosis

**Files:**
- Create: `.tmp/evaluation/evaluation_inference_profile/stage_profile_n16_e6_b128_w1_g256.json`
- Modify: `.tmp/evaluation/evaluation_inference_profile/summary.md`
- Modify: `evaluation/README.md`

**Interfaces:**
- Consumes: the stage timing JSON produced by Tasks 1–2.
- Produces: an evidence-backed bottleneck ranking and the next bounded optimization experiment.

- [x] Run a 4-game smoke and require 4/4 completion, zero errors, and 924 selections.
- [x] Run the 256-game fixed official-engine workload and require 256/256 completion and zero errors.
- [x] Calculate per-request and per-batch stage costs, GPU active fraction, batch-cap/deadline fractions, queue-delay percentiles, and IPC wait fractions.
- [x] State whether the evidence supports fragmented launches, transport overhead, central CPU preprocessing, GPU compute saturation, or insufficient concurrency.
- [x] Run the complete relevant test set, `py_compile`, `git diff --check`, and confirm `engine/source/` is untouched.

### Measured outcome

The profile rejects insufficient concurrency and H2D as leading bottlenecks. Most batches are near
the 96-environment ceiling and H2D costs 1.45 ms/batch, but GPU model execution occupies only 16.7%
of wall time. Central CPU collate costs 24.84 ms/batch, while ingress, handler wakeup, and a 44.65
ms/batch dispatch handoff residual identify Python record fan-in/GIL scheduling as the first target.

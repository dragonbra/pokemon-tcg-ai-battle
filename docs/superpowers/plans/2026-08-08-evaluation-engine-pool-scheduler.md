# Evaluation Engine Pool Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase Policy-0806 official-engine throughput at the required 16 worker processes by keeping a larger per-worker engine pool productive while resident GPU inference is in flight.

**Architecture:** Preserve one official runtime and multiple independent battle pointers per worker, but decouple resident engine count E from inference channel count I. Add explicit causal session IDs to the resident-inference protocol so many engine sessions can safely reuse a bounded connection pool without mixing encoder state; this prevents E16 from creating 512 server-side connection threads. Extend profiling with process-tree CPU accounting, then compare N16E8 and N16E16/I8 on an equal Policy-0806 workload.

**Tech Stack:** Python 3.11, ctypes official `libcg`, AF_UNIX `multiprocessing.connection`, threads for independent inference I/O, official-engine evaluation reports, unittest.

## Global Constraints

- Do not modify `engine/source/`, packaged `cg/`, feature compiler code, or policy/model code.
- Use exactly 16 outer worker processes for the acceptance benchmark.
- Every live environment owns one battle pointer and calls `BattleFinish` exactly once.
- Preserve stable game ordering, trace shape, seed, seat, error, retry, and report contracts.
- Benchmark the frozen Policy-0806 workload with zero errors and zero unfinished games.
- Store temporary benchmark reports and JSON under `.tmp/evaluation/evaluation_inference_profile/`.

---

### Task 1: Scheduler and CPU Observability

**Files:**
- Modify: `evaluation/runner/engine_pool_worker.py`
- Modify: `evaluation/performance_profile.py`
- Test: `tests/test_evaluation_engine_pool_worker.py`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Consumes: pool jobs, per-game remote inference connections, and official battle pointers.
- Produces: bounded inference-channel execution plus process-tree CPU core utilization.

- [ ] Add tests proving explicit session IDs preserve independent encoder state across reused connections and are closed exactly once.
- [ ] Add process-tree `/proc/<pid>/stat` CPU accounting sampled over the benchmark lifetime and report total CPU seconds, average utilized cores, and utilization percent.
- [ ] Add pool-worker counters for wall time, engine critical-section time, inference wait time, peak live games, and completed games.
- [ ] Run focused tests and retain the pre-change N16E8 result as the same-tree baseline.

### Task 2: Session-Aware Inference Channel Pool

**Files:**
- Modify: `evaluation/runner/engine_pool_worker.py`
- Modify: `evaluation/runner/inference_server.py`
- Modify: `evaluation/runner/batch.py`
- Test: `tests/test_evaluation_engine_pool_worker.py`
- Test: `tests/test_evaluation_inference_server.py`

**Interfaces:**
- Consumes: the existing `_run_pointer_game` contract, `pool_size`, and a configured inference-channel count per role.
- Produces: bounded reusable AF_UNIX connections carrying explicit per-game/per-role session IDs while retaining the existing implicit per-connection protocol for legacy workers.

- [ ] Write protocol tests for explicit session routing, close-session commands, legacy implicit sessions, and invalid session IDs.
- [ ] Implement a thread-safe connection lease pool; keep one synchronous request per leased channel and allow the next engine-ready thread to acquire it.
- [ ] Preserve failure isolation: one game result may fail without cancelling unrelated engines or leaking pointers.
- [ ] Run engine-pool, batch, worker, and inference-server regression tests.

### Task 3: Policy-0806 N16 E Scaling Benchmark

**Files:**
- Create: `.tmp/evaluation/evaluation_inference_profile/scheduler_n16_e8_*.json`
- Create: `.tmp/evaluation/evaluation_inference_profile/scheduler_n16_e16_*.json`
- Create: `.tmp/evaluation/evaluation_inference_profile/scheduler_n16_e32_*.json`
- Modify: `.tmp/evaluation/evaluation_inference_profile/summary.md`

**Interfaces:**
- Consumes: frozen Policy-0806, FP16, batch size 64, 2 ms wait, 16 workers, equal frozen schedule and game count.
- Produces: throughput, batch, latency, GPU-stage, RSS, true CPU utilization, completion, and error comparisons.

- [ ] Run a workload large enough to fill all environments for E8 and E16/I8; test E32 only if E16 is both stable and faster.
- [ ] Require completed games equal requested games, zero errors, and zero unfinished games for every accepted trial.
- [ ] Compare games/s and engine selections/s primarily; treat stochastic outcome and game length as non-causal.
- [ ] Recommend the smallest E within 10% of the best measured selection throughput and explicitly report whether 1000% average CPU was achieved.

### Task 4: Verification and Documentation

**Files:**
- Modify: `evaluation/README.md`
- Test: `tests/test_evaluation_engine_pool_worker.py`
- Test: `tests/test_evaluation_performance_profile.py`
- Test: `tests/test_evaluation_batch.py`

**Interfaces:**
- Consumes: the selected scheduler and benchmark evidence.
- Produces: documented N/E meaning, CPU metric semantics, recommended E, and a clean regression run.

- [ ] Document that pool E is concurrency per worker, not extra OS workers, and that true CPU utilization comes from process CPU time divided by wall time.
- [ ] Run `python3 -m unittest -v tests.test_evaluation_engine_pool_worker tests.test_evaluation_performance_profile tests.test_evaluation_batch tests.test_evaluation_worker tests.test_evaluation_inference_server`.
- [ ] Run `python3 -m py_compile evaluation/runner/engine_pool_worker.py evaluation/performance_profile.py` and `git diff --check`.

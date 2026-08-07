# Evaluation Engine Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and benchmark `N x E` official-engine evaluation, where each worker process owns `E` independent battle pointers and overlaps their resident-GPU inference waits.

**Architecture:** A new pool worker loads one official `cg` runtime, creates one pointer-safe engine adapter per game, and advances up to `E` game sessions concurrently. Each session owns two inference connections and therefore two independent causal encoder session IDs; while one session waits on GPU inference, other session threads advance their engine pointers. Existing one-game worker behavior remains the default and fallback.

**Tech Stack:** Python 3.11, ctypes official `libcg`, subprocess worker pools, threads for I/O overlap, AF_UNIX resident inference, unittest, Policy-0806 profiler.

## Global Constraints

- Do not modify `engine/source/` or packaged `cg` files.
- Every live environment owns exactly one `BattleStart` pointer and calls `BattleFinish` exactly once.
- Every game/player owns an independent inference connection and causal encoder session.
- Preserve existing `GameResult`, trace, metric, schedule, deck, seat, and report contracts.
- Pool mode initially requires shared resident inference for both players and rejects visualization.
- Write benchmarks below `.tmp/evaluation/evaluation_inference_profile/`.
- Fail closed on incomplete games, worker crashes, invalid actions, or engine cleanup failures.

---

### Task 1: Pointer-Safe Official Engine Adapter

**Files:**
- Create: `evaluation/runner/engine_pool_worker.py`
- Test: `tests/test_evaluation_engine_pool_worker.py`

**Interfaces:**
- Consumes: packaged `cg.sim.lib` functions `BattleStart`, `GetBattleData`, `Select`, `VisualizeData`, and `BattleFinish`.
- Produces: `_PointerBattle.start(deck0, deck1)`, `.select(action)`, `.finish()`, with pointer-local state and idempotent cleanup.

- [ ] **Step 1: Write a fake-lib isolation test with two pointers**

Assert alternating selects update only the selected pointer and both pointers are finished once.

- [ ] **Step 2: Run the test and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_engine_pool_worker`

Expected: import failure because the pool worker does not exist.

- [ ] **Step 3: Implement the pointer adapter**

Build the 120-card ctypes array, preserve the returned pointer, decode each pointer's `SerialData`, validate `Select` errors exactly like `cg.game`, and make cleanup idempotent.

- [ ] **Step 4: Prove real two-pointer isolation**

Use a copied official runtime package in read-only mode, start two valid games, alternate arbitrary legal actions, verify both observations progress independently, and finish both pointers.

### Task 2: Concurrent Pool Game Runner

**Files:**
- Modify: `evaluation/runner/engine_pool_worker.py`
- Test: `tests/test_evaluation_engine_pool_worker.py`

**Interfaces:**
- Consumes: a JSON payload containing multiple existing `GameRequest` payloads and trace paths plus two inference socket paths.
- Produces: one result JSON list with the existing serialized `GameResult` contract and one trace JSON per game.

- [ ] **Step 1: Write lifecycle and failure-isolation tests**

Cover `E=2`, independent inference connections, stable output order, one-session agent failure, and `BattleFinish` for every started pointer.

- [ ] **Step 2: Implement one pointer-local game loop**

Reuse the existing winner normalization, turn-limit, invalid-action forfeit, trace shape, and performance timing semantics without importing agent package code.

- [ ] **Step 3: Implement bounded concurrent execution**

Use a `ThreadPoolExecutor(max_workers=E)` inside the pool process. Each session blocks only its own thread on AF_UNIX inference; the resident server batches requests across all active sessions.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m unittest -v tests.test_evaluation_engine_pool_worker tests.test_evaluation_worker`

Expected: PASS.

### Task 3: Batch Scheduler Integration

**Files:**
- Modify: `evaluation/runner/batch.py`
- Test: `tests/test_evaluation_batch.py`

**Interfaces:**
- Consumes: `BatchConfig.engine_pool_size: int` where `1` preserves the legacy scheduler and `E>1` enables pool subprocesses.
- Produces: the same ordered iterator of `(GameRequest, trace_path, GameResult)` and manifest fields `engine_pool_size`, `worker_processes`, and `max_live_environments`.

- [ ] **Step 1: Write validation and grouping tests**

Reject `E<1`, visualization, missing shared inference, and arbitrary-legal mode for `E>1`; prove jobs are grouped into at most `workers` pool processes with at most `E` live games each.

- [ ] **Step 2: Implement pool subprocess invocation**

Write one request JSON per group, launch at most `workers` pool processes concurrently, apply a timeout scaled to group size, parse ordered results, and preserve worker-crash retry behavior by falling back to isolated retries.

- [ ] **Step 3: Add manifest and performance capacity fields**

Record actual OS worker processes separately from maximum live environments so throughput reports cannot confuse `N` and `N x E`.

- [ ] **Step 4: Run batch tests**

Run: `python3 -m unittest -v tests.test_evaluation_batch`

Expected: PASS.

### Task 4: Policy-0806 N x E Benchmark

**Files:**
- Modify: `evaluation/performance_profile.py`
- Modify: `.tmp/evaluation/evaluation_inference_profile/summary.md`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Consumes: CLI flags `--engine-pool-size E` and existing fixed Policy-0806 workload.
- Produces: profile JSON containing `worker_processes`, `engine_pool_size`, `max_live_environments`, RSS, stage timing, mean batch, request latency, selections/s, games/s, completion, and errors.

- [ ] **Step 1: Extend CLI and artifact schema tests**

Verify `E` reaches `BatchConfig` and is recorded in the immutable workload payload.

- [ ] **Step 2: Run equal-contract trials**

Run at least 64 games for `(N,E) = (16,1), (8,2), (4,4), (2,8), (2,10)`, keeping up to 16 or 20 live environments, then run higher concurrency when memory allows. Use the same schedule, seats, FP16, batch size 64, wait 2 ms, zero errors, and zero unfinished games.

- [ ] **Step 3: Select the smallest configuration within 10% of best throughput**

Compare per-selection engine, materialization, GPU, request latency, mean batch, and wall throughput. Treat game outcome/length variation separately.

- [ ] **Step 4: Update the evidence summary**

Document whether engine pooling improves GPU occupancy and formal evaluation throughput, its memory/process tradeoff, and the recommended defaults for evaluation versus rollout collection.

### Task 5: Regression Verification

**Files:**
- Test: `tests/test_evaluation_engine_pool_worker.py`
- Test: `tests/test_evaluation_batch.py`
- Test: `tests/test_evaluation_inference_server.py`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Consumes: all new and legacy execution paths.
- Produces: a clean test run and clickable official-engine reports.

- [ ] **Step 1: Run the complete relevant suite**

Run: `python3 -m unittest -v tests.test_evaluation_engine_pool_worker tests.test_evaluation_performance_profile tests.test_evaluation_inference_server tests.test_evaluation_worker tests.test_evaluation_batch tests.test_evaluation_frozen_0806_full_evaluation`

Expected: PASS.

- [ ] **Step 2: Run syntax and diff checks**

Run: `python3 -m py_compile evaluation/runner/engine_pool_worker.py evaluation/performance_profile.py evaluation/runner/batch.py && git diff --check`

Expected: PASS.

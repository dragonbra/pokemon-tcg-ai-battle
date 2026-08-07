# Parallel Feature Compiler Shards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the single-thread canonical feature-encoding funnel by compiling independent battle sessions in parallel CPU processes before one resident GPU batch inference service.

**Architecture:** Bind canonical feature compilation to each existing official-engine OS worker. Every game/player session owns an `OnlineCausalEncoder` in the same process that receives the engine observation, so the observation never crosses an extra compiler IPC boundary. The worker sends one raw canonical record plus minimal turn/selection control metadata to the resident GPU service; that service collates records once, performs GPU inference, and routes actions back. A server-side compiler-process prototype remains only as comparison evidence because its extra observation/record IPC limited C2 to a 3.69% gain.

**Tech Stack:** Python 3.11, multiprocessing spawn context and Pipes, canonical 0031 feature compiler, PyTorch, AF_UNIX inference service, official-engine Policy-0806 profiler, unittest.

## Global Constraints

- Never modify `engine/source/`, packaged policy code, model weights, the 39-key actor contract, action contract, reward, PPO loss, or official observation JSON.
- Parallelism is across independent battle/player sessions only; one session is strictly chronological and owned by one shard until close.
- Canonical record, collated tensor, logits, legal flag, deterministic action, trace, seed, and seat contracts must remain identical to serial compilation.
- Legacy/source-conditioned policies and unsupported compiler APIs retain the serial path.
- Compiler worker crashes, malformed replies, request-count mismatch, or session ownership errors fail closed; no silent serial fallback may contaminate a benchmark.
- Temporary benchmark reports belong below `.tmp/evaluation/evaluation_inference_profile/`.
- Formal evaluation remains `E=1` by default; this optimization is admitted only after official-engine parity and zero-error throughput evidence.

---

### Task 1: Worker-Local Session Compiler

**Files:**
- Modify: `evaluation/runner/engine_pool_worker.py`
- Test: `tests/test_evaluation_compiler_pool.py`

**Interfaces:**
- Produces one local raw-record encoder per game/player session inside each engine worker and a compiled inference request containing `session_id`, raw canonical `record`, and minimal `current/select` metadata.

- [ ] Add protocol tests proving local compilation receives the original observation, produces chronological raw records, and sends no full board/log observation to the GPU service.
- [ ] Add a GPU-service compiler-contract handshake containing the exact model config and canonical/persona-free capability; reject local compilation for incompatible or distinct policy roots.
- [ ] Cache immutable prototypes process-locally and create one encoder per game/player session without loading model weights or CUDA in engine workers.
- [ ] Record worker-local encode seconds/calls and compiled-request byte estimates separately from socket wait and engine time.

### Task 2: GPU Policy Server Integration

**Files:**
- Modify: `evaluation/runner/inference_server.py`
- Modify: `evaluation/runner/batch.py`
- Test: `tests/test_evaluation_inference_server.py`
- Test: `tests/test_evaluation_batch.py`

**Interfaces:**
- Adds a compiled-record request path to the resident server while retaining serial observation compilation for compatibility.

- [ ] Validate compiled records only for canonical persona-free policies and preserve fail-closed model legality.
- [ ] Skip parent encoding when a raw record is supplied; keep bounds checks, ability guard, one parent collate, GPU model, decode, and response completion unchanged.
- [ ] Preserve the serial observation path and server-side compiler prototype as diagnostic comparison modes.
- [ ] Extend inference and game performance payloads with compiler location, encode time, IPC payload path, collate, and GPU timings.

### Task 3: CLI and Profiler Contract

**Files:**
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/performance_profile.py`
- Modify: `evaluation/frozen_0806_full_evaluation.py`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Adds `worker_local_compiler` to `BatchConfig`, engine-pool payload, report manifest, performance-profile workload, and profile JSON.

- [ ] Keep worker-local compilation opt-in and require resident inference plus one identical canonical policy root for both roles.
- [ ] Record true process-tree CPU seconds/average cores, peak RSS, decisions/s, games/s, mean batch, p50/p95/p99 request latency, compiler IPC/encode, collate/H2D, and GPU model seconds.
- [ ] Ensure warm-up uses the same compiler-worker topology and reset clears both parent and child profile counters.

### Task 4: Exact Parity and Official-Engine Scaling

**Files:**
- Test: `tests/test_evaluation_compiler_pool.py`
- Create: `.tmp/evaluation/evaluation_inference_profile/compiler_n16_e8_c1_g256.json`
- Create: `.tmp/evaluation/evaluation_inference_profile/compiler_n16_e8_server_c2_g256.json`
- Create: `.tmp/evaluation/evaluation_inference_profile/compiler_n16_e8_worker_local_g256.json`
- Modify: `.tmp/evaluation/evaluation_inference_profile/summary.md`

**Interfaces:**
- Consumes frozen Policy-0806, 256 games, N16E8, FP16, batch size 64, 2 ms wait, seed 8062026, and the fixed schedule hash.
- Produces the fastest zero-error shard count and evidence explaining the next bottleneck.

- [ ] Compare serial and parallel raw records plus all 39 collated tensors on chronological multi-session fixtures; require exact equality.
- [ ] Run serial, server-C2 comparison, and worker-local C16 with 256/256 completed, zero errors, zero unfinished games, and identical seed/schedule/seat contracts.
- [ ] If worker-local compilation wins, sweep E and channel counts only until two consecutive throughput points regress.
- [ ] Report whether average CPU exceeded 400%/1000%, whether GPU batch/model utilization increased, and the measured speedup relative to the adjacent serial baseline.

### Task 5: Documentation and Regression Verification

**Files:**
- Modify: `evaluation/README.md`
- Test: `tests/test_evaluation_compiler_pool.py`
- Test: `tests/test_evaluation_inference_server.py`
- Test: `tests/test_evaluation_batch.py`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Produces an auditable explanation of session affinity, process ownership, defaults, failure behavior, and the recommended shard count.

- [ ] Document the compiler-shard topology and distinguish engine workers, environments per worker, compiler workers, GPU batch size, and PPO epochs.
- [ ] Run `python3 -m unittest -v tests.test_evaluation_compiler_pool tests.test_evaluation_inference_server tests.test_evaluation_batch tests.test_evaluation_performance_profile tests.test_evaluation_engine_pool_worker tests.test_evaluation_worker`.
- [ ] Run `python3 -m py_compile evaluation/runner/compiler_pool.py evaluation/runner/inference_server.py evaluation/runner/batch.py evaluation/performance_profile.py` and `git diff --check`.
- [ ] Verify `git diff --name-only -- engine/source` prints nothing.

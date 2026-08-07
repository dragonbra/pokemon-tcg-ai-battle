# 0035 Worker Compiler Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the 0035 lifetime-aware incremental compiler under the current fastest official-engine orchestration without changing Policy-0806 model or action semantics.

**Architecture:** Add an explicit worker-local compiler backend selector. The default keeps the frozen policy's stateless raw compiler; the research backend loads 0035's self-contained prototype, causal-knowledge, and `IncrementalCanonicalCompiler` modules inside each engine worker, emits the same canonical raw record, and sends it through the existing single IPC to the synchronous central B128 GPU collator/model server. The 0035 `PersistentTensorBank` is excluded from this bridge because loading PyTorch in 16 engine workers previously caused a 14 GB process tree and tensor views are not an appropriate cross-process contract.

**Tech Stack:** Python 3.11, 0035 pure-Python compiler, worker-local causal sessions, AF_UNIX raw-record protocol, resident PyTorch GPU inference, official-engine Policy-0806 profiler, unittest.

## Global Constraints

- Never modify `engine/source/`, Policy-0806 frozen packages, checkpoint weights, model structure, schema `0031_rule_faithful_semantic_decision_v2`, the 39-key tensor contract, or the action contract.
- Do not modify 0035 implementation files owned by the other session; bridge them from `evaluation/` only.
- Every engine worker must remain torch-free; importing the 0035 backend must fail closed if `torch` enters `sys.modules`.
- One game/player session owns one chronological incremental compiler and one causal knowledge instance until close.
- Default evaluation behavior remains the stateless policy compiler; 0035 is opt-in pending official-engine admission.
- Compare on N16E6/B128/1ms, synchronous H2D, FP16, 256 games, seed 8062026, and the fixed schedule hash.

---

### Task 1: Backend Adapter

**Files:**
- Modify: `evaluation/runner/engine_pool_worker.py`
- Test: `tests/test_evaluation_engine_pool_worker.py`

**Interfaces:**
- Adds backend IDs `policy_stateless` and `0035_incremental` to `_WorkerLocalCompilerFactory`.
- Produces the existing `encode_record(observation) -> canonical_record` interface.

- [x] Add a clean-subprocess test loading each backend and asserting `torch` is absent from `sys.modules`.
- [x] Install lightweight package boundaries for 0035 `contracts` and `features` so importing submodules does not execute tensor/model package initializers.
- [x] Load 0035's own immutable prototypes and causal knowledge, instantiate one `IncrementalCanonicalCompiler` per session, and preserve the existing bounds/deck/actor validation.
- [x] Add a chronological fixture parity test requiring exact Python-record equality between `policy_stateless` and `0035_incremental`.

### Task 2: Configuration and Audit Contract

**Files:**
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/runner/engine_pool_worker.py`
- Modify: `evaluation/performance_profile.py`
- Test: `tests/test_evaluation_batch.py`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Adds `worker_compiler_backend: str = "policy_stateless"` and CLI `--worker-compiler-backend`.
- Records the backend in engine-pool payloads, report manifests, and benchmark JSON.

- [x] Reject unknown backend IDs, 0035 without `worker_local_compiler`, and any backend combined with server compiler shards.
- [x] Pass the backend unchanged into every engine-pool subprocess and record it in all audit payloads.
- [x] Preserve `policy_stateless` as the default for every existing caller.

### Task 3: Official-Engine Admission Benchmark

**Files:**
- Create: `.tmp/evaluation/evaluation_inference_profile/compiler_0035_n16_e6_b128_w1_g256.json`
- Modify: `.tmp/evaluation/evaluation_inference_profile/summary.md`
- Modify: `evaluation/README.md`

**Interfaces:**
- Compares the established 557.45 selections/s stateless baseline against 0035 incremental compilation under an otherwise identical contract.

- [x] Run a four-game official-engine smoke and require the same 924 selections, 4/4 completion, and zero errors.
- [x] Run 256 games and require 256/256 completion, zero errors, and no torch import in workers.
- [x] Report wall throughput, CPU, RSS, compiler seconds/calls, batch distribution, p50/p95 request latency, collate/H2D, and GPU model time.
- [x] Keep 0035 opt-in unless exactness passes and measured throughput improves outside ordinary run noise.
- [x] Run the 0035 suite plus all relevant evaluation tests, `py_compile`, `git diff --check`, and confirm `engine/source/` remains untouched.

### Measured outcome

The bridge passed exactness and official-engine admission, but did not establish an end-to-end win.
The adjacent stateless control measured 541.04 selections/s and 1.280 ms/compiler call; 0035 measured
546.16 selections/s and 1.254 ms/call. The +0.9% wall result is inside normal run variation because
the earlier identical stateless baseline reached 557.45 selections/s. The 0035 backend therefore
remains opt-in and the production default remains `policy_stateless`.

# Production 256-Lane CPU/CUDA Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify decision-level parity between the canonical CPU engine and the production 256-lane CUDA collector, including one lane reuse/reset cycle.

**Architecture:** Add a default-disabled diagnostic callback at the resident decision boundary, then drive the existing `ChunkedCudaRolloutCollector` with the V2 production topology. A temporary harness builds canonical CPU reference traces and compares compact hashes for every decision, stopping at the first mismatch.

**Tech Stack:** Python 3.11, PyTorch CUDA, official CPU runtime, `_ptcg_cuda`, JSON diagnostic artifacts.

## Global Constraints

- Do not modify `engine/source/`, model structure, Frozen opponent protocol, Champion V1, optimizer, or game semantics.
- Do not start PPO training or fix unrelated static-audit findings.
- Use update-270, deterministic greedy selection, identical per-game seeds, 256 CUDA lanes, and a 512-game production chunk.
- A pass requires exact per-decision actor, observation, legal action, selected action, and terminal parity, not only equal win/loss.
- Stop at the first divergence and do not expand the sample after failure.

---

### Task 1: Default-Off Resident Trace Hook

**Files:**
- Modify: `engine_cuda/python/ptcg_cuda_engine/semantic0031_resident.py`
- Modify: `train/0040_dragapult_0809_action_boundary_rl/rollout/cuda_collector.py`

**Interfaces:**
- Consumes: existing resident ready masks, semantic tensors, routed/bypassed actions, lane/job mapping, and decision indices.
- Produces: optional `decision_trace_sink(**payload)` callback forwarded by the production collector.

- [ ] Add an optional callback argument with a `None` default.
- [ ] Invoke it after final action assembly and before applying the action.
- [ ] Forward the callback through `CudaFullSemanticRolloutCollector`; leave normal collection unchanged.
- [ ] Run focused collector/resident unit tests.

### Task 2: Temporary Canonical/Production Harness

**Files:**
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/production_256_lane/run.py`

**Interfaces:**
- Consumes: update-270 model, full Frozen Policy-0806, official CPU runtime, production CUDA rules/extension, and fixed seeds.
- Produces: compact per-game parity reports with first-divergence evidence.

- [ ] Generate a deterministic fixed 512-seed manifest whose first four entries are the original regression seeds.
- [ ] Build canonical CPU traces with actor, observation hash, legal hash, action, next-boundary hash, and terminal result.
- [ ] Run CUDA through `ChunkedCudaRolloutCollector` with `rollout_batch_size=512` and `lane_count=256`.
- [ ] Compare in game/decision order and classify the first mismatch as A/B/C/D/E.

### Task 3: Ordered Gates

**Files:**
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/production_256_lane/test0_original_four.json`
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/production_256_lane/test1_first_use_256.json`
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/production_256_lane/test2_reuse_512.json`

**Interfaces:**
- Consumes: Task 2 harness.
- Produces: final counts and first-divergence record.

- [ ] Run Test 0 through the production collector and require 4/4 exact.
- [ ] If Test 0 passes, run the fixed first-use 256 set and require 256/256 exact.
- [ ] If Test 1 passes, run all 512 games in one 256-lane resident invocation and report first-use/reused-lane counts separately.
- [ ] Verify report invariants and preserve only compact evidence.

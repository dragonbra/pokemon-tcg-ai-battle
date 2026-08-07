# Evaluation Inference Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and then improve Policy-0806 official-engine evaluation throughput without changing engine, action, seed, deck, or policy semantics.

**Architecture:** Add opt-in timing counters at the two existing component boundaries: official-engine workers and the resident GPU inference server. Establish an arbitrary-legal-action engine ceiling, add only a provably forced-action shortcut initially, and use equal-contract measurements to decide whether a multiplexed `N x E` engine-session scheduler and incremental materialization are justified.

**Tech Stack:** Python 3.11, multiprocessing/AF_UNIX IPC, PyTorch CUDA events, official `cg` runtime, unittest, Frozen-0806 evaluation assets.

## Global Constraints

- Do not modify `engine/source/`; all strength and throughput runs use the official engine runtime.
- Write all diagnostic artifacts below `.tmp/evaluation/evaluation_inference_profile/`.
- Keep Policy-0806 checkpoint, exact decks, seeds, seat balance, turn limit, and legal action contract fixed across comparable trials.
- A forced-action shortcut is legal only when `len(options) == minCount == maxCount == 1`.
- Existing reports and immutable Frozen-0806 package identities must not be overwritten.
- Preserve per-game causal encoder state and per-game official engine state under every concurrency mode.

---

### Task 1: Opt-In Timing Contract

**Files:**
- Modify: `evaluation/runner/models.py`
- Modify: `evaluation/runner/worker.py`
- Modify: `evaluation/traces/store.py`
- Test: `tests/test_evaluation_worker.py`
- Test: `tests/test_evaluation_batch.py`

**Interfaces:**
- Consumes: existing `GameRequest`, `GameResult`, `battle_start`, `battle_select`, and local/remote agent call paths.
- Produces: `GameResult.performance: dict[str, float | int] | None` with engine start/select, policy wait, selection count, and worker wall-clock totals.

- [x] **Step 1: Add failing serialization and aggregation tests**

Create worker tests asserting non-negative timing fields and batch tests asserting per-game performance survives the lightweight report record.

- [x] **Step 2: Run the focused tests and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_worker tests.test_evaluation_batch`

Expected: failure because `GameResult` and report records do not expose performance timing.

- [x] **Step 3: Instrument official-engine boundaries**

Use `time.perf_counter_ns()` around `battle_start`, each selected-agent call, `battle_select`, and total `run_game`. Store integer nanoseconds internally and publish seconds plus counts once per game. Do not time trace rendering as engine work.

- [x] **Step 4: Preserve timing in report records**

Copy the optional performance mapping into `TraceStore.write_game_record`; aggregate only additive counters in the benchmark command rather than changing existing strength metrics.

- [x] **Step 5: Run focused tests**

Run: `python3 -m unittest -v tests.test_evaluation_worker tests.test_evaluation_batch`

Expected: PASS.

### Task 2: Resident Inference Profiler And Forced Shortcut

**Files:**
- Modify: `evaluation/runner/inference_server.py`
- Test: `tests/test_evaluation_inference_server.py`

**Interfaces:**
- Consumes: `_InferenceRequest`, `OnlineCausalEncoder.encode`, `_stack_batches`, and `deterministic_action_tensors`.
- Produces: `PolicyServer.profile_snapshot() -> dict[str, object]`; control request `{"command": "profile"}`; strict `_forced_action(observation) -> list[int] | None`.

- [x] **Step 1: Write forced-action boundary tests**

Cover required singleton (`minCount=maxCount=1`), optional singleton (`minCount=0,maxCount=1`), zero-option, and multi-option observations.

- [x] **Step 2: Write profile-schema tests**

Use fake encoder/model objects to assert request count, shortcut count, batch histogram, encoder initialization, encode, collate, model, and decode counters are present and non-negative.

- [x] **Step 3: Run tests and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_inference_server`

Expected: failure because the shortcut and profile contract do not exist.

- [x] **Step 4: Implement timing with CUDA synchronization boundaries**

Record CPU stages with `perf_counter_ns`. For CUDA model time, use CUDA events and synchronize only when profiling is enabled; keep the default non-profile path free of added synchronizations. Report request-latency percentiles from a bounded histogram rather than storing observations.

- [x] **Step 5: Implement the forced-action shortcut**

Before feature encoding, return `[0]` only for a required singleton. Maintain causal knowledge correctness by making the shortcut opt-in for the first benchmark; do not enable it by default until parity confirms that skipped observations do not contain causal log updates needed by the next encoded state.

- [x] **Step 6: Run focused tests**

Run: `python3 -m unittest -v tests.test_evaluation_inference_server`

Expected: PASS.

### Task 3: Policy-0806 Profile And Engine Ceiling Command

**Files:**
- Create: `evaluation/performance_profile.py`
- Modify: `evaluation/frozen_0806_full_evaluation.py`
- Test: `tests/test_evaluation_performance_profile.py`

**Interfaces:**
- Consumes: Frozen-0806 runtime catalog, `BatchConfig`, resident inference server profile control, and per-game performance records.
- Produces: `.tmp/evaluation/evaluation_inference_profile/<trial>.json` with immutable workload identity, completion contract, wall throughput, worker timing, inference timing, and GPU/device metadata.

- [x] **Step 1: Write aggregation tests**

Assert additive timing aggregation, derived residual/overlap values, zero-division handling, and fail-closed rejection of incomplete/error trials.

- [x] **Step 2: Add `profile` benchmark mode**

Run the same even, seat-balanced Frozen-0806 subset with configurable workers, games, batch size, coalescing, dtype, forced shortcut, and output path. Query the resident server after workers finish and atomically write one JSON artifact.

- [x] **Step 3: Add arbitrary-legal engine ceiling mode**

Use `list(range(minCount))` for every non-initial observation and exact deck return for initialization. Run both sides through official engine workers with the same decks, seeds, seats, turn limit, trace behavior, and completion checks while bypassing feature materialization and model inference.

- [x] **Step 4: Run tests**

Run: `python3 -m unittest -v tests.test_evaluation_performance_profile`

Expected: PASS.

### Task 4: Equal-Contract Measurements And Root-Cause Decision

**Files:**
- Create: `.tmp/evaluation/evaluation_inference_profile/policy0806_baseline.json`
- Create: `.tmp/evaluation/evaluation_inference_profile/policy0806_forced_shortcut.json`
- Create: `.tmp/evaluation/evaluation_inference_profile/arbitrary_legal_engine_ceiling.json`
- Create: `.tmp/evaluation/evaluation_inference_profile/summary.md`

**Interfaces:**
- Consumes: Task 3 CLI and RTX 5080/16-core host.
- Produces: measured stage shares and a go/no-go decision for `N x E`, incremental features, static cache, and forced shortcut deployment.

- [x] **Step 1: Run warm-up outside measured trials**

Warm model load and CUDA kernels using two games; exclude this wall time from trial metrics.

- [x] **Step 2: Run baseline and repeat once**

Use at least 32 complete games, 16 workers, batch size 64, 2 ms coalescing, FP16, fixed seed `8062026`, balanced seats, and zero errors/unfinished games.

- [x] **Step 3: Run forced shortcut with the identical workload**

Require identical game count, deck sequence, seeds, seats, completion, and error counts. Record shortcut frequency and any action/result divergence separately from throughput.

- [x] **Step 4: Run arbitrary-legal ceiling**

Use the same official engine workload identity and report engine selections/s and games/s. Treat strength and episode length differences as expected; compare per-selection engine cost, not only games/s.

- [x] **Step 5: Write the evidence summary**

Report engine, materialization, collate/transfer, GPU model, IPC/wait, and residual shares. Recommend `N x E` only if GPU starvation/worker wait dominates after controlling for CPU saturation. Recommend incremental materialization only if repeated encode cost is a material share and parity tests can cover causal state updates.

### Task 5: Multiplexed Engine Sessions (Conditional)

**Files:**
- Create: `evaluation/runner/engine_pool_worker.py`
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/runner/inference_server.py`
- Test: `tests/test_evaluation_engine_pool_worker.py`

**Interfaces:**
- Consumes: official `BattleStart`, `GetBattleData`, `Select`, and `BattleFinish` pointer-based C API; inference requests carrying explicit `session_id`.
- Produces: one process hosting `E` independent `battle_ptr` states and scheduling whichever environment is CPU-ready while other environments await GPU responses.

- [x] **Step 1: Gate implementation on Task 4 evidence**

Decision: do not implement for formal evaluation. Sixteen isolated workers reached 90.9% of
the best measured 32-worker selection throughput, while multiple live games per worker would
violate the formal one-game-per-process isolation contract. Reconsider this architecture for
rollout collection, where the ownership contract can be designed explicitly.

Proceed only when measured inference wait is material and memory/RSS projections allow at least `E=2` without reducing completion reliability.

- [ ] **Step 2: Prove two-pointer isolation**

Start two battles in one process, alternate legal selections, and verify each observation/turn/hash progresses independently before implementing scheduling.

- [ ] **Step 3: Add explicit inference session routing**

Carry a stable game/player session ID per request so causal encoder state cannot leak between pooled environments sharing one socket.

- [ ] **Step 4: Implement bounded round-robin scheduling**

Maintain at most `E` live states per process, send inference asynchronously, advance other ready states, and close every engine pointer and inference session on success, error, timeout, or cancellation.

- [ ] **Step 5: Verify parity and scale E**

Run `E=1` parity first, then benchmark `E=2,4,8,10` under fixed total games and CPU affinity. Select the smallest E within 10% of best zero-error throughput.

### Task 6: Incremental Materialization (Conditional)

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/deployment/online_runtime.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/features/compiler.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_deployment.py`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.html`

**Interfaces:**
- Consumes: chronological observations and `CausalKnowledge.consume` deltas.
- Produces: exact-equivalent cached static prototype/deck tensors plus dirty-region updates for dynamic board, event, resource, and legal-option tensors.

- [x] **Step 1: Gate implementation on Task 4 evidence**

Decision: proceed as a separate follow-up project. Full rebuild materialization remains about
1.4 ms per decision after static caching. Any incremental implementation must first freeze
chronological official-engine fixtures and prove exact parity for every canonical tensor.

Proceed only if repeated feature encode is a material end-to-end share after static prototype loading is cached.

- [ ] **Step 2: Freeze full-rebuild parity fixtures**

Collect chronological official-engine trajectories covering setup, main action, search, evolution, ability, attack, KO replacement, and terminal states. Save only test fixtures permitted by repository policy.

- [ ] **Step 3: Split immutable and dynamic state**

Cache prototype indexes and exact-deck manifest/tensors once per policy/deck. Keep observations, legal options, causal event memory, visible zones, resource beliefs, and selection relations dynamic.

- [ ] **Step 4: Implement dirty-region updates**

Update only regions whose source fields changed, but fall back to full rebuild when identity, ordering, visibility, or causal event deltas cannot be proven local.

- [ ] **Step 5: Require tensor-exact trajectory parity**

Compare all 39 actor tensor keys, masks, decoded legal actions, and causal snapshots at every decision against the original full rebuild before measuring speed.

- [ ] **Step 6: Synchronize authoritative design documents**

Document cache ownership, invalidation, tensor equivalence, fallback conditions, benchmark results, and the unchanged actor-visible schema in both DESIGN files.

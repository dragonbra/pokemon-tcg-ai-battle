# 0020 Hybrid CPU/CUDA Engine Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with the real 0020 rollout topology, whether any currently
implemented CUDA primitive can safely replace official CPU-engine work and
produce a measurable end-to-end speedup.

**Architecture:** Keep `engine/source/` and formal evaluation untouched. First
make wall-time attribution reproducible, then test the only semantically narrow
candidate (`official_setup`) in a separately named research runtime. Do not
attempt per-selection CPU/GPU opcode ping-pong; further CUDA work is admitted
only if a coarse boundary can own a complete `battle_select` progression for a
closed-world deck/opponent set.

**Tech Stack:** Python 3.11, C++20, CUDA 12.8, PyTorch, ctypes, standard library
`unittest`, official engine runtime.

## Global Constraints

- Never modify `engine/source/` without stopping and obtaining a second explicit
  confirmation from the user after explaining loss of official-runtime status.
- Formal strength evaluation always uses the unmodified official runtime.
- Main-agent inference stays centralized on CUDA; opponents stay in CPU workers
  for the benchmark requested here.
- Compare identical decks, job schedule, worker count, action mode, coalescing
  window, model checkpoint, and machine load.
- The accepted RNG implementation difference does not waive parity for legal
  actions, state transitions, rewards, terminal results, or RNG consumption sites.
- Unsupported CUDA behavior fails closed; never resume an official battle from
  a partially represented CUDA state.
- Temporary reports stay under `.tmp/evaluation/`; tracked conclusions belong
  under `experiments/0020_pluggable_deck_rl/`.
- Do not commit or push unless the user explicitly requests it.

## Current Evidence And Stop Bound

The 2026-07-30 profile used 100 games, 16 official-engine workers, the fixed
Alakazam CPU opponent, the centralized Dragapult CUDA learner, sample mode, and
a 2 ms coalescing window. It completed 100/100 games in 85.182 seconds.

| Surface | Measured cost |
|---|---:|
| `battle_start` | 0.301 s aggregate; 3.012 ms/game |
| `battle_select` including observation JSON | 4.939 s aggregate; 0.355 ms/selection |
| CPU opponent inference | 221.725 s aggregate; 2.217 s/game |
| CPU opponent load | 139.812 s aggregate; 1.398 s/game |
| Main-agent feature encode | 7.461 s parent wall time |
| Main-agent collate + H2D | 1.245 s parent wall time |
| Main-agent CUDA action sampling | 29.846 s parent wall time |
| Trajectory D2H | 0.319 s parent wall time |

Worker aggregates overlap across 16 processes and must not be divided directly
by the run wall time. The official calls averaged 52.4 ms against 11.070 seconds
of worker wall time per game, or about 0.47% of the per-game worker path; the
largest per-game official-call total was 86.7 ms. Under ideal 16-way overlap,
their estimated direct contribution to the 85.182-second makespan is about
`(4.939 + 0.301) / 16 = 0.328` seconds, or 0.4%. This is an estimate rather than
a strict upper bound because scheduling imbalance and CPU contention can change
the observed makespan; only a second end-to-end arm can establish actual speedup.

---

### Task 1: Make The Real-Topology Timing Contract Reproducible

**Files:**
- Create: `train/0020_pluggable_deck_rl/rl/observability/timing.py`
- Modify: `train/0020_pluggable_deck_rl/rl/rollout/worker.py`
- Modify: `train/0020_pluggable_deck_rl/rl/rollout/collector.py`
- Create: `train/0020_pluggable_deck_rl/tests/test_rl_timing.py`
- Create: `.tmp/evaluation/0020_engine_accel_rl_profile/official_cpu_profile.json`

**Interfaces:**
- Produces: `TimingAccumulator.add(name: str, seconds: float) -> None` and a
  per-run JSON payload separating worker aggregate CPU time from parent wall time.
- Consumes: the existing `RolloutCollector` and `run_engine_episode` paths.

- [ ] **Step 1: Add a failing test for non-overlapping timing namespaces**

  Assert that output contains `worker/battle_start`, `worker/battle_select`,
  `worker/opponent_load`, `worker/opponent_inference`, `worker/candidate_wait`,
  `parent/feature_encode`, `parent/collate`, `parent/h2d`, `parent/learner`, and
  `parent/trajectory_d2h`, and labels aggregate CPU seconds separately from wall
  seconds.

- [ ] **Step 2: Run the timing test and confirm it fails**

  Run `python3 -m unittest -v train.0020_pluggable_deck_rl.tests.test_rl_timing`.

- [ ] **Step 3: Add opt-in timers around existing calls**

  Use `time.perf_counter()` and a disabled-by-default collector option. Do not
  change actions, observations, worker scheduling, or model calls. Include
  `BattleStart`, `Select + GetBattleData`, JSON decode, policy loading, policy
  execution, encoding, collation, transfer, sampling, and trajectory export.

- [ ] **Step 4: Verify timing does not change the rollout contract**

  Run the existing seven RL policy/training tests plus the new timing test.

- [ ] **Step 5: Run three 100-game profiles**

  Use the fixed Alakazam/Dragapult matchup, 16 workers, 2 ms coalescing, and both
  one greedy run and two explicitly torch-seeded sample runs. Report median and
  range for games/s and normalized milliseconds per decision.

- [ ] **Step 6: Record an Amdahl estimate and empirical confidence interval**

  Calculate best-case speedup for setup-only, all official calls, learner-only,
  opponent-load-only, and feature-encode-only removal. Label estimates separately
  from measured A/B speedup. Stop engine integration if the all-official-call
  estimate is below 3% and no contention experiment shows a larger effect.

### Task 2: Classify Current CUDA Components Against The Official ABI

**Files:**
- Create: `experiments/0020_pluggable_deck_rl/cuda_hybrid_component_matrix.json`
- Modify: `experiments/0020_pluggable_deck_rl/CUDA_ENGINE_AUDIT.md`
- Test: `engine_cuda/tests/test_schema.py`

**Interfaces:**
- Produces: one machine-readable row per CUDA component with `semantic_scope`,
  `official_import`, `official_export`, `0020_reachable`, `hot_path`, and
  `admission` fields.
- Consumes: the standard official ABI and imported CUDA commit hash.

- [ ] **Step 1: Record the official ABI boundary**

  Confirm that the callable runtime surface is opaque `BattleStart`, `Select`,
  `GetBattleData`, and `BattleFinish`, with no complete state import/export,
  effect-stack import/export, or RNG-position API.

- [ ] **Step 2: Classify `official_setup`**

  Mark it semantically useful for seeded shuffle, opening hands, and mulligans,
  but not directly insertable because it cannot construct the official opaque
  battle state. Mark expected throughput value as setup-only and non-hot-path.

- [ ] **Step 3: Classify `official_status_core`**

  Mark it rejected for 0020: it is a separate hard-coded Venipede/Swablu fixture,
  not a callable implementation of general poison, confusion, or KO logic.

- [ ] **Step 4: Classify generic opcodes**

  Mark `END_TURN`, `CHECK_KNOCKOUT`, and `MOVE_CARD` rejected because their state
  transitions are smoke semantics and cannot round-trip an official battle.

- [ ] **Step 5: Classify codec and policy routing**

  Mark them as future device-resident infrastructure, not official-engine
  replacements. Record the current 128-entity/80-option/6-category mismatch
  against the 0020 192/128/7 contract.

- [ ] **Step 6: Assert the current admission result**

  The matrix must fail if any current component is labelled production-safe for
  arbitrary 0020 `battle_select`. The expected current result is zero hot-path
  engine replacements and one setup research candidate.

### Task 3: Build A Setup-Only Research Spike Behind A Separate Runtime Name

**Files:**
- Create: `engine_hybrid/PROVENANCE.md`
- Create: `engine_hybrid/include/hybrid_setup_bridge.h`
- Create: `engine_hybrid/src/hybrid_setup_bridge.cpp`
- Create: `engine_hybrid/tests/setup_bridge_test.cpp`
- Create: `engine_hybrid/CMakeLists.txt`

**Interfaces:**
- Produces: a research-only `HybridBattleStart` that constructs a CPU battle from
  CUDA-produced setup results without ever being loaded as the official runtime.
- Consumes: exact deck arrays, explicit seed/RNG state, and `official_setup`
  output records.

- [ ] **Step 1: Stop for explicit authorization**

  Explain that constructing official internal state requires a separately named
  fork or bridge derived from official source and therefore is not the official
  runtime. Do not copy or alter source until the user explicitly confirms this
  research boundary.

- [ ] **Step 2: Write setup record round-trip tests**

  Cover deck order, both hands, mulligan counts, Prize order, chosen Active and
  Bench serials, starting player, RNG consumption index, and first legal options.

- [ ] **Step 3: Implement one batched CUDA setup call**

  Batch many episode resets in one launch. Reject batch-one per-game launch as the
  production path. Preserve every card instance serial and zone position needed
  to construct the CPU continuation.

- [ ] **Step 4: Construct a fresh CPU continuation state**

  Import the complete setup record once, before the first decision. There must be
  no mid-game state migration and no CUDA call inside `battle_select`.

- [ ] **Step 5: Run 10,000-seed differential setup parity**

  Require identical initial legal options, public observation, hidden-state
  digest, and first 10 official CPU transitions under a fixed legal-action
  schedule, subject only to the approved RNG generator difference.

- [ ] **Step 6: Benchmark the real 100-game topology**

  Compare unmodified official CPU setup with hybrid batched setup. Require 0
  errors, zero semantic mismatches, and at least 3% median end-to-end speedup over
  three runs. The current profile predicts that this gate will fail.

- [ ] **Step 7: Remove the spike from the rollout path if the gate fails**

  Keep parity evidence and provenance, but do not add a permanent engine selector
  or training config for a non-beneficial runtime.

### Task 4: Admit Only A Coarse Full-Selection CUDA Boundary

**Files:**
- Modify: `docs/superpowers/plans/2026-07-30-0020-cuda-engine-parity-gate.md`
- Modify: `engine_cuda/include/ptcg_cuda/state_layout.cuh`
- Modify: `engine_cuda/src/engine_kernels.cu`
- Create: `engine_cuda/tools/run_step_parity.py`
- Create: `engine_cuda/tests/test_step_parity.py`

**Interfaces:**
- Produces: a CUDA environment that owns reset through terminal for a declared
  closed-world matchup, or owns at least an entire `battle_select` progression
  without returning intermediate rule state to CPU.
- Consumes: a complete POD state/rule pack, explicit RNG state, and normalized
  actions.

- [ ] **Step 1: Reject per-opcode CPU/GPU offload**

  Add an architecture test or review gate forbidding a loop of CPU official
  state export, H2D, one CUDA opcode, D2H, official state import, and host sync.

- [ ] **Step 2: Implement the complete reachable rule surface on a CPU POD mirror**

  Cover every card/effect reachable by the promoted candidate and opponents,
  including legal options, selections, turn budgets, evolution timing, attack
  commit, KO/Prize/replacement, checkup, deck-out, and terminal results.

- [ ] **Step 3: Establish decision-level parity before CUDA execution**

  Require zero unexplained mismatch across at least 100,000 seeded official/POD
  decisions and targeted fixtures for every reachable effect.

- [ ] **Step 4: Run the same POD transition on CUDA**

  Keep environment state, codec tensors, learner action, next state, reward, and
  terminal flags on device. Host interaction is limited to reset scheduling,
  rollout export, metrics, and explicit debug barriers.

- [ ] **Step 5: Promote only after an end-to-end throughput gate**

  Compare full rollout plus learner training on identical hardware. Require at
  least 3x end-to-end throughput for the full CUDA environment, since the
  partial-engine profile shows that micro-offload cannot justify its complexity.

### Task 5: Optimize The Actual Current Bottlenecks Separately

**Files:**
- Modify: `train/0020_pluggable_deck_rl/rl/rollout/collector.py`
- Modify: `train/0020_pluggable_deck_rl/rl/rollout/worker.py`
- Modify: `train/0020_pluggable_deck_rl/rl/opponent_inference/resident_pool.py`
- Create: `train/0020_pluggable_deck_rl/tests/test_rollout_worker_pool.py`

**Interfaces:**
- Produces: persistent official-engine worker processes, cached CPU opponent
  packages, and unchanged centralized learner inference.
- Consumes: the existing `RolloutJob` and `Episode` contracts.

- [ ] **Step 1: Reuse worker processes across episodes**

  Reset and finish one official battle per job while keeping imported opponent
  modules and model weights resident. Clear only battle-local agent state.

- [ ] **Step 2: Prove episode isolation**

  Test that effect history, encoder history, random seeds, cg battle pointers,
  and module-local mutable state cannot leak between sequential jobs.

- [ ] **Step 3: Benchmark cached CPU opponents**

  Compare against the same three-run 100-game contract. The measured 1.398
  seconds/game load cost makes this a materially stronger candidate than setup
  CUDA offload.

- [ ] **Step 4: Tune learner batching and feature encoding independently**

  Sweep workers and coalescing window, then profile feature encoding and H2D.
  Do not label these improvements as engine CUDA acceleration.

- [ ] **Step 5: Update the 0020 design documents if rollout semantics change**

  Synchronize `experiments/0020_pluggable_deck_rl/DESIGN.md` and `DESIGN.html`
  when worker lifetime, opponent state/reset semantics, feature flow, or rollout
  contracts change.

## Recommended Decision

Do Task 1 and Task 2. Do not start Task 3 unless the user explicitly authorizes
a non-official research fork, and expect its speed gate to fail. Prioritize Task
5 for near-term 0020 throughput. Continue the separate full CUDA parity plan
only if the strategic goal is device-resident mass environments rather than
accelerating the current opaque official-engine calls.

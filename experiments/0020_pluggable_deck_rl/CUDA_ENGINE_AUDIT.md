# 0020 CUDA Engine Semantic Audit

**Audit date:** 2026-07-30
**Imported source:** `engine_cuda/` from commit
`bf56dfb4b1d58279f0cd1e49a646b46174d8a589`
**Decision:** **not admissible as the 0020 rollout or evaluation engine**

## Executive conclusion

The imported code is a useful CUDA architecture prototype, but it is not a
semantic rewrite of the official engine. It currently proves that fixed-size
device state, a small numeric interpreter, policy routing, codec-shaped buffers,
and controlled setup/status kernels can run on CUDA. It does not generate or
execute complete official legal-action transitions for arbitrary decks.

The official engine remains the only runtime allowed to produce 0020 training
strength claims and formal evaluation results. `engine_cuda/` may become an
accelerated rollout environment only after step-level differential promotion.
The accepted RNG exception is limited to how randomness is seeded/generated;
it does not waive parity for RNG consumption order, reachable selections,
state transitions, rewards, or terminal results.

## Blocking findings

### 1. The generic interpreter does not implement official battle semantics

The generic engine exposes 15 opcodes including `HALT`, while the official card
snapshot contains 1,556 card IDs and a much larger state/effect surface. Major
official behaviors absent from the generic interpreter include legal option
generation, setup/mulligan, turn draw and deck-out, Energy requirements,
evolution timing, Trainer/Ability/Stadium rules, retreat, Bench replacement,
Prize handling, weakness/resistance, special conditions, effect stacks, copied
attacks, and card-specific effects.

Several implemented opcode names are only smoke semantics:

- `END_TURN` increments a counter, toggles the actor, and returns to policy. It
  does not run official checkup, clear per-turn budgets, draw, or deck-out.
- `CHECK_KNOCKOUT` immediately declares the other player the winner when the
  Active reaches zero HP. It does not discard the stack, take Prizes, select a
  replacement Active, or continue when a Bench exists.
- `MOVE_CARD` changes one zone byte without maintaining zone counts, attachment
  topology, slot order, triggers, or visibility.
- Reset creates two synthetic Active entities from counts supplied by the host;
  it does not construct a 120-card battle and execute official setup.

These are deliberate vertical-slice choices, not small edge-case differences.

### 2. The strongest parity kernel is a separate hard-coded micro-engine

`benchmarks/official_status_core.cu` hard-codes Venipede `535`, Swablu `197`,
Poison Spray `765`, Disarming Voice `266`, their HP, and controlled decks of
exactly four expected Basic Pokemon plus 56 Energy. It does not call the generic
opcode interpreter. Its historical 18,485-decision result therefore validates a
targeted setup/poison/confusion/KO fixture, not arbitrary card execution.

Likewise, `official_setup.cu` validates a specific libstdc++ shuffle contract and
mulligan draw order. Setup parity is valuable, but it does not establish main
game parity.

### 3. The current device codec cannot represent the 0020 model contract

The CUDA constants are `kMaxCodecEntities=128` and `kMaxCodecOptions=80` with
entity categorical width 6. The frozen 0020 model requires up to 192 entities,
128 options, and entity categorical width 7. Connecting the current buffers to
0020 would either raise named overflow errors or silently require a different
feature contract. Silent truncation is forbidden.

### 4. There is no locally reproducible complete official/CUDA paired replay

The committed comparison tools require a custom `libcg_seeded.so` exporting
`BattleStartSeeded`, state digests, tensors, option records, and batch APIs. That
library and the historical oracle/profiler artifacts are intentionally absent
from the imported commit. Standard official `libcg.so` exposes the competition
ABI but not those seeded/digest exports. Consequently, the historical parity
numbers can be read, but not independently reproduced from this snapshot.

The CUDA subtree itself reports zero of six formal promotion gates complete.
It also records missing search, evolution, Item/Supporter/Stadium effects,
arbitrary damage-counter movement, special selections, device codecs, complete
games, sanitizer soak, and end-to-end throughput comparison.

### 5. Policy-only CUDA evidence is not engine parity

Historical PolicyCodecV1 replay used CPU-encoded tensors and found 12 raw
CPU/GPU policy action flips in 127,205 decisions, including eight inequivalent
flips. A 600-pair battle A/B used the same CPU engine on both sides and found one
policy action flip that changed the winner. The PPO smoke was explicitly CPU
official engine plus CUDA learner. None of these exercises a CUDA battle engine.

## Local verification

### Environment installed

- GPU: NVIDIA GeForce RTX 5080, compute capability 12.0
- driver: 596.49, reported CUDA compatibility 13.2
- persistent toolkit: `/home/cyd/.local/cuda-12.8`
- nvcc: 12.8.93
- GCC/G++: 13.4.0
- CMake: 4.4.1
- Ninja: 1.13.2
- Nsight Compute: 2025.1.1
- Compute Sanitizer: 2025.1.0

The tools are linked from `/home/cyd/.local/bin` and are not stored in a project
virtual environment.

### Tests and runs

| Check | Result | Evidence boundary |
|---|---:|---|
| imported Python suite | 27/27 pass | schema, reference smoke interpreter, routing, pool helpers |
| 0020 foundation suite | 8/8 pass | frozen model/package contracts, not CUDA integration |
| SM 120 CUDA build | pass | three native executables compiled with nvcc 12.8.93 |
| generic smoke | 16,384 envs x 1,000 steps; 14.005M env-steps/s; error 0 | synthetic smoke opcodes only |
| setup kernel | 1,000 seeds; 2,000 hands; CUDA execution pass | no local official oracle available |
| status-core kernel | 1,000 traces; 18,485 decisions; error 0 | hard-coded two-card micro-engine |
| Compute Sanitizer | blocked | WSL/WDDM debugger interface unsupported |
| Nsight Compute counters | blocked | `ERR_NVGPUCTRPERM` on this WSL device |

The installed environment probe incorrectly reported that NCU counters were
readable; the actual profiler launch is authoritative and failed permission
access. No NCU-derived performance conclusion is accepted from this host.

### Direct Arena-deck comparison

The audit also exercised the exact 60-card packages
`alakazam_dudunsparce_04_sota` and `dragapult_ex_03_v20260729_rl` rather than
only synthetic fixtures.

The unmodified official runtime completed 10/10 games with 0 errors and 0
unfinished games. Alakazam won 9 and lost 1; the games contained 1,371 policy
selections and 19 observed KO events. The temporary report is
`.tmp/evaluation/0020_cuda_engine_comparison/run-86b01c39f218482b8f627852ee3a7567/report.html`.
This establishes that both packages and their full card interactions are
runnable through the official engine; the 9-1 result is not a CUDA parity
claim.

The same deck files were then supplied to every relevant CUDA executable:

- `ptcg_cuda_official_setup` accepted both decks and produced seeded shuffle,
  opening-hand, and mulligan output. It stopped at setup and exposed no legal
  action or battle-step interface.
- `ptcg_cuda_status_core` rejected the first deck with
  `controlled status deck must contain 4 expected Pokemon and 56 expected
  Energy`. This is its intended hard-coded fixture boundary.
- `ptcg_cuda_smoke` returned a successful synthetic one-step result, but its
  argument parser ignores the deck arguments and initializes two artificial
  Active cards plus deck/hand counts. It therefore did not play either Arena
  deck.

Consequently, there is no CUDA terminal result or step trace to compare with
the official games. The observed blocker is stronger than a full-game semantic
mismatch: this snapshot has no executable that can represent and start an
arbitrary Arena battle. Setup is usable as a tested primitive; complete rollout
collection is not.

### Arena-only policy-placement A/B (not an RL engine benchmark)

This experiment was initially run as a throughput proxy, but it does not match
the project's RL topology and must not be used to estimate engine acceleration.
It moved an Arena package's batch-one inference from CPU to independent CUDA
contexts; it did not keep one centralized GPU learner or replace any official
engine work.

A separate throughput test kept the official CPU battle runtime, the exact
Alakazam and Dragapult decks, 100 games, eight game workers, one CPU thread per
worker, and the same Alakazam checkpoint. The only operational change was the
Alakazam policy device: packaged CPU inference versus CUDA inference on the RTX
5080. The Dragapult opponent remained on CPU in both arms.

| Mode | Completed | Errors | Selections | Wall time | Games/s | Selections/s |
|---|---:|---:|---:|---:|---:|---:|
| official engine + CPU candidate policy | 100/100 | 0 | 13,933 | 82.714 s | 1.209 | 168.449 |
| official engine + CUDA candidate policy | 100/100 | 0 | 13,738 | 112.803 s | 0.887 | 121.788 |

The naive CUDA-policy arm was 26.7% lower in games/s and took 36.4% more wall
time. The evaluation runner intentionally isolates every game in its own
process, so this arm created independent CUDA contexts and performed batch-one
inference with per-decision CPU-to-device tensor movement. It did not use the
prototype's centralized policy pool or device codec because neither is wired
to the Arena runtime or compatible with the current 0020 model contract.

The two runs were not seeded paired replays, so their identical 95-5 outcome is
not parity evidence. Throughput is compared using completed games and
selection-normalized rates; both arms completed without errors. Reports:

- CPU: `.tmp/evaluation/0020_hybrid_throughput_cpu/run-289b8115e46d447bbfe4bc8492569776/report.html`
- CUDA policy: `.tmp/evaluation/0020_hybrid_throughput_gpu/run-0904d5428f5046c9a914c671e19bd5e9/report.html`

This result does not show that GPU inference is intrinsically slower. It shows
that the currently runnable composition is the wrong execution topology. A
credible hybrid speedup requires one resident batched GPU policy service (or an
equivalent centralized router), persistent official-engine environments, and
amortized transfers across many simultaneous decisions.

### Real RL-topology baseline

The corrected benchmark used the current 0020 rollout collector: one centralized
Dragapult learner on CUDA, 16 official-engine worker processes, CPU opponent
inference inside each worker, two-millisecond learner-request coalescing, and
sample-mode actions. It fixed `alakazam_dudunsparce_04_sota` as the opponent for
100 games.

| Metric | Result |
|---|---:|
| valid episodes / errors | 100 / 0 |
| wall time | 131.592 s |
| episodes/s | 0.760 |
| official engine selections | 13,613 |
| learner decisions | 7,838 |
| learner decisions/s | 59.563 |
| learner inference batches | 904 |
| mean / maximum learner batch | 8.670 / 16 |
| measured learner inference time | 48.881 s |
| learner inference share of wall time | 37.1% |

The remaining 82.711 seconds (62.9%) is not an engine-only measurement. It
combines official `battle_start`/`battle_select`, CPU opponent inference,
observation serialization and IPC, CPU feature encoding, process startup, and
scheduling. The machine-readable output is
`.tmp/evaluation/0020_engine_accel_rl_baseline/official_cpu_engine_gpu_0020_learner_cpu_opponent.json`.

No partial-CUDA engine arm exists for a paired comparison in this snapshot.
The official worker owns a complete opaque battle and advances it through
`battle_select`. In contrast, the CUDA prototype resets from only seed, two
Active IDs/HP values, deck/hand counts, and policy IDs. It has no import/export
bridge for the official state, legal options, effect stack, or RNG position.
Consequently, its setup, status-core, and smoke kernels cannot replace a slice
of `battle_select` and then return control to the official engine.

### Real RL-topology timing decomposition

An opt-in temporary profiler then timed the same logical topology without
changing official source or substituting a CUDA engine. This independent
sample-mode run completed 100/100 games with 0 errors, 13,923 engine selections,
and 7,876 learner decisions in 85.182 seconds. Its different wall time from the
baseline is not treated as a policy or engine comparison: the official runtime
uses uncontrolled randomness and sample-mode trajectories are not paired.

| Surface | Measured time | Normalized cost |
|---|---:|---:|
| official `battle_start` | 0.301 s worker aggregate | 3.012 ms/game |
| official `battle_select` including observation JSON | 4.939 s worker aggregate | 0.355 ms/selection |
| CPU opponent inference | 221.725 s worker aggregate | 2.217 s/game |
| CPU opponent/package load | 139.812 s worker aggregate | 1.398 s/game |
| main-agent feature encoding | 7.461 s parent wall | 0.947 ms/decision |
| collate plus H2D | 1.245 s parent wall | 1.334 ms/batch |
| CUDA learner action sampling | 29.846 s parent wall | 31.989 ms/batch |
| trajectory D2H | 0.319 s parent wall | 0.041 ms/decision |

The worker times overlap across 16 processes, so aggregate worker seconds are
not wall-time shares. The official calls averaged 52.4 ms against 11.070 seconds
of worker wall time per game, about 0.47% of the per-game worker path; the
largest per-game official-call total was 86.7 ms. Ideal 16-way overlap would
assign only about 0.328 seconds, or 0.4%, of this run's makespan directly to all
measured official calls. This is an estimate, not a strict speedup bound, because
scheduling imbalance and CPU contention can alter makespan.

The important engineering conclusion is nevertheless robust: in this workload,
the official engine calls are much smaller than CPU opponent execution,
per-episode opponent loading, centralized learner service, and CPU feature
encoding. A setup-only CUDA bridge targets about 3 ms/game before bridge costs.
Per-opcode CPU/GPU migration would add state serialization, transfers, and host
synchronization around a roughly 0.355-ms official call and is therefore not a
credible acceleration boundary. The machine-readable profile is
`.tmp/evaluation/0020_engine_accel_rl_profile/official_cpu_profile.json`; its
implementation plan is
`docs/superpowers/plans/2026-07-30-0020-hybrid-engine-feasibility.md`.

## Promotion contract for 0020

`engine_cuda/` may enter 0020 rollout collection only when all of the following
are true for the exact promoted deck/opponent pool:

1. A reproducible official seeded oracle and immutable source/binary hashes are
   available without changing `engine/source/`.
2. CPU POD and CUDA use one shared rule/state contract; bespoke per-deck CUDA
   battle kernels do not count as general-engine coverage.
3. Every official decision compares legal options, selected action, public
   observation, hidden digest, turn budgets, RNG consumption index, rewards,
   error state, and terminal result after each transition.
4. At least 100,000 seeded decisions pass with zero unexplained mismatch, plus
   targeted fixtures for every reachable card/effect in each promoted deck.
5. Unsupported behavior fails closed with a stable code; promoted paired games
   contain zero unsupported events and no mid-game CPU fallback.
6. Device features exactly reproduce the 0020 `192/128` model contract with no
   truncation and pass frozen-corpus elementwise comparison.
7. Sanitizer passes on a supported native Linux host, followed by a 24-hour
   full-pipeline soak with no memory growth or engine errors.
8. Identical-hardware end-to-end rollout plus learner throughput is at least 3x
   the tuned CPU-engine pipeline; micro-opcode throughput is not substituted.

Until then, CUDA work is an isolated research branch. Formal 0020 reports and
policy promotion continue to use the unmodified official engine runtime.

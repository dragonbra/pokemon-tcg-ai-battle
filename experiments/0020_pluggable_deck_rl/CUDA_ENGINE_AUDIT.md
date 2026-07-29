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

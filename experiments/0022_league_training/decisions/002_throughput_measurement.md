# 0022 Decision 002: Throughput Measurement

**Date:** 2026-07-31
**Status:** measured; end-to-end gate passed, opponent-only gate pending

## Scope

This official-engine benchmark used the 16-deck snapshot with catalog SHA-256
`92221e5a6b080d46f85d499e491a371a610a94087d7f001a1663bc657cfbe5c0` and Foundation
`V1_universal_winner_r15`. `engine/source/` was not modified.

Arm A was the current heterogeneous CPU Arena package path with GPU candidate, 16 workers,
2 ms coalescing and 300 stochastic games. Arm D used one resident GPU encoder, 16 independent
deck decoder slices plus one candidate decoder/value slice, stacked expert GEMM, 256 workers,
5 ms coalescing and 512 stochastic games. Both arms completed every game with zero errors.

## PPO-ready Results

| Metric | A current CPU packages | D stacked shared GPU | D / A |
| --- | ---: | ---: | ---: |
| Completed games | 300/300 | 512/512 | 100% |
| Episodes/s | 1.990 | 3.568 | **1.79x** |
| Engine selections/s | 339.14 | 582.17 | **1.72x** |
| Opponent decisions/s | 152.62 | 264.44 | **1.73x** |
| Candidate decisions/s | 186.50 | 317.73 | **1.70x** |
| Mean shared request batch | 11.86 | 163.81 | 13.8x |
| Non-finite log-prob/entropy/value | 0 | 0 | no regression |

D produced `log_prob=-0.6251`, `entropy=0.6279`, and `value=0.0` mean candidate trajectory
scalars without a backward pass. The neutral value is expected for this inference benchmark.

The resident encoder used 66,915,840 parameter bytes and the 17 decoder slices used
69,849,736 bytes. Peak PyTorch allocated memory was 708,751,360 bytes. Memory was not the
limiting resource. With a Torch-light worker entry path, 256 engine workers used about 6.8 GiB
RSS; the earlier 32-worker path that imported PyTorch unnecessarily used about 16 GiB RSS.

## Decision

The 1.5x end-to-end threshold is passed. The 2.0x opponent-decision threshold is not passed;
the measured value is 1.73x. This is a partial throughput gate, not authorization to start
Live self-evolution yet.

The causal result is that resident encoder placement alone is insufficient. Per-head small GPU
launches made decoder routing the bottleneck. Stacking independent decoder parameters along an
expert dimension preserved deck-specific trainability and reduced routing overhead; exact greedy
action parity with the original decoder is covered by a focused test.

The comparison is throughput evidence, not a strength comparison: A uses heterogeneous current
opponents while D uses the common Foundation. The next gate should optimize stacked decoder and
engine scheduling until the opponent threshold is reached, or record a new decision to revise
that threshold. Frozen quality evaluation and PPO training remain separate official-engine work.

## Artifacts

- [A sample baseline](/home/cyd/repos/pokemon-tcg-ai-battle/.tmp/evaluation/0022_league_throughput/formal_300_a_sample.json)
- [D PPO-ready 512 benchmark](/home/cyd/repos/pokemon-tcg-ai-battle/.tmp/evaluation/0022_league_throughput/formal_512_unified_gpu_stacked_sample.json)
- [B/C device comparison](/home/cyd/repos/pokemon-tcg-ai-battle/.tmp/evaluation/0022_league_throughput/device_64_b_vs_c.json)

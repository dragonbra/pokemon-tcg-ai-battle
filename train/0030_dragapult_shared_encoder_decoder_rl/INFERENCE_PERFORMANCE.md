# 0030 inference performance notes

This document is the handoff for later agents working on the 0030 pure Dragapult decoder RL run.

## What was slow before

The 0026 decoder-only PPO collector used two independent model stacks and emitted separate focal and opponent GPU batches. Its PPO loop retained raw canonical feature batches and reran the frozen 4-layer state encoder plus 3-layer option encoder for every minibatch in every PPO epoch. Neither behavior changes the policy contract, but both waste GPU time.

The relevant 0023 V6 baseline completed 512-game update intervals in roughly 408-410 seconds, or about 1.25 games/s, with 128 workers. Its latest measured per-update inference time was 169-177 seconds for roughly 82k-84k inference requests.

## 0030 hot path

1. Every engine game still has an isolated spawned process. Causal state remains per actor and per game.
2. Ready focal and opponent observations are CPU-compiled together and collated once.
3. One 0028 model instance precomputes immutable official prototype embeddings once, then runs state and option encoding once for each combined GPU batch.
4. A boolean route selects either the live focal decoder or one shared immutable opponent decoder. Opponents always decode greedily; the focal decoder samples during rollout and is greedy during frozen evaluation.
5. Focal trajectories store only `summary`, contextual `options`, `option_mask`, `min_count`, and `max_count`, plus action/value scalars. Option tensors are cropped to each decision's real option count and the two floating caches use CPU fp16; PPO casts them back to fp32 before the decoder. No raw semantic batch or cross-row padding is retained.
6. PPO epochs reuse those cached decoder inputs. A fail-closed benchmark monkey-patches `actor.encode` to raise if PPO attempts representation work.
7. The rollout package exports the collector lazily. Spawned official-engine workers load protocol and `libcg` only; importing a worker is tested to leave Torch absent from `sys.modules`.

There are therefore two decoder copies on GPU: the focal live decoder and one frozen decoder shared by all 51 opponent identities. There are not 51 opponent models or decoder copies. The much larger representation stack exists once.

## Measured gates on RTX 5080

The 4-game diagnostic completed with zero engine errors, 450 focal decisions, 784 total inference requests, and 349 MB peak allocated GPU memory. Cached PPO processed about 1,509 decoder decisions/s across two epochs and made zero encoder calls.

The historical V2 representative 64-game / 64-worker gate is recorded at `.tmp/evaluation/0030_performance/performance_gate_64games_prototype_cached.json`:

- 64/64 official-engine games, zero errors;
- 52.165 seconds, 1.227 games/s;
- 12,348 shared inference requests in 312 batches;
- 31.598 seconds total GPU inference time;
- 7,563 focal decisions;
- 10,632 cached PPO decoder decisions/s across two epochs;
- zero representation encode calls during PPO;
- unchanged representation and opponent decoder hashes;
- 0.74 GB peak allocated GPU memory and 2.11 GiB process max RSS.

The V2 1.227 games/s result was about 1.8% below the approximately 1.25 games/s 0023 V6 baseline and passed its 10% no-material-regression gate. V2's 128-worker calibration exceeded the collector's 60-second response timeout because heavyweight worker imports exhausted host memory; 64 workers was therefore the highest tested zero-error V2 setting. The V3 correction and replacement calibration are recorded below.

The first formal `V1_shared_encoder_cached_decoder` attempt was intentionally interrupted before update 1. It exposed that per-decision caches retained the unified batch's padded option width, causing falling RAM and swap growth during a 512-game rollout. `V2_trimmed_fp16_prototype_cache` contains the crop/fp16 fix and cross-batch prototype cache; V1 remains preserved and must not be resumed.

## V3 lightweight worker correction

V2 exposed a separate process-memory defect. `rollout/__init__.py` eagerly imported `collector`, and `collector` imports Torch. Because every official-engine game uses a fresh spawn process, each worker mapped Torch, CUDA and NumPy even though inference and feature compilation run only in the parent. Live PSS was about 280 MiB per worker and about 17.3 GiB at 64 workers. The parent training process was about 1.28 GiB PSS; GPU use was only about 6.7/16.3 GiB, so feature tensors and VRAM were not the dominant host-memory pressure.

`V3_lightweight_engine_workers` makes only the collector export lazy. A subprocess regression test proves that importing `rollout.worker` does not import Torch. The benchmark must be launched through `python3 -m train.0030_dragapult_shared_encoder_decoder_rl benchmark`; launching the standalone `benchmark` module causes Python spawn to re-import that Torch-owning main module and is not equivalent to the formal training entry path.

All V3 calibrations used the V2 update-5 model-only checkpoint (`7065bdd8fa49472b64c871a1c4e470c58a390ed5710968550386e297d1b7a7bf`), 128 official-engine games and unchanged timeouts:

| Workers | Games/s | Worker PSS peak | Min available RAM | Torch/CUDA workers | Errors |
|---:|---:|---:|---:|---:|---:|
| 64 | 2.080 | 0.86 GiB | 18.6 GiB | 0 / 0 | 0 |
| 96 | 2.190 | 1.28 GiB | 18.1 GiB | 0 / 0 | 0 |
| 128 | 2.384 | 1.57 GiB | 17.8 GiB | 0 / 0 | 0 |

None of the runs increased swap. The formal V3 setting is therefore 128 workers. Its 2.384 games/s gate is about 76% faster than V2 update 1's 1.355 games/s, though the first complete 512-game V3 update remains the production throughput confirmation.

## Do not regress these properties

- Do not move feature compilation into engine workers unless causal state equivalence and serialization overhead are re-benchmarked.
- Do not cache across decisions or games. Only reuse a decision's immutable summary/options within its PPO update.
- Never retain the unified batch's padded option width in individual trajectories. Crop with the row's true `option_mask` count before moving to CPU.
- Do not call the representation encoder from PPO minibatches.
- Do not instantiate one opponent model or decoder per deck.
- Do not call `prototype_encoder.encode_all()` per inference batch. The decoder-only contract makes this memory immutable; prepare it once on GPU.
- Do not train the opponent decoder, prototype encoder, state encoder, or option encoder.
- Do not mix sampled rollout win rate with frozen greedy checkpoint evaluation.
- Keep coalescing short (currently 0.5 ms); excessive waiting lowers engine utilization.
- Benchmark with official-engine games. Decoder-only microbenchmarks are necessary but cannot establish rollout throughput or policy validity.
- Launch process-memory benchmarks through the project `__main__` CLI so spawn behavior matches formal training.
- Keep engine workers Torch-light; any worker Torch or CUDA mapping fails the performance gate.

## Comparison with 0022 pure Dragapult RL

At the same first completed update, 0030 V2 sampled 227-285 (44.34%) over 512 games; 0022 V2 sampled 140-372 (27.34%). The 17.0-point gap is favorable to 0030, but it is not a controlled strength comparison: 0030 starts from 0028 epoch 4 and faces 51 decks driven by one 0028 decoder, while 0022 starts from the older foundation and its then-current 48-deck league decoder set.

The initial frozen greedy snapshots show the same directional gap: 0030 is 66-36 (64.71%, 102 games), while 0022 V2 is 43-53 (44.79%, 96 games). Again, different pool/policy contracts prevent treating the gap as a head-to-head improvement.

0022 V2 update 1 took 199.60 seconds (2.565 games/s) with 128 workers, versus 377.74 seconds (1.355 games/s) for 0030 V2 with 64 heavyweight workers. The V3 128-worker gate reached 2.384 games/s, about 7% below 0022's full-update throughput. That comparison is still approximate until V3 completes a full 512-game update, and policy strength remains incomparable across the different opponent/foundation contracts.

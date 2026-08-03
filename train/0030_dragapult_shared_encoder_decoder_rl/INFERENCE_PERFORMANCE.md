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

There are therefore two decoder copies on GPU: the focal live decoder and one frozen decoder shared by all 51 opponent identities. There are not 51 opponent models or decoder copies. The much larger representation stack exists once.

## Measured gates on RTX 5080

The 4-game diagnostic completed with zero engine errors, 450 focal decisions, 784 total inference requests, and 349 MB peak allocated GPU memory. Cached PPO processed about 1,509 decoder decisions/s across two epochs and made zero encoder calls.

The final representative 64-game / 64-worker gate is recorded at `.tmp/evaluation/0030_performance/performance_gate_64games_prototype_cached.json`:

- 64/64 official-engine games, zero errors;
- 52.165 seconds, 1.227 games/s;
- 12,348 shared inference requests in 312 batches;
- 31.598 seconds total GPU inference time;
- 7,563 focal decisions;
- 10,632 cached PPO decoder decisions/s across two epochs;
- zero representation encode calls during PPO;
- unchanged representation and opponent decoder hashes;
- 0.74 GB peak allocated GPU memory and 2.11 GiB process max RSS.

The 1.227 games/s result is about 1.8% below the approximately 1.25 games/s 0023 V6 baseline and passes the 10% no-material-regression gate. A 128-worker calibration exceeded the collector's 60-second worker-response limit before producing a valid report; all workers were cleaned up and no GPU allocation leaked. On this 23 GiB host, 64 workers is therefore the formal setting: it is the highest tested zero-error configuration and does not weaken the timeout health contract merely to admit oversubscription.

The first formal `V1_shared_encoder_cached_decoder` attempt was intentionally interrupted before update 1. It exposed that per-decision caches retained the unified batch's padded option width, causing falling RAM and swap growth during a 512-game rollout. `V2_trimmed_fp16_prototype_cache` contains the crop/fp16 fix and cross-batch prototype cache; V1 remains preserved and must not be resumed.

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

## Comparison with 0022 pure Dragapult RL

At the same first completed update, 0030 V2 sampled 227-285 (44.34%) over 512 games; 0022 V2 sampled 140-372 (27.34%). The 17.0-point gap is favorable to 0030, but it is not a controlled strength comparison: 0030 starts from 0028 epoch 4 and faces 51 decks driven by one 0028 decoder, while 0022 starts from the older foundation and its then-current 48-deck league decoder set.

The initial frozen greedy snapshots show the same directional gap: 0030 is 66-36 (64.71%, 102 games), while 0022 V2 is 43-53 (44.79%, 96 games). Again, different pool/policy contracts prevent treating the gap as a head-to-head improvement.

0022 V2 was substantially faster: update 1 took 199.60 seconds (2.565 games/s) with 128 workers, versus 377.74 seconds (1.355 games/s) for 0030 V2 with 64 workers. The semantic encoder's per-request cost and the lower safe concurrency on this 23 GiB host are real costs. 0030 meets the later 0023 throughput baseline, but it does not match 0022 V2 throughput. Watch both RAM and swap: even cropped caches can coexist with 64 engine workers under memory pressure.

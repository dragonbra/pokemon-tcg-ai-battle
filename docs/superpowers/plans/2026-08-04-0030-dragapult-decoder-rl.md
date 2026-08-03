# 0030 Dragapult Shared-Encoder Decoder RL Implementation Plan

**Goal:** Fine-tune only the pure Dragapult action decoder and value head from the 0028 epoch-4 semantic checkpoint against all 51 Frozen 0019 exact decks, while keeping every opponent on one shared immutable 0028 decoder and avoiding rollout throughput regression.

**Project:** `0030_dragapult_shared_encoder_decoder_rl`

## Immutable identities

- Focal deck: `evaluation/arena/frozen/dragapult_ex_001/deck.csv`.
- Foundation checkpoint: `rl_runs/0028_universal_semantic_foundation_pretraining/versions/V3_shared_prototype_batch512/checkpoint/universal_semantic/latest.pt`.
- Checkpoint SHA-256: `5e0a6eea42bf228a9bd977cf56fdd14ad360fbceffcf5c19e2d9fe1b39713d98`.
- Opponent pool: `0019_foundation_51_exact_decks_v4`, all 51 exact decks, but using the same frozen 0028 checkpoint rather than the historical 0019 weights.
- Official engine runtime remains unchanged and authoritative.

## Implementation tasks

1. Create self-contained roots under `train/`, `experiments/`, and `rl_runs/`; freeze the required 0028 model, feature compiler, causal knowledge, prototype assets, deck catalog, and the proven official-engine worker protocol into 0030.
2. Build one `SemanticActorCritic` with an immutable 0028 representation, one live focal decoder, one immutable reference/opponent decoder, and one live value head. Assert exact trainable parameter names and representation hashes.
3. Replace heterogeneous inference with a shared request path: compile all ready focal and opponent observations, collate once, encode once on GPU, then route cached summaries/options to the live or frozen decoder. Preserve actor-local causal encoders and per-game process isolation.
4. Store compact decoder inputs in focal trajectories. PPO minibatches reuse cached state summaries, option embeddings, masks, and selection bounds; they must never call the state, option, or prototype encoders.
5. Keep terminal official-engine reward, episode-balanced GAE, PPO clipping/KL/entropy/reference-anchor controls, balanced seats, and 51-opponent coverage. Log the sampled source policy update separately from the resulting checkpoint update.
6. Save atomic model-only checkpoints containing only the focal decoder and value head, bound to the immutable foundation and representation hashes. Reject optimizer, scheduler, RNG, replay, rollout, and dataloader state.
7. Add tests for checkpoint identity, exact deck/catalog coverage, trainable/frozen parameter contracts, cached/full-forward numerical equivalence, no encoder invocation during PPO, opponent decoder immutability, schedule balance, checkpoint schema, and worker smoke behavior.
8. Add two performance gates: a decoder microbenchmark comparing cached PPO evaluation against full encoder recomputation, and an official-engine rollout benchmark reporting games/s, decisions/s, inference seconds, mean batch size, GPU memory, and comparison with the 0023 recorded baseline. Formal training is blocked on semantic equivalence, zero engine errors, and no material rollout throughput regression.
9. Record the optimization decisions and measurements in `train/0030_dragapult_shared_encoder_decoder_rl/INFERENCE_PERFORMANCE.md`; maintain authoritative `experiments/.../DESIGN.md` and `DESIGN.html`.
10. Allocate `V1_shared_encoder_cached_decoder`, run tests/smoke/benchmark, then launch online W&B PPO through a foreground watchdog. Poll the session at no more than 60-second intervals and diagnose any alert immediately.

## Performance acceptance gates

- Cached and uncached action log-probability, entropy, value, and greedy action outputs agree within floating-point tolerance.
- Representation encode calls during a PPO update: exactly zero.
- Opponent decoder parameters and frozen representation hashes remain unchanged.
- Official-engine canary completes without worker error, illegal action, timeout, or fallback increase.
- Shared inference uses one encoder instance and one frozen opponent decoder instance on GPU.
- Rollout decisions/s is at least the comparable 0023 baseline within a 10% tolerance, or a documented hardware/load-normalized benchmark shows the official engine rather than feature encoding is the limiting factor.
- Peak GPU memory fits the local RTX 5080 with margin for rollout and PPO batching.

## Formal run contract

- Version: `V1_shared_encoder_cached_decoder`.
- W&B: private `dragon_bra/pokemon-tcg-policy-learning`, online.
- Only `action_decoder.*` and `value_head.*` are optimized.
- Checkpoints are model-only and retention is bounded to latest, best, and a small number of branch points.
- Frozen greedy evaluation is a separate balanced 102-game pass and is not conflated with sampled rollout metrics.

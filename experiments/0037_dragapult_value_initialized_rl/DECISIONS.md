# 0037 Decisions

## 2026-08-08 — Allocate the Value-initialized seeded PPO baseline

- Use exact Frozen-0806 deck 007 as the focal Dragapult deck and only immutable friend-0806 epoch-11 pretrained policies as opponents.
- Initialize the critic from 0036 V2 epoch 5, the nominal exact-007 validation champion. Freeze the pretrained Encoder; train the original action decoder and the latent-query critic trunk/scalar output. Auxiliary archetype/final-diff outputs remain frozen and receive no PPO loss.
- Preserve terminal official outcome as the only environment reward. Do not add PBRS or Value-delta shaping in V1.
- Use turn-clock GAE with `gamma=1.0` and `lambda=0.95`: consecutive focal decisions inside the same official turn do not decay; a crossed turn boundary applies lambda once.
- Each update samples all 256 committed Frozen-0806 scenario slots in a fresh deterministic permutation, assigns 256 unique engine seeds, and emits opposite-seat mates for 512 Episodes. Training, repeated validation and final holdout use disjoint seed domains.
- Paired seats reduce environmental and seat variance but do not make the 512 games independent. Metrics retain the 256 pair groups and WW/split/LL counts.
- Adopt the admitted worker-local stateless feature compiler and centralized CUDA batching. Do not enable the slightly slower incremental compiler research path.
- Deferred branches are additional Encoder unfreezing/LoRA and Value-guided action search. Either requires a later immutable version.

## 2026-08-08 — Launch V1

- All 55 focused 0037 tests pass, including exact 0036 critic tensor identity and worker-local versus central 39-key feature parity.
- The four-game official-engine PPO canary completed 4/4 with zero errors. Behavior log-probability replay maximum error was `2.861e-6`; the frozen representation hash remained unchanged and both decoder and critic received finite updates.
- The 64-game/32-worker throughput smoke completed 64/64 with zero errors in 61.80 seconds. This is a capacity gate, not policy-strength evidence.
- V1 `V1_value_initialized_turn_clock_seeded512` launched with W&B run ID `0037-v1-value-initialized-turn-clock-seeded512`. It begins with the fixed seeded-512 update-0 greedy validation suite before collecting the first fresh seeded-512 stochastic rollout.

## 2026-08-08 — Stop V1: incomplete admitted-runtime contract

- V1 completed only its fixed checkpoint-0 seeded-512 evaluation: 512/512, zero errors, 287–225, `56.05%`, wall `417.389 s`. It was interrupted before a PPO checkpoint was produced.
- Audit found that V1 used N32E1 per-game process spawning. It did compile features inside each worker, but it did not use the admitted multi-battle engine pool and still recomputed `OfficialPrototypeEncoder.encode_all()` every model forward.
- Preserve V1 and its W&B run as a failed immutable version with reason `missing_engine_pool_and_prototype_cache_contract`. Do not compare its interrupted source-update state as checkpoint-1 strength.

## 2026-08-08 — Admit prototype cache and select N16E8/I8

- Port the admitted model-static prototype cache into the self-contained 0037 policy. Cache tensors are detached derived runtime state, never checkpoint keys; `_apply` and `load_state_dict` invalidate them. Trainable prototype parameters bypass the cache, while 0037's frozen prototypes reuse it during PPO and inference.
- Replace per-game spawning with N persistent worker processes per collection, E independent official battle pointers per process, and I bounded synchronous inference channels per role. Each battle side owns causal worker-local stateless compiler state; collation, stochastic sampling, actor/value inference and trajectory capture stay centralized on CUDA.
- The strict greedy seeded-512 finalists used identical 95,214 selections in every arm. N16E8/I8 at 5 ms completed twice in `190.420/191.264 s` (median `190.842 s`, `2.683 games/s`, `498.9 selections/s`, median process-tree RSS `15.74 GiB`). N8E16/I8 completed in `195.831/195.696 s` (median `195.763 s`, `2.615 games/s`, `486.4 selections/s`, `11.08 GiB`).
- Select N16E8/I8 with 5 ms coalescing for the fastest formal V2 baseline. It cuts fixed seeded-512 wall by `54.3%` versus V1 and increases games/s by about `118.7%`. Retain N8E16/I8 as the low-memory profile.
- A 16-game pooled stochastic PPO canary completed 16/16 with zero errors: behavior replay MAE `3.576e-6`, explained variance `0.5197`, finite Value/KL/gradient metrics, decoder changed, and frozen Encoder hash unchanged.
- All 62 project tests pass; compileall, diff check, and the no-`engine/source/`-change gate pass.

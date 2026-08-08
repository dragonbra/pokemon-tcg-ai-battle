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

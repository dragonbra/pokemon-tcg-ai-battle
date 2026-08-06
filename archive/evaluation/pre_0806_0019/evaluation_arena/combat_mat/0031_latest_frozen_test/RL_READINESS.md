# CUDA Engine RL Readiness For Frozen51

## Acceptance result

- Frozen pool: `0019_foundation_51_exact_decks_v4`, 51 exact 60-card decks.
- Admitted CUDA subset: 38 decks.
- Rejected fail-closed: 13 decks.
- Differential gate: all 38 x 38 ordered matchups, seed 1, decision limit 2,048, `coverage-first-legal`.
- Evidence: 1,444 completed battles and 297,285 decisions with zero CPU/POD/CUDA state, status, or outcome mismatch.
- This is finite trajectory evidence. It does not prove every reachable stochastic interaction.

The machine-readable deck list and blocker details are in [cuda_support.json](cuda_support.json), with a browsable table in [cuda_support.html](cuda_support.html).

## Why strict 0031 continuation still cannot run resident CUDA rollouts

The CUDA binding currently exposes `PolicyCodecV1` tensors from `OfficialStatePod`. The 0031 checkpoint uses a different actor-visible contract built by `OnlineCausalEncoder` from official JSON observations plus chronological `CausalKnowledge`.

The missing resident projection includes:

- the rolling typed-event window used by `event_cat`, `event_num`, `event_state`, and event source/target/before/after links;
- known, possible, unknown, and remembered opponent-hand identities with their causal provenance;
- exact known self-deck ordering after reveal/search/shuffle events;
- knowledge-stage and source-event fields in the deck/prize resource ledger;
- the exact 0031 semantic option/card/effect/skill tensor layout and canonical action normalization.

`OfficialStatePod` retains current zones, legal options, limited turn histories, internal card state, and two log indices. It does not retain the exact official observation/log stream or the 64-event causal knowledge state required to reproduce 0031 features. Zero-filling these fields changes the policy. Copying official observations through CPU breaks the claimed fully GPU-resident hot path. Both alternatives are rejected.

## Required work for strict 0031 parity

1. Define a versioned resident observation ABI that captures public/revealed events as they happen, separately for both player perspectives. Hidden information must never leak across the actor boundary.
2. Add device-resident causal ledgers for opponent-hand certainty, remembered identities, known deck order, prize/deck bounds, and a 64-event ring buffer.
3. Implement a GPU feature compiler that produces every 0031 tensor with the exact schema, shape, padding, ordering, numeric normalization, and knowledge-state codes.
4. Implement the 0031 option/action codec against CUDA option ordering, including variable-cardinality unique selections and canonical normalized actions.
5. Run step-by-step A/B tests against the existing official-engine `OnlineCausalEncoder`; require tensor equality and greedy action equality over representative full games before rollout benchmarking.
6. Add reward, terminal, draw, truncation, and value-bootstrap contracts. In particular, `mega_lopunny_ex_001` demonstrated a nonterminal trajectory beyond 2,048 decisions without semantic mismatch, so the RL horizon cannot remain implicit.
7. Freeze the admitted 38-deck snapshot, exact deck hashes, sampling weights, seat balancing, seed schedule, and CUDA support evidence hash in every RL run manifest.
8. Keep engine errors and unsupported continuations terminal and fail-closed. Do not silently route a lane to the CPU engine.

## 0032 POD-native adapter result

0032 selected the new resident policy contract route. `V1_pod_native_adapter` now passes the runtime gates:

- strict `[B, 8]` / `[B, 16]` global, `[B, 128, 6]` / `[B, 128, 10]` entity, and `[B, 128, 12]` / `[B, 128, 4]` option tensor contract;
- 14,579,843-parameter actor-critic with a 64-step ordered unique-option decoder and scalar value head;
- explicit 0031 transfer of 11,052,162 parameters (75.8044%), without shape-only matching;
- 400-decision CPU/CUDA codec parity with zero field and status mismatch;
- fixed-seed 500-step eager and CUDA Graph A/B over all 38 admitted opponents, each with 19,000 decisions, 187 completed episodes, and zero legality, overflow, or engine errors;
- strict graph replay speedup from 501.212 to 1,923.473 decisions/s at 38 lanes (3.84x);
- recommended 152-lane rollout throughput of 3,579.865 decisions/s and 35.328 episodes/s, or contextual 21.25x and 29.22x ratios against the current same-machine CPU reference.

The 128-option capacity is required by observed runtime evidence: an official state exposed 81 legal options. The generic prototype engine remains at 80 options; only the official resident arena and 0032 contract use 128.

## Formal PPO launch gate

A formal `0032` run based on the 0031 checkpoint must not start until one of these contracts is chosen and validated:

- **Strict 0031 continuation:** implement the exact resident causal adapter above and demonstrate feature/action parity. This preserves checkpoint semantics.
- **New resident policy contract:** introduce a new POD-derived actor input and an explicit input adapter or new encoder. Reusing compatible 0031 weights is initialization only, not strict checkpoint parity; it requires a new versioned design, fresh optimizer, and BC/adaptation validation before PPO.

The runtime adapter gate is closed, but formal PPO must not start from V1. The remaining requirements are:

1. Allocate a new strictly increasing formal version with a fresh optimizer; V1 remains diagnostic-only.
2. Build Dragapult-specific official-engine rollout data. The audited 0031 raw corpus contains zero exact `dragapult_ex_001` decisions, so the existing BC smoke is cross-source diagnostics only.
3. Define reward, terminal, draw, truncation, value bootstrap, advantage, and PPO loss contracts, and synchronize them into the 0032 design documents with tests.
4. Freeze the 38-deck opponent package identifiers/hashes, sampling weights, seat balance, seeds, and curriculum in the run manifest.
5. Use W&B online logging, model-only checkpoints retained at every update, a foreground watchdog, and separate sampled rollout metrics from frozen greedy official-engine evaluation.
6. Run a strict same-contract CPU/CUDA A/B if a defensible CUDA speedup claim, rather than the current contextual throughput comparison, is needed.

Use `dragapult_ex_001` as the focal deck and the admitted 38-deck snapshot as the initial opponent pool. Unsupported engine paths remain fail-closed; no lane may silently fall back to CPU.

## Throughput benchmark interpretation

The completed resident loop measures a compatible `PolicyCodecV1` policy: device reset, feature codec, CUDA Graph actor replay, action packing, and apply all remain on GPU. The 3.84x eager-to-graph result is a strict same-policy execution A/B. The 21.25x decision and 29.22x episode ratios for 152 lanes are contextual topology evidence against the historical CPU report, not 0031 speedup. A valid strict CPU/CUDA PPO comparison still requires the same policy, deck pool, episode horizon, reward handling, batch contract, and seeds on both paths.

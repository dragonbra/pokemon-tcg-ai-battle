# 0043 Decisions

## 2026-08-12: establish the self-contained Champion League boundary

- Name the numbered project `0043_champion_league_rl`.
- Freeze the emitted V22 training config as the regression baseline; do not use old parser defaults to reconstruct approved hyperparameters.
- Treat the existing `Champion-G1` archive as a user-designated generation anchor. Its import is asset migration, not a new automatic Promotion.
- Separate immutable deck identity, policy identity, and evaluation composition. The historical string `0806` may remain only in provenance; it does not define the 0043 Frozen opponent.
- Register the existing 55 exact decks in both the initial Training Deck Pool and FrozenMeta256-V1, while retaining separate pool manifests so later training additions cannot mutate evaluation.
- Use `001`–`055` as the sole canonical identities of the initial exact decks and reserve monotonically appended `056` onward for future decks. Names, archetypes, historical IDs and hashes are metadata only.
- Store Policy-0809 and Champion-G1 under human-readable project-local directories `policy_0809` and `champion_g001`. Hashes validate content but never act as semantic folder names. Runtime code must not fall back to imported historical paths.
- Keep formal training blocked until complete G1/0809 materialization, no-hybrid, sampler, PFSP, telemetry, PPO regression, and Frozen parity gates pass.

## 2026-08-13: adopt CUDA Engine 2.0 with a low-memory build gate

- Use `engine_cuda_2_0/` as the required 0043 GPU backend while retaining `engine/source/` as the official CPU transition oracle.
- Resolve complete Policy-0809 or Champion-G1 identity and exact numeric deck identity before CUDA lane routing. Architecture compatibility never authorizes mutable tensor or cache sharing.
- Pin the actual tracked engine source bytes, state ABI 7, rule ABI 1, private rule-pack hash and SM120 device capability. Native binaries and the PyTorch extension receive separate content hashes.
- Treat `056`–`065` as the append-only training extension already present in the 0043 registry. FrozenMeta256-V1 remains `001`–`055`; no Frozen65 evaluation contract is inferred.
- Build only `ptcg_cuda_smoke` and `_ptcg_cuda` with `--parallel 1`. An optional continuation diagnostic translation unit consumed roughly 30 GiB RSS alone under CUDA 12.8, so increasing global build parallelism or building CMake `all` is not safe on this machine.
- Record native/runtime smokes as implementation evidence only. Keep formal training unauthorized until complete official-game CPU/CUDA observation/action first-divergence parity and fresh version/W&B gates pass.


## 2026-08-13: prepare V1 focal 002/007 without declaring Champion-G2

- Keep immutable policy assets exactly `Policy-0809` and `Champion-G1`. The expanded 29-row initialization is the mutable run-local `V1-Focal-Seed`, not a champion, candidate opponent, or promoted generation.
- Store the seed and all future updates only under `rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/`; never register them in the opponent asset pool before a human Promote decision.
- Expand only `ValueResidualAdapter.own_embedding` and `PolicyStrategyAdapter.own_embedding` from 15×16 to 29×16 using declared G1 parent-row copies. Keep the opponent Meta classifier exactly 15-way, frozen, and outside the optimizer.
- Train focal policy from exact decks 002 and 007 with 128 episodes each per update. Sample opponent deck 001–067 independently from frozen policy identity Policy-0809/Champion-G1.
- Load the mutable focal plus both frozen opponents once per run, keep feature tensors device-resident, and reject requested/materialized policy or exact-deck mismatch before routing. Champion-G1 opponent routing includes its complete strategy/value adapters; it may never silently fall back to the 0809 actor-only path.
- Use the 0042 PPO protocol and trainable boundary unchanged. The only runtime change is the validated CUDA Engine 2.0 backend and the project-local multi-policy/deck routing adapter.
- Mirror sampled-rollout, rolling 100/500/2000, per focal deck, per opponent policy, per opponent deck, branch, prize, PPO/KL, throughput, cache, D2H and routing-health scalars to canonical JSONL, TensorBoard and private W&B. These diagnose and locate candidate updates; they are not frozen greedy strength evidence.
- Mark the project `READY_AWAITING_USER_LAUNCH`. The long run remains unstarted and requires the user's explicit `--launch-formal` authorization.

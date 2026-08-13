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

## 2026-08-13: freeze V1 at C002 and continue with mixed-focal cohorts

- Freeze V1 after checkpoint 21. Its source-update-20 rollout completed under C002; no V1 files will receive further training metrics or checkpoints.
- Treat opponent policy identity, not focal deck identity, as the CUDA resident cohort boundary. Keep each focal deck ID and own-archetype ID bound to its resident job/lane and preserve exact-deck static-field audits.
- Preserve separate complete Policy-0809 and Champion-G1 collectors and immutable effective weights. No cross-policy tensor, cache, or routing sharing is introduced.
- Accept the batching change after an identical 99-game U20 Policy-0809 comparison showed 2.96 to 4.47 games/s (1.51x), with exact outcome/turn/error/routing parity, plus a 64-game mixed-deck smoke with zero error, routing failure, or feature D2H.
- Allocate `V2_mixed_focal_cohorts` as a new version and W&B run. Load V1 update 21 as model-only parent, initialize a fresh optimizer, carry the C002 PFSP state, and keep reference KL anchored to the original G1/update-0 model rather than silently re-anchoring it to update 21.

## 2026-08-13: stop V2 and evaluate rollout-selected checkpoints

- Stop V2 on the user's instruction after source update 206 produced model-only checkpoint U207. Do not append more updates to this repository version; any continuation requires a new version, fresh optimizer, newly collected on-policy data and a new W&B run.
- Select U57, U103, U136 and U190 from the actual rollout history instead of arbitrary round-number checkpoints. They represent, respectively, the early deck-002 peak, the largest deck-007 point spike, the strongest joint rolling region and a late stable region.
- Evaluate each selected checkpoint with both focal deck 002 and focal deck 007 against complete frozen Policy-0809 using the formal `kaggle_fp16_storage_fp32_runtime_v1` candidate contract and CUDA Engine 2.0 CUDA-2048 schedule. Keep all eight reports independent; do not merge their identities or outcomes.
- Accept all eight reports as valid Frozen evidence: each completed 2,048/2,048 official-engine games with zero error, unfinished game, routing failure or feature D2H copy, and passed candidate, opponent, deck and CUDA identity gates.
- Retain U190 as the best tested single-checkpoint compromise and the leading deck-002 candidate. Retain U136 as the preferred deck-007 candidate, with U57 as an equal raw-win-rate alternative. Treat these as descriptive candidate rankings only: two-sided checkpoint comparisons versus U20 are not statistically significant, and no automatic Promote decision follows.

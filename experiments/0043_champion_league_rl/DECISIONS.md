# 0043 Decisions

## 2026-08-14: publish Benchmark V1 as the Champion-G2 mirror league

- Define every opponent as the fixed pair `(exact deck 001–067, Champion-G2)`; policy identity never varies inside Benchmark V1.
- Run exactly 2,048 official CUDA Engine 2.0 greedy games per focal policy/deck identity.
- Balance sampling first across the 28 non-empty `own_archetypes_v2` classes (73 or 74 games each), then across exact member decks (counts differ by at most one), with seeded remainder rotation and shuffle. Class 14 Other has no registered deck and remains explicit `n=0`.
- Bind engine/search/policy/toss seeds to focal deployment identity, immutable G2 identity, exact opponent deck and schedule slot. The seeded toss winner remains responsible for context-41 first/second choice.
- Publish each focal report below `docs/evaluation/combat_mat/benchmark_v1/<focal-policy-and-deck>/`, with a parallel top-level index, Meta and exact-deck win rates, sample counts and representative card thumbnails.
- Use Champion-G2 as the first focal policy for decks 002, 003, 007 and 009; each is an independent CUDA-2048 report.

## 2026-08-14: human PROMOTE U407 as immutable Champion-G2

- Record the user's explicit `PROMOTE` decision for `Candidate-0043-U407`; no score or automation performed the promotion.
- Freeze V6 model-only checkpoint U407 as `Champion-G2` under `assets/policies/definitions/champion_g002/`, using semantic generation naming rather than a hash directory.
- Bind the decision to the complete Full67 evidence: 67 exact focal decks, independent Champion-G1 and U407 arms, 2,048 official CUDA Engine 2.0 games per arm, frozen Policy-0809 opponent, and `kaggle_fp16_storage_fp32_runtime_v1` deployment semantics.
- Preserve Policy-0809 and Champion-G1 byte-for-byte. Set the current active pool to `Policy-0809`, `Champion-G1`, `Champion-G2`, and move the latest-Champion pointer to G2.
- Keep G2 resident once per policy cohort. Route exact deck identity and the 29-way own embedding row per lane; do not create 67 model copies or share mutable tensor/cache storage across policies.
- Keep the opponent Meta classifier frozen at 15 classes. The promotion changes only the admitted own-policy identity and league composition; it does not mutate the value network's opponent taxonomy.

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

## 2026-08-13: prepare V3 generalist focal training from V2 U207

- Allocate the new immutable version `V3_generalist_focal_001_067`; never append this new training distribution to V2. Bind its parent to the latest V2 model-only checkpoint U207 and final V2 PFSP state. U136 remains only the selected deck-007 submission/evaluation candidate.
- Initialize a fresh optimizer, recollect all on-policy data and keep reference KL anchored to the original G1/update-0 model. Preserve rollout 256, PPO parameter groups/hyperparameters/losses, PFSP 128/64/64, complete Policy-0809 and Champion-G1 opponents, CUDA Engine 2.0, FP32 training and model-only all-checkpoint retention.
- Allow focal policy lanes to use every exact deck 001–067. Give each deck three lanes per rollout, rotate the remaining 55 lanes evenly through the registry, and seed-shuffle the final 256 lanes. This provides random lane switching, all-deck coverage every update and exact equality over each 67-update rotation.
- Keep `focal_deck_id` and its 29-way own-archetype ID as lane-bound feature inputs. Continue grouping resident CUDA inference only by independently materialized opponent policy; never create 67 focal-deck resident cohorts.
- Accept pre-launch readiness after 67 schedules produced 17,152 correctly bound lanes with 256 focal games per deck, and a real U207 256-game CUDA smoke completed all games across 67 focal decks and both opponent policies at 5.27 games/s with zero routing failure or feature D2H. Do not interpret this smoke as Frozen strength evidence.
- Launch V3 after explicit user authorization. Two environment-only service attempts failed before any rollout metric or PPO update: first from the wrong Python interpreter, then from a systemd PATH without the compiler. Permit reuse of the stable V3/W&B identity only through a hard pristine-restart gate requiring zero metrics, exactly the U207 parent checkpoint, exact parent/config hashes and no later checkpoint; once U208 exists, the gate rejects every restart as intended.
- Accept the first formal update as healthy: source U207 completed 256/256 sampled games at 127-127-2, emitted model-only U208, recorded behavior KL 0.000276, reference KL 0.01233 and clip fraction 0.00826, and retained zero routing failure and feature D2H. Continue the infinite-update service; do not reinterpret this sampled rollout as Frozen strength.
- Archive V3 after its source-U208 PPO lost the CUDA driver context at epoch 1 step 7 and archive V4 after a clean U208/fresh-optimizer retry reproduced the same driver error at epoch 1 step 1. Neither emitted U209 or a second canonical metric. Do not resume either model-only run in place.
- Allocate `V5_generalist_focal_001_067_u208_micro512` from complete U208. Keep logical minibatch 2,048, three epochs, 27 optimizer steps, loss weights, LR and every sampling/model semantic unchanged; reduce only the physical forward/backward graph from 1,024 to 512 and accumulate four graphs per logical step. This lowers observed PPO peak from about 15.1 GiB to 11.2 GiB without changing the effective optimization batch.
- Move PFSP persistence after successful PPO/checkpoint creation. For V5, use V2's final committed PFSP state to avoid inheriting the failed V3/V4 U208 rollout observations; accept omission of the single successful V3 U207 observation batch as the conservative alternative. C020 is unchanged during its current ten-update window.
- Accept V5 U209 as healthy: 27/27 optimizer steps completed, model-only U209 was emitted, source-U208 sampled rollout was 126-129-1, behavior KL 0.000172, reference KL 0.01131, clip fraction 0.00492, routing failures 0 and feature D2H 0. Continue the infinite service under the new stable W&B run.

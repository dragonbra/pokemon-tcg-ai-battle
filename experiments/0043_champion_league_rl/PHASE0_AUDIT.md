# 0043 Phase 0 Repository Audit

Date: 2026-08-12  
Status: `PASS_PHASE0`; Phase 1 asset import implemented, later training gates remain closed.

## Audited active baseline

1. Active training entrypoint: `python3 -m train.0042_full_model_design.training.run_full_semantic`.
2. Approved config source: `rl_runs/0042_full_model_design/versions/V22_archaludon_ex_cinderace_048/artifact/training_config.json`.
3. Frozen 0043 config: `experiments/0043_champion_league_rl/active_training_config.json`.
4. Frozen config file SHA-256: `fe6e1862a9f8660b3b19505898dacf55a2955a07385958639a5cb1f80d94264c`.
5. Formal update contract: 256 games, 256 rollout batch, 256 retained trajectories, PPO Protocol V2, at most three complete shuffled passes without replacement, logical minibatch 2,048 and physical forward microbatch 1,024.
6. Approved actor learning rates: decoder `2e-5`, policy strategy adapter `4e-5`, allocation head `2e-5`. Value and prize groups remain `1e-4`.
7. Approved PPO values include `gamma=1`, `GAE lambda=0.95`, clip `0.1`, entropy coefficient `0.003`, reference-KL coefficient `0.02`, target behavior KL `0.015`, and hard behavior-KL guard `0.025`.

0043 may not infer approval from the defaults in the old parser. Its regression guard uses the frozen emitted V22 config because the parser defaults still contain older `5e-6` actor values.

## Current asset paths and identities

| Domain | Audited source | Identity |
|---|---|---|
| 55-deck schedule | `train/0042_full_model_design/league/frozen_catalog.json` | file SHA `b1147f...93d2c`; 55 exact decks; 256 games |
| Policy-0809 | `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt` | file SHA `926321...7c7f`; effective SHA `0d0091...da96` |
| Champion-G1 delta | `archive/pretrained/0042_champion_g1/source_update_000010.pt` | file SHA `aea408...fe3` |
| Champion-G1 portable | `archive/pretrained/0042_champion_g1/fp16_reference_048_package/strategy/model.bin` | file SHA `cd5c05...e84`; deployment-effective SHA `0804ca...7c5` |
| Champion designation | `archive/pretrained/0042_champion_g1/manifest.json` | `USER_DESIGNATED_G1_NOT_AUTOMATIC_PROMOTE` |

The G1 source is therefore explicit in the repository; 0043 does not guess a checkpoint. Importing the already user-designated anchor is not a new Promotion decision.

## Existing routing and evaluation paths

- The old opponent loader is `train/0042_full_model_design/policy_identity.py`; it materializes complete immutable Policy-0809.
- The old 55-deck schedule builder is `build_jobs()` in `training/run_full_semantic.py`, backed by `league/catalog.py`.
- The formal Frozen path uses `evaluation/frozen_jobs.py` and the same training runner.
- Current W&B logging flows through `rl_environment.logging.TrainingLogger`; it already retains PPO, KL, value, entropy, gradient, optimizer coverage, rollout runtime and engine-turn health metrics.
- Existing episode aggregation does not yet satisfy all V2 prize split, win/loss turn, PFSP entropy/tail, and curriculum-boundary fields.

## Dependency findings

- 0042 runtime code imports numerous project-local model, semantic policy, rollout, action-boundary and deployment modules; 0043 cannot import those executable modules.
- Approved 0809/G1 weights and exact decks were historical assets outside 0043. Phase 1 imports them into content-addressed 0043-local paths.
- Engine runtime, `rl_environment/`, `evaluation/`, and official card data remain allowed shared infrastructure.
- The existing Frozen pool directory is about 307 MB because it embeds historical policy runtimes. 0043 imports only its exact deck composition and separately registers Policy-0809, avoiding that ambiguous coupling.

## Phase 1 copy/refactor inventory

- Imported now: exact decks under canonical identities `001`–`055`, Policy-0809 under `policy_0809`, G1 under `champion_g001`, FrozenMeta256-V1 composition and seed contract metadata. Hashes are integrity evidence rather than directory identities.
- To port before training: the minimum actor/value/model codec, policy resolver, semantic compiler, action-boundary runtime, CUDA collector/routing, PPO trainer, storage/logger adapter, and candidate materializer.
- To implement new: 128/64/64 sampler, PFSP state/curriculum manifests, V2 telemetry, FrozenMeta evaluator adapter, and manual Promote V2 commands.

## Explicit open gates

- The G1 complete component-level reconstruction/no-hybrid runtime audit is Task 3, not claimed by Phase 1 file-hash validation.
- FrozenMeta seed blocks and old/new official-engine parity are Task 8. The imported seed contract says `SCHEDULE_MATERIALIZATION_PENDING_PHASE5` until then.
- No 0043 PPO run is authorized or launchable yet. Project status remains `BLOCKED_PRETRAINING_CONTRACT` until Tasks 3–8 pass.

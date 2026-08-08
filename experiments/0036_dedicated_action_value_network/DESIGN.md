# 0036 Dedicated Action Value Network

Status: **the all-date, recency-weighted Dragapult focal dataset and its mmap tensor cache are complete and verified. V1 exposed a gzip/JSON input bottleneck and was intentionally stopped before completing epoch 1. The first production-target run is now V2's full latent-query critic; no offline result or RL strength claim exists yet.**

## Evidence boundary

General game rules follow `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`: attacks end the turn, and a game can end by taking all Prize cards, leaving the opponent without a battle-capable Pokémon, or deck-out at the start-of-turn draw. Exact observation fields and terminal state come from unmodified official-engine Episode replays. The targets, weighting, archetype taxonomy and critic architecture below are 0036 project choices, not official game rules.

## Goal and frozen lineage

0036 trains a dedicated critic before PPO instead of asking a small initial on-policy rollout to learn Value from scratch. Its deployment target is 0034's Frozen-0806 exact deck 007, `dragapult_ex_07bedfffbfad`, exact-deck SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`. It snapshots the 0031 friend-0806 epoch-11 model-only checkpoint at `rl_runs/0036_dedicated_action_value_network/source/friend_0806_epoch11/model.pt` (SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`, 56,352,322 parameters). The self-contained model, compiler, causal-knowledge code and prototype assets live inside `train/0036_dedicated_action_value_network/`; no numbered training project is imported at runtime.

All source-policy parameters, including prototype embeddings and the semantic Encoder, are frozen and kept in eval mode. Only the new Value head is optimized.

## Dataset and label contract

The formal small dataset targets 10,000 unique **Episodes containing at least one focal player**. A player qualifies only when the registered exact 60-card multiset contains Dragapult ex card ID 121. Only that player's pre-action callbacks are supervised; the opponent perspective is excluded unless the opponent independently has card 121. Dragapult mirror Episodes count once toward the 10,000-Episode budget but contribute two focal trajectories, and both remain in one split.

Selection scans every locally acquired official daily archive at or before the 2026-08-06 frozen endpoint, without an early-date cutoff. Every date with eligible games receives at least one Episode. The remaining budget uses a deterministic mixture of 25% uniform-over-date weight and 75% exponential recency weight with a seven-day half-life; dates that lack enough candidates donate unused quota to other dates. Episode selection within a date is SHA-ranked. If fewer than 10,000 unique eligible Episodes exist, all are retained and the shortfall is reported; Episodes are never duplicated. The split unit is the entire Episode and stratifies date, focal outcome, opponent archetype and exact-007 membership.

The verified formal catalog covers 30 continuous dates from 2026-07-08 through 2026-08-06. It contains exactly 10,000 unique Episodes, split 9,000/1,000, and 10,155 focal player trajectories because 155 mirror Episodes contribute two eligible perspectives. Focal outcomes are 5,338 wins and 4,817 losses. Exact 007 contributes 1,951 trajectories, including 190 in validation. A deterministic same-date whole-Episode swap guarantees all 15 opponent classes appear in validation without changing Episode selection or counts. The immutable catalog commitment is `4e8148e672079ef5fc0c6ef65ffa1ae6340886b1cb65d9696dff082dd9da1d47`.

The verified local action dataset contains 1,061,249 real pre-action callbacks: 954,942 train and 106,307 validation records in 260 gzip shards (about 1.1 GB). All shard hashes and row counts pass, all 10,155 trajectory weights sum to one, and validation contains all 15 opponent classes, including 125 action samples for the rarest admitted class. Its manifest SHA-256 is `24cd4d89bd89d0dae0f0fd71281a40e8cf6142e7dc41811f3819e39011fe44ef`; the verification-report SHA-256 is `c5e27d6a5585e46e9c3d2c62cf1dcab0508e035f5e51f4ec975e0718f3084ba3`. The full dataset remains under the Git-ignored `rl_runs/0036_dedicated_action_value_network/datasets/` path.

The completed generic four-day corpus remains superseded pipeline evidence: 10,000 Episodes, 1,588,444 action decisions, 1.6 GB, and catalog commitment `4228c7e2d3c57d24c8cc1094764c80306590160a753dbb6e5ae279cd5d0d9374`. No formal training version consumed it. Its local shards remain Git-ignored.

The verified JSON corpus is the immutable semantic source, but it is not decoded in the epoch loop. A one-time, hash-committed materialization converts the 260 source shards to 260 mmap-compatible PyTorch tensor shards at `rl_runs/0036_dedicated_action_value_network/datasets/dragapult_focal_all_dates_weighted_exact10k_tensor_v1/`. Integer actor fields and class labels are stored as `int16`, actor numeric fields as `float16`, masks as `bool`, while `value_target` and `episode_weight` remain exact `float32`; runtime batches restore the original `long`/`float32`/`bool` contract. The cache contains the same 954,942/106,307 train/validation records, occupies 17,223,219,032 bytes, and commits to source manifest SHA-256 `24cd4d...44ef`. Its reference content commitment is `02e6651e1bf98262837c65b07e568ace6a79be4b77b682619e20c8dd5f3fb236`.

Training shuffles tensor-shard order and row order deterministically per epoch, reads shards with `torch.load(..., weights_only=True, mmap=True)`, selects one batch, restores runtime dtypes, pins host tensors, and keeps two prepared batches in a bounded producer queue. It never inflates the whole 17 GB cache in RAM. A 100-batch full latent-query GPU benchmark measured 0.2402 seconds/batch and 2.21% data-wait fraction, versus about 1.10 seconds/batch for the former sequential gzip/JSON path. These are throughput diagnostics, not model-quality evidence.

One record is emitted for every real non-registration `agent(observation)` callback, before its complete ordered action. Registration callbacks, engine automatic transitions and decoder sub-tokens emit no labels. Each record contains the unchanged 39-key actor feature contract plus:

| target | shape | definition |
|---|---:|---|
| `value_target` | scalar `{0,1}` | eventual loss/win from acting player's perspective |
| `archetype_target` | scalar integer `[0,14]` | opponent's 14-axis meta class or Other |
| `final_diff_target` | scalar integer `[0,12]` | signed final Prize difference `-6..+6`, shifted by 6 |
| `episode_weight` | scalar float | reciprocal of this focal actor's action-decision count in the Episode |
| `is_exact_007` | scalar bool, metric-only | complete registered multiset equals Frozen-0806 exact 007 |

For final Prize difference, the winner's remaining Prize count is set to zero regardless of win condition, while the loser's observed remaining count supplies the magnitude. Thus winner and loser labels are sign reversals. The weighting makes every player trajectory contribute total weight one, avoiding domination by long games while still supervising every deployment-time decision boundary.

Exact deck hashes, `is_exact_007`, player/team identity, source archive, action target and payload hashes are audit/metric-only. They cannot enter the forward path. The ordinary actor-visible resource ledger already describes the focal player's exact registered deck; no new deck-ID token is added.

## Actor feature shapes

0036 preserves the 0031 semantic schema and 39 tensor/mask keys. Variable axes are batch-padded.

| family | categorical width | numeric width | field-state width | relation inputs |
|---|---:|---:|---:|---|
| global | 12 | 24 | 24 | none |
| card / resolved Energy | 9 | 7 | 7 | parent |
| own resource ledger | 4 | 15 | 15 | none |
| bounded event | 31 | 4 | 4 | source, target, before, after |
| legal option | 19 | 2 | 2 | source, target, context, effect card, skill, effect |

The Encoder produces `state.tokens [B,S,320]`, `state.mask [B,S]`, `state.summary [B,320]`, encoded legal options `[B,O,320]`, and `option_mask [B,O]`.

## Network data flow

```text
39 causal actor tensors
  -> frozen semantic Encoder
       state tokens/mask/summary + legal-option tokens/mask
  -> one of three Value trunks
       raw_pool_mlp: masked mean(state + options) -> MLP
       summary_mlp: pretrained state.summary -> MLP
       latent_queries: 8 learned queries
         -> 2 x pre-LN cross-attention over state + legal options
         -> latent self-attention -> FFN
  -> query 0: value logit [B]
     query 1: opponent archetype logits [B,15]
     query 2: final Prize-difference logits [B,13]
```

For MLP baselines the same representation feeds the three output heads. The reported scalar PPO-style value is `2 * sigmoid(value_logit) - 1`; supervised fitting and model selection use the numerically stable binary logit loss.

## Objective and metrics

The objective is:

`L = BCE_value + lambda_class * CE_archetype + lambda_diff * CE_final_diff`

All terms use `episode_weight`. Hidden-hand prediction is off and structurally fail-closed because no fully causal partially observed label contract has been admitted. V1 raw pooled MLP was stopped before epoch 1 completed after its sequential gzip/JSON loader was measured as the dominant bottleneck. Per the production-priority decision, V2 trains the full latent-query Value plus archetype and final-diff auxiliaries (`lambda_class=lambda_diff=0.1`) first on the optimized cache. Summary MLP, latent-query Value-only, latent plus archetype, and a rerun of the raw-pool baseline remain follow-up ablations and must receive fresh monotonically increasing versions.

Checkpoint selection is lexicographic: lowest episode-weighted **exact-007 validation Value BCE** first, then lowest all-Dragapult validation Value BCE. Brier score, ECE, accuracy and AUROC assess calibration/discrimination; archetype accuracy and final-diff exact accuracy/MAE diagnose representation learning but cannot override worse exact-007 Value BCE. The exact-007 mask is used only after forward for metric slicing. Metrics are computed once online during the train optimization pass and once with the fixed epoch-end model over the full validation split.

## Training and artifact contract

Each epoch performs exactly one shuffled train update pass and one complete, non-updating validation pass. AdamW updates only Value-head parameters. CUDA bfloat16 autocast is the default. Every epoch writes a model-only checkpoint, with retention `all`; optimizer, scheduler, scaler, RNG, loader position and replay state are forbidden.

Formal runs use strictly immutable versions. `V1_raw_pool_mlp` is a retained failed performance-preflight record with no completed epoch, metric row, or checkpoint; it cannot be reused. `V2_latent_value_archetype_diff` is the first production-target training version and uses 30 epochs so every epoch yields a model-only checkpoint while the lexicographic validation selector retains a stable best pointer. After successful V2 completion, the fail-closed queue is V3 optimized raw-pool MLP, V4 summary MLP, V5 latent Value-only, and V6 latent plus archetype, each for 20 epochs. `training_metrics.jsonl` is written before TensorBoard and W&B online mirror data. The private W&B target is `dragon_bra/pokemon-tcg-policy-learning`; display names begin with `0036`. A mirror failure does not erase local evidence and must remain visible in status metadata.

## RL boundary and next stage

The selected head will initialize exact-007 PPO Value/GAE prediction while the actor continues from the same frozen semantic representation. Offline targets estimate return under a mixture of historical Dragapult behavior policies and nearby Dragapult deck variants; they are not universally correct for exact 007 after policy improvement. PPO must therefore continue on-policy critic calibration. A separate post-ablation exact-007-only calibration version is admissible only if held-out exact-007 BCE improves. Value deltas may be logged for action-level diagnostics but do not become shaped environment rewards without a separate reviewed experiment.

The immediate gates are: complete V2, inspect exact-007 held-out Value BCE and calibration across all retained epochs, export the selected head with logit parity, then perform a small exact-007 official-engine rollout calibration smoke. Follow-up ablations quantify the auxiliary/trunk effects but no longer block producing the first usable full critic. Only frozen official-engine evaluation can support an RL policy-strength conclusion.

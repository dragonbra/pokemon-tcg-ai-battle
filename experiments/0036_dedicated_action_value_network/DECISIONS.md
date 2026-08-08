# 0036 Decisions

## 2026-08-08 — V1 contracts

- The supervised unit is one real, non-registration `agent(observation)` callback immediately before the complete ordered action. It is not a turn, engine auto-step, or pointer-decoder token.
- The primary target is undiscounted terminal win probability under the behavior policy. Every action decision in a player trajectory shares the terminal outcome, while its weight is `1 / number_of_actor_decisions` so each player trajectory contributes total weight one.
- Signed final Prize difference uses the terminal convention requested for alternate win conditions: the winner's remaining Prize count is treated as zero; the loser's observed remaining Prize count defines magnitude. The 13 classes represent `-6..+6` from the acting player's perspective.
- Opponent archetype is a 15-class training-only target: 14 explicit meta axes plus Other. Exact deck, team/source identity and opponent label are never actor-visible inputs.
- Opponent hidden-hand prediction is disabled in the initial experiments. A nonzero hidden-hand loss weight fails closed because no causally valid target/schema has been admitted.
- The 0031 friend-0806 epoch-11 semantic policy is physically snapshotted for 0036 and loaded only after model-only schema, metadata, parameter count and SHA-256 checks. All Encoder weights remain frozen.
- Validation isolates whole Episodes. Both seats always remain in the same split. The primary model-selection metric is episode-weighted validation Value BCE; Brier score, ECE, AUROC and per-archetype results are guardrails.
- `V(next)-V(current)` is a diagnostic/GAE ingredient, not an environment reward. A selected offline critic estimates behavior-policy value and must be recalibrated on-policy as the actor changes.
- The initial newest-heavy 2026-08-04..06 quota failed closed because 2026-08-06 had 2,886 fully qualified Episodes, below its proposed 4,500. The admitted small-dataset window is therefore 2026-08-03..06 with an equal 2,500 quota per day.
- The first proportional Episode split left the two rarest archetype labels absent from validation. That catalog remains as failed audit evidence; split V2 uses deterministic same-date swaps to guarantee every observed class appears in validation without changing the selected 10k Episodes or splitting player perspectives.

## Planned immutable ablations

1. `V1_raw_pool_mlp`
2. `V2_latent_value_archetype_diff`
3. `V3_summary_mlp`
4. `V4_latent_value_only`
5. `V5_latent_value_archetype`

No version is considered successful until its dataset hashes, per-epoch validation, model-only checkpoints, TensorBoard and W&B mirror status are recorded. Offline Value quality is not a policy-strength claim; any RL strength conclusion requires frozen official-engine evaluation.

Every formal W&B display name is explicit and begins with the project number: `0036 · dedicated_action_value_network · <version>`. Repository version, stable W&B run ID and private project remain separate auditable fields.

0036 eagerly initializes W&B after model/dataset/logger setup and before the first train pass. This makes the run configuration and W&B system telemetry visible while a long first epoch is still executing; canonical scalar training metrics remain one train/validation snapshot per completed epoch.

Long train/validation passes also emit a lightweight console progress record every 100 batches with epoch, split, decisions, elapsed seconds and throughput. These records are observability only: they do not add optimizer passes, validation passes or extra W&B scalar snapshots.

The full latent-query critic with both admitted auxiliary losses is promoted to V2, immediately after the raw-pool MLP baseline. This scheduling change prioritizes having a deployable full critic candidate for the next RL interface check; the summary, Value-only and single-auxiliary variants remain later ablations and do not alter V1/V2 data, epoch or validation contracts.

## 2026-08-08 — Redirect formal corpus to Dragapult exact-007 deployment

- 0034's authority documents confirm the focal RL deck is Frozen-0806 exact 007, `dragapult_ex_07bedfffbfad`, not the historical Third list used by 0034 V1–V5.
- No generic 0036 training version had launched, so V1–V5 are redirected without rewriting run history. The completed generic 10k dataset remains local, ignored, superseded pipeline evidence.
- Formal selection now keeps only actor perspectives whose registered 60-card deck contains Dragapult ex card ID 121. An opponent perspective is never included merely because it faced Dragapult.
- Exact 007 is recognized only by complete multiset equality and is metric/audit-only. No exact-deck identity token, source identity or persona is added to forward.
- The initial five-way comparison does not oversample exact 007. Checkpoints are selected by exact-007 validation Value BCE first and all-Dragapult Value BCE second. Any later exact-007-only calibration requires a new version and held-out improvement.
- The focal budget is 10,000 unique Episodes rather than 10,000 focal trajectories. A Dragapult mirror consumes one Episode slot but supplies both eligible player trajectories.
- Earlier dates are not truncated. Every available date with an eligible Dragapult Episode remains represented; allocation mixes 25% uniform-over-date weight with 75% exponential recency weight using a seven-day half-life. Sparse dates donate unused capacity, within-date choice is deterministic, and no Episode is duplicated.
- The initial local 2026-07-10..08-06 scan produced 9,787 Episodes, and adding 07-09 produced 9,994. Both shortfall catalogs are retained as audit evidence. Adding the adjacent official 07-08 archive produced the final exact 10,000-Episode, 30-date catalog with commitment `283ae4adc9f30b6110cc8c6780dde15a8c85f5fba7b28e8b6ef68009a2992010`.
- The first focal split omitted rare opponent class 8 from validation. Its catalog, materialized shards and verification report remain preserved with the `split_v1_missing_class8` suffix. A deterministic same-date whole-Episode swap fixes validation to 15/15 classes while preserving Episode selection, 9,000/1,000 counts and 190 exact-007 validation trajectories. The admitted catalog commitment is `4e8148e672079ef5fc0c6ef65ffa1ae6340886b1cb65d9696dff082dd9da1d47`.

## 2026-08-08 — Stop V1, remove realtime gzip/JSON from training, run the full critic first

- V1 started as `V1_raw_pool_mlp` and initialized its online W&B run, but it was intentionally interrupted before epoch 1 completed. It remains immutable with failed status, an empty canonical metrics file, no checkpoint, and its synced W&B identity. It cannot be resumed or overwritten.
- Root-cause timing separated input from compute: the sequential gzip/stdlib-JSON loader averaged about 1.10 seconds per batch and spiked to about 2.8 seconds whenever a 4,096-row shard was opened, while a warmed raw-pool GPU train step was about 0.14 seconds. Realtime decompression and Python object collation were therefore the dominant bottleneck.
- The admitted optimization is a one-time compact tensor materialization, not an in-memory copy of the entire dataset. Source shard hashes and manifest SHA remain provenance; output tensor shards are separately hash committed, loaded with safe `weights_only=True` and `mmap=True`, deterministically shuffled per epoch, restored to the original runtime dtypes, pinned, and bounded-prefetched at depth two.
- The tensor cache has 260 shards, exactly 954,942 train and 106,307 validation actions, 17,223,219,032 payload bytes, and reference content commitment `02e6651e1bf98262837c65b07e568ace6a79be4b77b682619e20c8dd5f3fb236`. It remains under the Git-ignored project dataset path.
- A 100-batch end-to-end full latent-query benchmark measured 0.240157 seconds/batch, 4,263.87 actions/s, 0.52999 seconds total data wait, 2.2068% data-wait fraction, and 4.741 GiB peak allocated GPU memory. This passes the performance gate but makes no quality claim.
- User priority supersedes the former baseline-first schedule: `V2_latent_value_archetype_diff` is the first run after the I/O gate and trains the full SOTA candidate with both auxiliary losses. It uses 30 epochs and retains a model-only checkpoint from every epoch. Remaining ablations use later unused versions and do not delay V2.
- To keep the six-hour GPU window productive without delaying V2, successful V2 completion gates a fail-closed sequential queue: `V3_raw_pool_mlp_io_optimized`, `V4_summary_mlp`, `V5_latent_value_only`, then `V6_latent_value_archetype`, each for 20 epochs with the same cache, split, optimizer contract and per-epoch model-only retention. A failed or interrupted version stops the queue; no later result is fabricated or appended to an older version.

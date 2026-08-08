# 0034 Dragapult Third Large Model RL Design

## Scope and stage

0034 now learns exact Frozen-0806 deck 007, `dragapult_ex_07bedfffbfad`, exact-deck SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`. V1-V5 used the historical Third PTCG Club Dragapult list and remain immutable records. Initialization remains the 0806 large-model checkpoint: 0031 `V2_full_winners_bs1024_20260616_20260803`, Epoch 11 / step 141878, SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`. The sampling frame remains the immutable Frozen-0806 55-deck, 256-slot distribution; V7 samples 128 slots/seeds per update and evaluates each in both seats, for 256 Episodes.

V1 is the historical pre-RL 510-game Frozen 0019 baseline and completed 257-253-0 (50.3922%) with zero error. V6 is the most recent historical PPO run for exact 007. V7 is allocated in code but has not been launched. Official rules, legal actions and terminal results still come from the official CPU engine implementation; the local seeded ABI is compiled around the untouched official source and changes initialization control, not game rules. Feature interpretation, value estimation, seed pairing, GAE and PPO are project choices. Source/team identity is provenance and never actor-visible.

## Actor-visible input and action contract

Every decision uses `0031_rule_faithful_semantic_decision_v2` with all 39 actor keys and no reduced fallback codec.

| Family | Ragged tensors | Contents |
|---|---|---|
| Global | `[B,12]` categorical; `[B,24]` numeric and state | turn, selection, zones and visible game facts |
| Card | `[B,C<=160,9]` categorical; `[B,C<=160,7]` numeric and state | instances, zones, HP, attachment and resolved Energy facts |
| Resource | `[B,R,4]` categorical; `[B,R,15]` numeric and state | exact-deck ledger and known/bounded zone counts |
| Event | `[B,E,31]` categorical; `[B,E,4]` numeric and state | chronological official logs and typed relations |
| Option | `[B,O,19]` categorical; `[B,O,2]` numeric and state | every legal option, source, target, context and prototype facts |
| Relations | ragged one-based indices plus masks | card parents, event links and option source/target/context/effect-card links |
| Selection | `[B]` minimum and maximum | ordered unique legal-option pointer sequence plus stop |

Each game/player owns independent causal knowledge: registered deck, visible and possible hand, resource ledger, chronological events and card-instance history. Immutable prototype tables are the only shared read-only state. A missing key, tensor mismatch, illegal selection, worker error or incomplete Episode fails closed.

The CUDA semantic0031 codec reserves 160 ragged card rows, matching the largest audited training bucket. This is intentionally separate from the legacy POD-native codec's fixed 128-entity ABI: legal late-game states can exceed 128 semantic rows after attached cards and resolved Energy units are expanded. Rows 129-160 prevent observation collection from incorrectly converting such a legal state into an engine error; masked padding remains actor-invisible.

## Network and trainable boundary

The source actor is the exact 56,352,322-parameter SemanticPolicy with `d_model=320`, eight heads, four hierarchical state Transformer layers, one event layer, two option cross-attention layers and the original autoregressive GRU pointer decoder. All 293 source tensors load strictly. Its frozen representation SHA-256 is `53d5bf2043304cb636e978e7bf0dc49fc51b7b80b012a634d78b0fa9708e33a7`.

Only `actor.action_decoder.*` (1,027,202 parameters) and a separate critic (103,681 parameters) are trainable, 1,130,883 total. The critic reads the frozen state summary and never changes actor logits. Each model-only checkpoint stores decoder/value state, schema, update and hashes; it excludes optimizer, scheduler, RNG, rollout and replay state. Updates 0 through 500 are retained.

## Reward, credit and PPO

Net reward is only the official terminal result: win `+1`, draw `0`, loss `-1`; all intermediate rewards are zero. There is no Prize, damage, attack or setup shaping. `gamma=1.0`. V6 restores the audited 0023 selection-clock credit assignment: every adjacent focal decision applies lambda `0.95`, including decisions in the same official turn. Each Episode receives total loss mass one, split equally over its focal decisions (`episode_equal_decisions`). This temporal-credit rule is a project optimization choice, not an official TCG rule.

PPO uses four epochs, minibatch 1024, decoder LR `1e-5`, critic LR `1e-4`, clip ratio `0.1`, value coefficient `0.5`, entropy `0.01`, reference-decoder KL `0.02`, target behavior KL `0.02`, max gradient norm `0.5` and zero weight decay. These match 0023 V2. V7 samples 128 engine seeds and evaluates each in both seats, producing 256 Episodes per update. Behavior log probability must replay with mean absolute error at most `1e-4` before an update.

The resident CUDA implementation keeps rollout feature tensors, ordered action targets, complete-Episode filtering, terminal reward construction, GAE, Episode-equal weights, advantage normalization and PPO minibatch indexing on the source CUDA device. Only scalar monitoring fields and model-only checkpoints cross to CPU. Terminal lanes are reset in place to disjoint deterministic seeds during collection rather than waiting for the batch. This is a runtime-residency optimization only: it does not change the feature schema, action sequence, reward, credit rule, loss or checkpoint boundary, and it is not evidence that V7 has been launched.

## Opponents, schedule and evidence

V7 samples 128 slots without replacement from the 256 immutable scenario slots in `0806_kaggle_top100_plus_v1`, schedule SHA-256 `16dbd18ce417405571c88997c9e97f9b2ec2adf96db544d1a9af988bb3c3cc3c`. The same run seed and source-policy update reconstruct the same slot sample and 128 unique engine seeds. Each sampled seed emits two adjacent Episodes with fixed physical deck slots: focal is always player 0, opponent is always player 1, and the harness chooses focal-first then focal-second through the official setup selection. The pair shares the engine seed, Search seed, opponent deck and focal deck. Stochastic focal actions use a distinct per-Episode `torch.Generator`, so sampling is independent of batch coalescing order. This is paired environment randomness, not stronger action-indexed CRN: divergent actions may consume later random events differently.

The seeded runtime is locked to `engine/build/seeded_official/0001/libcg.so`; its manifest records the official source hash, external adapter hash, compiler identity, ABI and library hash. `engine/source/` remains untouched. Evaluation and RL both resolve this path through the same builder and pass it explicitly to workers. Search uses its own seeded `AgentStart` state and does not consume the live battle RNG; search branches inside one Search session still share that search session's RNG according to the official implementation. Opponents use an independent immutable copy of the Policy-0806 pretrained checkpoint, SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`; this copy never receives PPO updates.

Canonical metrics write in JSONL, TensorBoard, then W&B order. A row at `trainer/update=k` reports sampled rollout generated by `rollout/source_policy_update=k-1` and the resulting `checkpoint/update=k`. Training uses 128 newly sampled seed pairs per update; frozen greedy probes reconstruct the same fixed 128-seed/256-Episode suite across checkpoints. `rollout/*` remains stochastic on-policy diagnosis and `eval/*` remains a frozen greedy diagnostic. Final selection requires a separate zero-error official-engine evaluation under the same runtime, schedule and policy hashes.

## Current and next stage

V1 and V2 remain immutable historical evidence under the old Frozen-0019 contract. V3 was stopped before update 1 after discovering that its opponent path still used Policy-0019; V4 was stopped after one 256-game batch exposed an unbounded repeated-selection path. V5 `V5_policy0806_frozen_256g_500u_termination_guard` completed 76 updates before user-requested stop and remains negative evidence. V6 `V6_exact007_0023_selection_lambda095` is immutable historical evidence under the unseeded, globally alternating seat contract. The next unlaunched version is `V7_seeded_paired_official_engine`: runtime `0001`, fixed deck slots, 128 sampled seed pairs / 256 Episodes, per-game policy RNG, unchanged terminal reward, repeat-forfeit and 50-round draw rules. No V7 training or strength claim exists yet.

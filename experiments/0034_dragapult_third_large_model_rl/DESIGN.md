# 0034 Dragapult Third Large Model RL Design

## Scope and stage

0034 now learns exact Frozen-0806 deck 007, `dragapult_ex_07bedfffbfad`, exact-deck SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`. V1-V5 used the historical Third PTCG Club Dragapult list and remain immutable records. Initialization remains the 0806 large-model checkpoint: 0031 `V2_full_winners_bs1024_20260616_20260803`, Epoch 11 / step 141878, SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`. Opponents use the immutable Frozen-0806 55-deck, 256-game distribution.

V1 is the historical pre-RL 510-game Frozen 0019 baseline and completed 257-253-0 (50.3922%) with zero error. V6 is a fresh 500-update PPO run for exact 007. Official observations, legal actions and terminal results come only from the unmodified official CPU engine. Feature interpretation, value estimation, GAE and PPO are project choices. Source/team identity is provenance and never actor-visible.

## Actor-visible input and action contract

Every decision uses `0031_rule_faithful_semantic_decision_v2` with all 39 actor keys and no reduced fallback codec.

| Family | Ragged tensors | Contents |
|---|---|---|
| Global | `[B,12]` categorical; `[B,24]` numeric and state | turn, selection, zones and visible game facts |
| Card | `[B,C,9]` categorical; `[B,C,7]` numeric and state | instances, zones, HP, attachment and resolved Energy facts |
| Resource | `[B,R,4]` categorical; `[B,R,15]` numeric and state | exact-deck ledger and known/bounded zone counts |
| Event | `[B,E,31]` categorical; `[B,E,4]` numeric and state | chronological official logs and typed relations |
| Option | `[B,O,19]` categorical; `[B,O,2]` numeric and state | every legal option, source, target, context and prototype facts |
| Relations | ragged one-based indices plus masks | card parents, event links and option source/target/context/effect-card links |
| Selection | `[B]` minimum and maximum | ordered unique legal-option pointer sequence plus stop |

Each game/player owns independent causal knowledge: registered deck, visible and possible hand, resource ledger, chronological events and card-instance history. Immutable prototype tables are the only shared read-only state. A missing key, tensor mismatch, illegal selection, worker error or incomplete Episode fails closed.

## Network and trainable boundary

The source actor is the exact 56,352,322-parameter SemanticPolicy with `d_model=320`, eight heads, four hierarchical state Transformer layers, one event layer, two option cross-attention layers and the original autoregressive GRU pointer decoder. All 293 source tensors load strictly. Its frozen representation SHA-256 is `53d5bf2043304cb636e978e7bf0dc49fc51b7b80b012a634d78b0fa9708e33a7`.

Only `actor.action_decoder.*` (1,027,202 parameters) and a separate critic (103,681 parameters) are trainable, 1,130,883 total. The critic reads the frozen state summary and never changes actor logits. Each model-only checkpoint stores decoder/value state, schema, update and hashes; it excludes optimizer, scheduler, RNG, rollout and replay state. Updates 0 through 500 are retained.

## Reward, credit and PPO

Net reward is only the official terminal result: win `+1`, draw `0`, loss `-1`; all intermediate rewards are zero. There is no Prize, damage, attack or setup shaping. `gamma=1.0`. V6 restores the audited 0023 selection-clock credit assignment: every adjacent focal decision applies lambda `0.95`, including decisions in the same official turn. Each Episode receives total loss mass one, split equally over its focal decisions (`episode_equal_decisions`). This temporal-credit rule is a project optimization choice, not an official TCG rule.

PPO uses four epochs, minibatch 1024, decoder LR `1e-5`, critic LR `1e-4`, clip ratio `0.1`, value coefficient `0.5`, entropy `0.01`, reference-decoder KL `0.02`, target behavior KL `0.02`, max gradient norm `0.5` and zero weight decay. These match 0023 V2 except that each current rollout intentionally remains the user-defined 256-game Frozen-0806 distribution rather than 0023's 512-game league batch. Behavior log probability must replay with mean absolute error at most `1e-4` before an update.

## Opponents, schedule and evidence

The next fresh version uses exactly 256 complete official-engine Episodes per update. Its 55 unique exact decks and integer frequencies come from `0806_kaggle_top100_plus_v1`, schedule SHA-256 `16dbd18ce417405571c88997c9e97f9b2ec2adf96db544d1a9af988bb3c3cc3c`: 240 slots reproduce Top 100 exact-deck frequency and 16 slots cover selected ranks 101–500 potential decks. Global schedule order alternates the focal seat for exactly 128 first-seat and 128 second-seat games. Opponents use an independent immutable copy of the Policy-0806 pretrained checkpoint, SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`; this copy never receives PPO updates. Each opponent deck has its own exact-deck online encoder while requests share one resident GPU model. Execution keeps isolated official-engine workers and shared CUDA inference.

Canonical metrics write in JSONL, TensorBoard, then W&B order. A row at `trainer/update=k` reports sampled rollout generated by `rollout/source_policy_update=k-1` and the resulting `checkpoint/update=k`. Both training rollout and frozen greedy probe use the same 256-slot distribution; `rollout/*` remains stochastic on-policy diagnosis and `eval/*` remains a frozen greedy diagnostic. V6 records a greedy update-0 baseline and every ten-update probe uses the identical engine seeds, seats, decks and order, so checkpoint probes are paired at the environment-schedule level. Final selection requires a separate zero-error 256-game Frozen-0806 evaluation under the same schedule and policy hashes.

## Current and next stage

V1 and V2 remain immutable historical evidence under the old Frozen-0019 contract. V3 was stopped before update 1 after discovering that its opponent path still used Policy-0019; V4 was stopped after one 256-game batch exposed an unbounded repeated-selection path. V5 `V5_policy0806_frozen_256g_500u_termination_guard` completed 76 updates before user-requested stop: its stochastic rollout remained roughly flat while its non-paired greedy probes fell from 49.61% at update 10 to 41.02% at update 70. V5 is retained as negative evidence; no gradient-sign defect was found. V6 `V6_exact007_0023_selection_lambda095` starts from Policy-0806 with a fresh optimizer, exact 007, the restored 0023 PPO setting, fixed comparable probes and the unchanged repeat-forfeit/50-round-draw termination contract.

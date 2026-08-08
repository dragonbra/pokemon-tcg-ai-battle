# 0037 Dragapult Value-Initialized RL

## Evidence boundary

General game rules follow `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`. Attacks end the acting turn; wins arise from Prize completion, no battle-capable opposing Pokémon, or start-of-turn deck-out. Exact observations, legal actions, terminal results and turn numbers come from the unmodified official engine rules executed through seeded runtime 0002. Value estimation, seed pairing, GAE, PPO, worker topology and evaluation splits are 0037 project choices.

## Objective and sources

0037 is the first PPO baseline using the dedicated 0036 critic. The focal policy is exact Frozen-0806 deck 007, `dragapult_ex_07bedfffbfad`, exact-deck SHA-256 `07bed...e725`. Actor and all 55 opponent policies start from friend-0806 epoch 11, SHA-256 `0ca395...5df8`. Opponents are immutable and deterministic; only focal trajectories enter PPO.

The critic source is 0036 `V2_latent_value_archetype_diff/epoch_0005`, checkpoint SHA-256 `e88b2f...6360`, exact-007 validation BCE `0.5267365`. It estimates `V(s)=2*sigmoid(value_logit)-1`. This is an initialization under historical Dragapult behavior, not an optimal Value oracle; PPO continues on-policy calibration.

## Input and model data flow

The actor-visible schema remains `0031_rule_faithful_semantic_decision_v2`, exactly 39 tensor/mask keys. Variable card, resource, event, option, skill and effect axes are batch-padded. No source/team identity or hidden opponent truth enters forward.

```text
official observation
  -> worker-local causal knowledge + stateless feature compiler
  -> unchanged 39-key canonical record
  -> central collate/H2D
  -> frozen friend-0806 Encoder (one shared pass)
       state tokens [B,S,320], state mask [B,S]
       option tokens [B,O,320], option mask [B,O]
       |
       +-> trainable original Action Decoder -> ordered legal action distribution
       +-> trainable 8-query, 2-block latent Value trunk
             -> query 0 scalar logit -> V(s) in [-1,1]
```

The auxiliary opponent-archetype and final-diff heads from supervised training remain frozen provenance tensors and are not evaluated or optimized during PPO. V5/V6 retain every original Encoder tensor as a frozen base and add rank-4, alpha-8 LoRA only to the merged Q and V projections of both self-attention and state cross-attention in the final Option TransformerDecoder block (index 1). K, attention output, FFN, the earlier Option block and all board/event layers remain frozen. All Q/V LoRA output factors start at zero, so checkpoint 0 is functionally identical to the source actor. The actor contains four board, one event and two option Transformer blocks. V6 additionally tunes only the affine weight/bias of the Option TransformerDecoder stack's final output LayerNorm.

The same adapted option features feed both Action Decoder and Value Head. Policy and Value losses therefore both backpropagate through focal Option LoRA and, in V6, the local output LayerNorm. The separately loaded Frozen-0806 opponent actor has no adapters and receives no gradient. V5/V6 trainable parameters are the full Action Decoder, 10,240 Q/V LoRA parameters, Value queries/blocks/final norm/value scalar head, and—for V6 only—640 output-LayerNorm parameters. Actor decoder, Value, LoRA and LayerNorm learning rates are respectively `1e-5`, `1e-4`, `3e-5` and `1e-5`.

The semantic policy caches the model-static card/attack/skill/effect prototype embeddings once after checkpoint/device/dtype placement. The cache is detached derived GPU state, is absent from `state_dict`, and is invalidated by `_apply` and `load_state_dict`. If prototype parameters ever become trainable, train mode bypasses the cache; current frozen-prototype PPO safely reuses it while decoder/value gradients remain live.

## Rollout and variance-reduction contract

The Frozen-0806 manifest contains 55 exact opponent decks and a committed 256-slot distribution, SHA-256 `16dbd1...c3cc3c`. Every PPO update consumes each slot once in a fresh deterministic permutation and samples 256 unique engine seeds. Each `(slot, engine_seed)` produces two Episodes with candidate/opponent fixed as physical players 0/1 and only requested first player swapped. Pair mates share engine and Search seed; each focal stochastic policy uses its own recorded generator seed, independent of worker batching order. Divergent actions can consume later random events differently, so this is paired environment randomness rather than action-indexed common random numbers.

Every schedule is written before rollout with deck, opponent slot, engine/Search/policy seed, seat, source-policy update and a full provenance SHA-256. A second environment-only commitment hashes opponent, seat, engine/Search/policy seeds and deliberately excludes `game_id` and `source_policy_update`. Fixed-evaluation guards compare the environment commitment, preventing provenance-only update changes from causing V2's false failure while still failing closed on any actual environment drift. Training seeds never enter the fixed 512-Episode greedy validation suite. Greedy validation runs at checkpoint 0 and every five updates. A separate final holdout seed namespace remains unopened until candidate checkpoints are locked. Pair-aware metrics report WW/split/LL and do not treat 512 mates as independent samples.

## Reward, GAE and PPO

Environment reward remains terminal only: win `+1`, draw `0`, loss `-1`. No PBRS or intermediate Value-delta reward is used. For consecutive focal action callbacks:

`delta_t = r_t + V(s_{t+1}) - V(s_t)`

GAE uses `gamma=1.0`, `lambda=0.95` and the official turn clock. Same-turn decisions use transition lambda `1.0`; when the recorded official turn changes, lambda is applied once. Episode-equal decision weighting keeps every completed Episode at total weight one.

PPO uses four epochs, decision minibatch 1024, actor LR `1e-5`, critic LR `1e-4`, clip `0.10`, Value coefficient `0.5`, entropy `0.01`, immutable source-decoder reference KL `0.02`, target behavior KL `0.02`, max gradient norm `0.5` and zero weight decay. Behavior log probabilities must replay within `1e-4` in a complete read-only preflight before optimization. The optimization forward then runs the adapted Encoder with autograd once per minibatch, allowing both policy and Value loss to update LoRA/LayerNorm; it does not reuse the old frozen-Encoder `no_grad` path.

`value_diag/*` describes the rollout source policy's pre-update Value, never the newly produced checkpoint. It reports Value mean/std, GAE and terminal targets, residual bias and explained variance against both targets. Absolute official-engine turn bins are 0–3, 4–7, 8–11 and 12+; distance-to-terminal bins are 0–1, 2–3 and 4+ turns. Outcome bins include focal win/loss/draw plus winning/losing opening and near-terminal slices. “Winning-side Value” means focal Value on Episodes ultimately won by the focal policy; the immutable opponent has no 0037 Value Head.

## Runtime, logging and artifacts

Seeded runtime 0002 has official-source SHA-256 `c3f00d...f9f2` and library SHA-256 `549a0c...53e6`; `engine/source/` remains untouched. V2 uses N16E8/I8: 16 spawned worker processes, up to 8 independent official battle pointers per process, and 8 bounded inference channels per role. One per-process library lock protects official ABI calls; while one battle waits for centralized CUDA inference, other sessions in that worker may compile or progress. Every battle side owns independent causal compiler state. The 0035 incremental compiler is not enabled; the admitted stateless worker compiler is used.

Strict fixed-greedy seeded-512 topology trials used identical 95,214 official selections. N16E8/I8 at 5 ms ran in `190.420` and `191.264 s`, median `2.683 games/s`, with median process-tree RSS `15.74 GiB`. N8E16/I8 ran in `195.831` and `195.696 s`, median `2.615 games/s`, with `11.08 GiB`; it is the low-memory fallback. V1's N32E1/per-game-spawn fixed baseline took `417.389 s`, so selected V2 wall is 54.3% lower. Benchmark artifacts are under `.tmp/evaluation/0037_rl_pool_benchmark/` and are local, reproducible, disposable evidence rather than tracked training data.

V1 `V1_value_initialized_turn_clock_seeded512` is retained as failed because it omitted the engine pool and prototype cache. V2 `V2_engine_pool_prototype_cache_seeded512` reached model-only checkpoint 10, then its fixed-eval guard falsely compared provenance fields and terminated before evaluation. Independent official-engine update-10 evaluation was 287–225 (56.05%), exactly equal in aggregate to checkpoint 0, so V2 provides no PPO strength gain evidence. Canonical JSONL writes before TensorBoard and W&B online mirror to private `dragon_bra/pokemon-tcg-policy-learning`; display names begin `0037`. Each update saves an atomic model-only decoder/critic/adapter checkpoint and retains all checkpoints. Optimizer, scheduler, scaler, RNG, rollout and replay are excluded.

`rollout/*` describes the stochastic source policy `checkpoint/update=k-1`; `checkpoint/update=k` is produced afterward. Fixed greedy `eval/*` is the checkpoint validation curve. Only a separate zero-error official-engine frozen evaluation can support a policy-strength conclusion.

## Current and next stage

V3 is retained as a zero-update launcher failure. V4 `V4_lora_r8_eval5_50u` is retained as a user-stopped cost feasibility failure: checkpoint-0 evaluation completed, but the first full board-LoRA PPO update drove GPU memory to 15.9/16.3 GiB and exceeded the acceptable wall budget. The replacement fair 50-update arms start from identical friend-0806 actor and 0036 V2 epoch-5 critic weights, use identical schedules and evaluate every five updates. V5 `V5_last_option_qv_lora_r4_eval5_50u` trains Decoder + Value + last-Option Q/V LoRA. V6 `V6_last_option_qv_lora_r4_outnorm_eval5_50u` adds only the Option stack output LayerNorm. Neither arm uses explicit PBRS. Selection or continuation to 200 updates requires comparable checkpoint-50 frozen evidence.

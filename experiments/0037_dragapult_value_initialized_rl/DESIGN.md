# 0037 Dragapult Value-Initialized RL

## Evidence boundary

General game rules follow `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`. Attacks end the acting turn; wins arise from Prize completion, no battle-capable opposing Pokémon, or start-of-turn deck-out. Exact observations, legal actions, terminal results and turn numbers come from unmodified official rules through the seeded CPU runtime or official-rule CUDA POD runtime; `engine/source/` is never modified. CUDA reports remain explicitly labelled CUDA evidence until official-CPU parity is closed. Value estimation, seed contracts, GAE, PPO, topology and evaluation splits are project choices rather than game rules.

## Objective and sources

0037 is the first PPO baseline using the dedicated 0036 critic. The focal policy is exact Frozen-0806 deck 007, `dragapult_ex_07bedfffbfad`, exact-deck SHA-256 `07bed...e725`. Actor and all 55 opponent policies start from friend-0806 epoch 11, SHA-256 `0ca395...5df8`. Opponents are immutable and deterministic; only focal trajectories enter PPO.

The critic source is 0036 `V2_latent_value_archetype_diff/epoch_0005`, checkpoint SHA-256 `e88b2f...6360`, exact-007 validation BCE `0.5267365`. It estimates `V(s)=2*sigmoid(value_logit)-1`. This is an initialization under historical Dragapult behavior, not an optimal Value oracle; PPO continues on-policy calibration.

## Input and model data flow

The actor-visible schema remains `0031_rule_faithful_semantic_decision_v2`, exactly 39 tensor/mask keys. Variable card, resource, event, option, skill and effect axes are batch-padded. No source/team identity or hidden opponent truth enters forward.

```text
official observation
  -> official CUDA engine state (resident lane)
  -> device Semantic0031 v2 compiler + masked-prefix compaction
  -> unchanged 39-key canonical record on CUDA
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

The CUDA Option compiler uses only the canonical per-decision `option_skill_*` and `option_effect_*` IDs, roles, one-based parents and masks. It never infers an active relation merely because a card prototype can possess that skill/effect. Its device scatter/mean-pool matches the actor's canonical `mean_pool_by_parent`; this boundary is required for behavior-log-prob replay.

## Rollout and variance-reduction contract

The Frozen-0806 manifest contains 55 exact opponent decks and a committed 256-slot distribution, SHA-256 `16dbd1...c3cc3c`. Every PPO update consumes each slot once in a fresh deterministic permutation and samples 256 unique engine seeds. Each `(slot, engine_seed)` produces two Episodes with candidate/opponent fixed as physical players 0/1 and only requested first player swapped. Pair mates share engine and Search seed; each focal stochastic policy uses its own recorded seed. CUDA sampling is statelessly keyed by `(policy seed, focal decision index, ordered-action step)`, so lane refill timing and batch row order cannot change a game's policy RNG stream. Divergent actions can consume later environment random events differently, so this is paired environment randomness rather than action-indexed common random numbers.

Before each rollout, its deck, opponent slot, engine/Search/policy seed, seat, source-policy update and full provenance SHA-256 are atomically appended to the version-level `artifact/schedules/train_schedules.json`. This single aggregate replaces per-update `train_update_*.json` fragments without discarding job-level reconstruction data. A second environment-only commitment excludes `game_id` and `source_policy_update`, so checkpoint provenance cannot cause V2's false schedule-drift failure.

Training remains 512 on-policy Episodes per update: 256 sampled slots/seeds, each run in both seats. Frozen Greedy validation is a separate `frozen_0806_seeded_2048_v2` contract: eight independent 256-slot replicas, eight distinct engine/Search seeds per fixed slot, four focal-first and four focal-second observations per slot, hence 2,048 games total with 1,024 in each seat. It runs at checkpoint 0 and every five updates and is checkpoint-independent. A separate final holdout namespace remains unopened until candidate checkpoints are locked. Historical seeded-512 results remain valid only under their old contract and must not be pooled with seeded-2048.

0037 also defines two project-level termination guards, neither of which is an official Pokémon rule. First, if the same physical actor makes the same fully identified selection for the twentieth time in one official turn, that actor immediately forfeits. CPU identity includes selection type/context, the full option fingerprint and returned action; CUDA identity uses selection type plus the complete canonical 19-field selected-option row. CUDA keeps independent counters for both physical actors, so alternating actors cannot clear each other's repeat history; counters reset only when that lane enters a new official turn or is refilled. Both paths use the twentieth occurrence (`>= 20`), rather than waiting for the official runtime's finite turn-record buffer to overflow. Second, evaluation declares a draw at raw engine turn 100 (50 complete rounds), while the 0037 RL collector matches its historical CPU contract by declaring a draw at raw engine turn 99. CUDA reads `OfficialStatePod.turn` directly through a zero-copy tensor view rather than a normalized Semantic0031 feature. Reports and rollout diagnostics preserve repeat forfeits and turn-limit draws separately so neither can be confused with native official-engine terminal reasons.

## Reward, GAE and PPO

Environment reward remains terminal only: win `+1`, draw `0`, loss `-1`. No PBRS or intermediate Value-delta reward is used. For consecutive focal action callbacks:

`delta_t = r_t + V(s_{t+1}) - V(s_t)`

GAE uses `gamma=1.0`, `lambda=0.95` and the official turn clock. Same-turn decisions use transition lambda `1.0`; when the recorded official turn changes, lambda is applied once. Episode-equal decision weighting keeps every completed Episode at total weight one.

PPO uses four epochs, decision minibatch 1024, actor LR `1e-5`, critic LR `1e-4`, clip `0.10`, Value coefficient `0.5`, entropy `0.01`, immutable source-decoder reference KL `0.02`, target behavior KL `0.02`, max gradient norm `0.5` and zero weight decay. Behavior log probabilities must replay within `1e-4` in a complete read-only preflight before optimization. The optimization forward then runs the adapted Encoder with autograd once per minibatch, allowing both policy and Value loss to update LoRA/LayerNorm; it does not reuse the old frozen-Encoder `no_grad` path.

`value_diag/*` describes the rollout source policy's pre-update Value, never the newly produced checkpoint. It reports Value mean/std, GAE and terminal targets, residual bias and explained variance against both targets. Absolute official-engine turn bins are 0–3, 4–7, 8–11 and 12+; distance-to-terminal bins are 0–1, 2–3 and 4+ turns. Outcome bins include focal win/loss/draw plus winning/losing opening and near-terminal slices. “Winning-side Value” means focal Value on Episodes ultimately won by the focal policy; the immutable opponent has no 0037 Value Head.

## Runtime, logging and artifacts

`engine/source/` remains untouched. The active collector uses the official-rule CUDA POD ABI v7, 256 resident lanes, device Semantic0031 compilation and masked lane refill. One focal model owns the shared state Encoder and first Option block; only the final Option block and Action Decoder branch between adapted focal and immutable opponent endpoints. The CPU seeded-runtime N16E8/I8 collector remains an explicit fallback and historical comparison, not the default.

Strict fixed-greedy seeded-512 topology trials used identical 95,214 official selections. N16E8/I8 at 5 ms ran in `190.420` and `191.264 s`, median `2.683 games/s`, with median process-tree RSS `15.74 GiB`. N8E16/I8 ran in `195.831` and `195.696 s`, median `2.615 games/s`, with `11.08 GiB`; it is the low-memory fallback. V1's N32E1/per-game-spawn fixed baseline took `417.389 s`, so selected V2 wall is 54.3% lower. Benchmark artifacts are under `.tmp/evaluation/0037_rl_pool_benchmark/` and are local, reproducible, disposable evidence rather than tracked training data.

The shared resident path was measured with exact FP32 Update-32 and immutable FP32 0806 on RTX 5080 / CUDA 12.8. It trims masked semantic prefixes, shares the state Encoder and first Option block, branches only at the adapted final Option block, and bounds ordered decode by routed `max_count`. After correcting the Option relation compiler, evaluation/training use strict FP32 (`highest`, TF32 disabled). Two complete 2,048-game strict-FP32 reruns produced identical outcomes, forfeit indices and raw terminal bytes at `21.893` and `21.888 games/s`; the current tracked 007 report ran at `21.753 games/s`, 2,048/2,048 finished and zero engine errors. Greedy-only endpoints skip unused `Categorical`, log-prob and entropy construction; PPO focal sampling still computes behavior log-prob and entropy.

Masked refill removes longest-Episode shard stalls and keeps model/engine allocation resident. The same loop's training mode records stochastic focal action, log-prob, entropy, adapted Value, official turn and source-policy update; immutable opponents remain Greedy and skip statistics. A real four-game end-to-end PPO smoke produced 321 focal decisions, replayed all behavior log-probabilities with maximum absolute error `2.86e-6` against the canonical autograd forward (limit `1e-4`), and completed one optimizer minibatch. TF32 was rejected because two 2,048-game runs differed in one ordinary outcome despite identical repeat-forfeit indices. Strict FP32 reproduced the complete outcome sequence, repeat-forfeit indices and raw terminal bytes exactly, so it is mandatory for current Frozen evaluation and CUDA rollout. CUDA currently consumes engine seed but not a separate Search RNG stream, which the user has explicitly excluded from the present reproducibility requirement.

The formal 512-Episode stochastic topology was also exercised end to end: 46,629 focal decisions and 95,313 total policy requests completed in `45.670 s` of resident hot loop plus `7.649 s` of CPU trajectory materialization, or `9.60 games/s` collector end to end (`11.21 games/s` hot loop). Staged on-policy records occupied `2.409 GB`; peak PyTorch allocation/reservation was `4.425/9.460 GB`, so the 16 GB device retained practical headroom. This is rollout-only cost; PPO optimization remains a separate update phase.

V1 `V1_value_initialized_turn_clock_seeded512` is retained as failed because it omitted the engine pool and prototype cache. V2 `V2_engine_pool_prototype_cache_seeded512` reached model-only checkpoint 10, then its fixed-eval guard falsely compared provenance fields and terminated before evaluation. Independent official-engine update-10 evaluation was 287–225 (56.05%), exactly equal in aggregate to checkpoint 0, so V2 provides no PPO strength gain evidence. Canonical JSONL writes before TensorBoard and W&B online mirror to private `dragon_bra/pokemon-tcg-policy-learning`; display names begin `0037`. Each update saves an atomic model-only decoder/critic/adapter checkpoint and retains all checkpoints. Optimizer, scheduler, scaler, RNG, rollout and replay are excluded.

`rollout/*` describes the stochastic source policy `checkpoint/update=k-1`; `checkpoint/update=k` is produced afterward. Fixed greedy `eval/*` is the checkpoint validation curve. Only a separate zero-error official-engine frozen evaluation can support a policy-strength conclusion.

## Current and next stage

V3 is retained as a zero-update launcher failure. V4 `V4_lora_r8_eval5_50u` is retained as a user-stopped cost feasibility failure: checkpoint-0 evaluation completed, but the first full board-LoRA PPO update drove GPU memory to 15.9/16.3 GiB and exceeded the acceptable wall budget. V5 `V5_last_option_qv_lora_r4_eval5_50u` trains Decoder + Value + last-Option Q/V LoRA and was user-stopped after update 32 because cross-update allocator/RSS growth made later updates too expensive. Under the historical official-engine seeded-512 suite, checkpoint 0 was 287–225 (56.05%), update 30 was 298–214 (58.20%), and update 32 was 305–207 (59.57%). Those results are not pooled with the new seeded-2048 contract. The next formal version uses CUDA-resident rollout and 2,048-game Frozen evaluation; it has not started.

Update 32 is the first 0037 Kaggle work package. The deployable actor is materialized under `archive/submission/0037_dragapult_ex_007_update32_last_option_qv_lora/`: the trained Action Decoder replaces the base decoder, and the rank-4/alpha-8 deltas are mathematically merged into the Q and V rows of the final Option block's self-attention and state cross-attention `in_proj_weight`. K rows, attention output, FFN, preceding Option block, board/event paths and prototypes remain the audited 0806 base. The training-only Value Head is intentionally excluded from inference. The corresponding root-content archive is `archive/submission/dist/0037_dragapult_ex_007_update32_last_option_qv_lora.tar.gz`, SHA-256 `67c1cac407a650229c2236d46795b22722c6e89860749fad6f3f9a80c4023c8a`. Fresh-extraction structure, exact-60 deck, no-symlink/cache, package validation and no-`__file__` dynamic execution all passed. The official seeded-engine smoke used the extracted package with isolated resident PyTorch inference and finished 10/10 games with 0 errors and 0 unfinished (5–5); this is a runtime gate, not policy-strength evidence.

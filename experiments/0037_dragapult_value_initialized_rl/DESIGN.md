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

Before each rollout, its deck, opponent slot, engine/Search/policy seed, seat, source-policy update and full provenance SHA-256 are atomically appended to the version-level `artifact/schedules/train_schedules.json`. This single aggregate replaces per-update `train_update_*.json` fragments without discarding any job-level reconstruction data. A second environment-only commitment hashes opponent, seat, engine/Search/policy seeds and deliberately excludes `game_id` and `source_policy_update`. Fixed-evaluation guards compare the environment commitment, preventing provenance-only update changes from causing V2's false failure while still failing closed on any actual environment drift. Training seeds never enter the fixed 512-Episode greedy validation suite. Greedy validation runs at checkpoint 0 and every five updates. A separate final holdout seed namespace remains unopened until candidate checkpoints are locked. Pair-aware metrics report WW/split/LL and do not treat 512 mates as independent samples.

0037 also defines a project-level progress guard, not an official Pokémon rule: if the same actor makes the same fully identified selection for the twentieth time in one official turn, that actor immediately forfeits. CPU identity includes selection type/context, the full option fingerprint and returned action; CUDA identity uses selection type plus the complete canonical 19-field selected-option row. The counter resets when actor or turn changes. Both paths use the twentieth occurrence (`>= 20`), rather than waiting for the official runtime's finite turn-record buffer to overflow. Reports preserve these outcomes separately as `ability_repeat_forfeit` / `repeated_selection_forfeit` or CUDA `progress_guard` metadata so they cannot be confused with native official-engine terminal reasons.

## Reward, GAE and PPO

Environment reward remains terminal only: win `+1`, draw `0`, loss `-1`. No PBRS or intermediate Value-delta reward is used. For consecutive focal action callbacks:

`delta_t = r_t + V(s_{t+1}) - V(s_t)`

GAE uses `gamma=1.0`, `lambda=0.95` and the official turn clock. Same-turn decisions use transition lambda `1.0`; when the recorded official turn changes, lambda is applied once. Episode-equal decision weighting keeps every completed Episode at total weight one.

PPO uses four epochs, decision minibatch 1024, actor LR `1e-5`, critic LR `1e-4`, clip `0.10`, Value coefficient `0.5`, entropy `0.01`, immutable source-decoder reference KL `0.02`, target behavior KL `0.02`, max gradient norm `0.5` and zero weight decay. Behavior log probabilities must replay within `1e-4` in a complete read-only preflight before optimization. The optimization forward then runs the adapted Encoder with autograd once per minibatch, allowing both policy and Value loss to update LoRA/LayerNorm; it does not reuse the old frozen-Encoder `no_grad` path.

`value_diag/*` describes the rollout source policy's pre-update Value, never the newly produced checkpoint. It reports Value mean/std, GAE and terminal targets, residual bias and explained variance against both targets. Absolute official-engine turn bins are 0–3, 4–7, 8–11 and 12+; distance-to-terminal bins are 0–1, 2–3 and 4+ turns. Outcome bins include focal win/loss/draw plus winning/losing opening and near-terminal slices. “Winning-side Value” means focal Value on Episodes ultimately won by the focal policy; the immutable opponent has no 0037 Value Head.

## Runtime, logging and artifacts

Seeded runtime 0002 has official-source SHA-256 `c3f00d...f9f2` and library SHA-256 `549a0c...53e6`; `engine/source/` remains untouched. V2 uses N16E8/I8: 16 spawned worker processes, up to 8 independent official battle pointers per process, and 8 bounded inference channels per role. One per-process library lock protects official ABI calls; while one battle waits for centralized CUDA inference, other sessions in that worker may compile or progress. Every battle side owns independent causal compiler state. The 0035 incremental compiler is not enabled; the admitted stateless worker compiler is used.

Strict fixed-greedy seeded-512 topology trials used identical 95,214 official selections. N16E8/I8 at 5 ms ran in `190.420` and `191.264 s`, median `2.683 games/s`, with median process-tree RSS `15.74 GiB`. N8E16/I8 ran in `195.831` and `195.696 s`, median `2.615 games/s`, with `11.08 GiB`; it is the low-memory fallback. V1's N32E1/per-game-spawn fixed baseline took `417.389 s`, so selected V2 wall is 54.3% lower. Benchmark artifacts are under `.tmp/evaluation/0037_rl_pool_benchmark/` and are local, reproducible, disposable evidence rather than tracked training data.

The experimental `engine_cuda` path was also exercised with the exact FP32 Update-32 checkpoint and immutable FP32 0806 opponent on RTX 5080 / CUDA 12.8. Following `cuda_ppo_quickstart_zh.md`, the optimized path trims masked semantic prefixes, shares the identical state Encoder and first Option block, branches only at the adapted final Option block, and bounds each greedy decode by the current routed `max_count`. This changed 64-lane throughput from `1.880` to `11.816 games/s`; decoder GPU time fell from `29.740` to `1.241 s`. The recommended 152-lane run completed 152/152 with zero error at `15.250 games/s`, 2.43 GiB peak reserved memory, and `9.967 s` wall; 256 lanes completed 256/256 at `14.751 games/s` with 4.37 GiB reserved, so 152 is the current static-batch choice. Relative to CPU N16E8's `2.683 games/s`, the 152-lane measurement is about 5.68× faster. An engine-plus-semantic-codec isolation loop sustained about 218k lane decisions/s. A first static 512-lane attempt exposed one `turn_record_overflow` at fixed schedule index 344: the Mega Venusaur ex opponent repeatedly selected Solar Transfer and its singleton follow-up choices, while two other lanes were merely left unfinished by fail-fast termination. This was not a batch-capacity failure. With the shared 20-occurrence progress guard, the same 512-job schedule run as four fixed 128-lane shards completed 512/512 with zero engine error and zero unfinished; four guarded forfeits occurred at schedule indices 69, 344, 416 and 417. A second complete 4×128 pass reproduced every game result, terminal-state hash and forfeit index exactly. A guarded single 512-lane batch also completed 512/512 twice with identical results and terminal hashes, using 9.79 GiB peak reserved memory. It disagreed with 4×128 on one game (schedule index 129), consistent with batch-shape-dependent TF32 greedy-logit ordering; therefore the CUDA schedule contract fixes shard size/order at 4×128 and forbids cross-topology strength comparisons. CUDA reset still consumes only engine seed rather than 0037's separate Search seed, static batches still wait for the longest Episode, and the current benchmark does not collect PPO Value/log-prob/targets; therefore these are throughput and stability results only and the official CPU engine remains the policy-strength oracle. A formal CUDA PPO collector still needs streaming lane replacement, exact seed-contract bridging and Value/behavior-policy storage.

V1 `V1_value_initialized_turn_clock_seeded512` is retained as failed because it omitted the engine pool and prototype cache. V2 `V2_engine_pool_prototype_cache_seeded512` reached model-only checkpoint 10, then its fixed-eval guard falsely compared provenance fields and terminated before evaluation. Independent official-engine update-10 evaluation was 287–225 (56.05%), exactly equal in aggregate to checkpoint 0, so V2 provides no PPO strength gain evidence. Canonical JSONL writes before TensorBoard and W&B online mirror to private `dragon_bra/pokemon-tcg-policy-learning`; display names begin `0037`. Each update saves an atomic model-only decoder/critic/adapter checkpoint and retains all checkpoints. Optimizer, scheduler, scaler, RNG, rollout and replay are excluded.

`rollout/*` describes the stochastic source policy `checkpoint/update=k-1`; `checkpoint/update=k` is produced afterward. Fixed greedy `eval/*` is the checkpoint validation curve. Only a separate zero-error official-engine frozen evaluation can support a policy-strength conclusion.

## Current and next stage

V3 is retained as a zero-update launcher failure. V4 `V4_lora_r8_eval5_50u` is retained as a user-stopped cost feasibility failure: checkpoint-0 evaluation completed, but the first full board-LoRA PPO update drove GPU memory to 15.9/16.3 GiB and exceeded the acceptable wall budget. V5 `V5_last_option_qv_lora_r4_eval5_50u` trains Decoder + Value + last-Option Q/V LoRA and was user-stopped after completing update 32 because cross-update allocator/RSS growth made later updates too expensive. On the identical official-engine seeded-512 suite, checkpoint 0 was 287–225 (56.05%), update 30 was 298–214 (58.20%), and update 32 was 305–207 (59.57%). V6 has not started.

Update 32 is the first 0037 Kaggle work package. The deployable actor is materialized under `archive/submission/0037_dragapult_ex_007_update32_last_option_qv_lora/`: the trained Action Decoder replaces the base decoder, and the rank-4/alpha-8 deltas are mathematically merged into the Q and V rows of the final Option block's self-attention and state cross-attention `in_proj_weight`. K rows, attention output, FFN, preceding Option block, board/event paths and prototypes remain the audited 0806 base. The training-only Value Head is intentionally excluded from inference. The corresponding root-content archive is `archive/submission/dist/0037_dragapult_ex_007_update32_last_option_qv_lora.tar.gz`, SHA-256 `67c1cac407a650229c2236d46795b22722c6e89860749fad6f3f9a80c4023c8a`. Fresh-extraction structure, exact-60 deck, no-symlink/cache, package validation and no-`__file__` dynamic execution all passed. The official seeded-engine smoke used the extracted package with isolated resident PyTorch inference and finished 10/10 games with 0 errors and 0 unfinished (5–5); this is a runtime gate, not policy-strength evidence.

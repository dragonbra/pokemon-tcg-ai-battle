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

The auxiliary opponent-archetype and final-diff heads from supervised training remain frozen provenance tensors and are not evaluated or optimized during PPO. Frozen Encoder hashes are checked before and after every update. V1 trainable parameters are only `actor.action_decoder.*`, Value queries/blocks/final norm and `heads.value.*`.

## Rollout and variance-reduction contract

The Frozen-0806 manifest contains 55 exact opponent decks and a committed 256-slot distribution, SHA-256 `16dbd1...c3cc3c`. Every PPO update consumes each slot once in a fresh deterministic permutation and samples 256 unique engine seeds. Each `(slot, engine_seed)` produces two Episodes with candidate/opponent fixed as physical players 0/1 and only requested first player swapped. Pair mates share engine and Search seed; each focal stochastic policy uses its own recorded generator seed, independent of worker batching order. Divergent actions can consume later random events differently, so this is paired environment randomness rather than action-indexed common random numbers.

Every schedule is written before rollout with deck, opponent slot, engine/Search/policy seed, seat, source-policy update and schedule SHA-256. Training seeds never enter the fixed 512-Episode greedy validation suite. A separate final holdout seed namespace remains unopened until candidate checkpoints are locked. Pair-aware metrics report WW/split/LL and do not treat 512 mates as independent samples.

## Reward, GAE and PPO

Environment reward remains terminal only: win `+1`, draw `0`, loss `-1`. No PBRS or intermediate Value-delta reward is used. For consecutive focal action callbacks:

`delta_t = r_t + V(s_{t+1}) - V(s_t)`

GAE uses `gamma=1.0`, `lambda=0.95` and the official turn clock. Same-turn decisions use transition lambda `1.0`; when the recorded official turn changes, lambda is applied once. Episode-equal decision weighting keeps every completed Episode at total weight one.

PPO uses four epochs, decision minibatch 1024, actor LR `1e-5`, critic LR `1e-4`, clip `0.10`, Value coefficient `0.5`, entropy `0.01`, immutable source-decoder reference KL `0.02`, target behavior KL `0.02`, max gradient norm `0.5` and zero weight decay. Behavior log probabilities must replay within `1e-4` before optimization.

## Runtime, logging and artifacts

Seeded runtime 0002 has official-source SHA-256 `c3f00d...f9f2` and library SHA-256 `549a0c...53e6`; `engine/source/` remains untouched. Engine-owning workers compile causal stateless records locally, while collation, CUDA inference and stochastic decoding remain centralized. The 0035 incremental compiler is not enabled.

Formal V1 is `V1_value_initialized_turn_clock_seeded512`. Canonical JSONL writes before TensorBoard and W&B online mirror to private `dragon_bra/pokemon-tcg-policy-learning`; display names begin `0037`. Each update saves an atomic model-only decoder/critic checkpoint and retains all checkpoints. Optimizer, scheduler, scaler, RNG, rollout and replay are excluded.

`rollout/*` describes the stochastic source policy `checkpoint/update=k-1`; `checkpoint/update=k` is produced afterward. Fixed greedy `eval/*` is the checkpoint validation curve. Only a separate zero-error official-engine frozen evaluation can support a policy-strength conclusion.

## Current and next stage

V1 is running after actor logit/action parity, exact 0036 critic tensor parity, worker-local feature parity, paired schedule audit, a finite official-engine PPO canary, unchanged Encoder hash and clean W&B startup. The online run is `0037-v1-value-initialized-turn-clock-seeded512`. Later versions may test partial Encoder unfreezing/LoRA or Value-guided action search; neither is part of V1.

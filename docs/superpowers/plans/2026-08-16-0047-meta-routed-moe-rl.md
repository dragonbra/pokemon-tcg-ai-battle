# 0047 Meta-Routed MoE RL Implementation Plan

**Goal:** Build a self-contained 0047 project from the verified 0045 PPO/CUDA stack, materialize both focal initialization and immutable opponent from Policy-0814, and launch a deck-070 long PPO run with seven independently trainable Actor experts and deployable public-information routing.

**Project identity:** `0047_meta_routed_moe_rl`

**Formal run identity:** `V8_deck070_policy0814_moe7_sparse_512lane_micro256` (V1–V7 remain auditable construction/capacity records; V8 uses true sparse gate-support execution, historical 0045 512-lane and PPO microbatch-256 execution settings, progress logging, and the requested seven-deck half schedule).

## Guardrails

- Do not edit `engine/source/` or mutate any 0045/0814 source artifact.
- Copy executable 0045 code into 0047; 0047 may use shared `rl_environment/`, `evaluation/`, official card data, and engine runtime, but may not import `train/0045_*`.
- Resolve and hash the one audited source at `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/` before construction.
- Materialize focal and opponent separately. Opponent uses the complete immutable Policy-0814 effective state and never shares mutable storage with focal.
- Router inputs are only the focal agent's accumulated public observations. The evaluator's true meta label is reporting/scheduling metadata only.

## Implementation

### 1. Self-contained project and assets

- Copy the current `train/0045_single_deck_expert_minimal_lora/` implementation to `train/0047_meta_routed_moe_rl/`, then rewrite project-local imports and identities.
- Copy the required 0814 actor/value files and immutable manifests into the 0047 asset tree, recording source paths and SHA-256 hashes.
- Create `experiments/0047_meta_routed_moe_rl/` and `rl_runs/0047_meta_routed_moe_rl/versions/V8_deck070_policy0814_moe7_sparse_512lane_micro256/` with fresh artifact, checkpoint, TensorBoard, and W&B directories.
- Register focal deck 070 and Policy-0814 in the project-local registries. Validate exact deck hash, model architecture, complete effective identity, and focal/opponent storage separation before CUDA work.

### 2. Actor and Router

- Retain one shared frozen semantic backbone and one shared Critic. Remove the shared Actor decoder from the active forward graph.
- Define seven independent Actor modules: E0 plus E00/E01/E02/E03/E05/E27. Each owns exactly the audited Actor trainables: final Option-block Q/V LoRA, ActionDecoder, and Allocation Head.
- Initialize every Actor from the same migrated 0814 Actor seed: exact 0814 decoder tensors, identical zero-delta rank-4/alpha-8 LoRA, and one deterministic allocation-head initialization copied seven ways. Report the allocation head as newly initialized because raw 0814 has no compatible tensor.
- Verify all corresponding expert tensors are numerically equal at U0 and have distinct `Parameter` objects/storage.
- Add six scalar Router logits for 00/01/02/03/05/27, hard-warmup phase, and deployable public-meta memory. UNKNOWN and non-core metas are fixed to E0. Warm-up routes each core meta to its own Specialist.
- At the U5 boundary enable six independent E0↔Em gates at alpha 0.80 and Router gradients. Use Router LR `0.2 * decoder_lr` as a separate optimizer group. No dense mixture, clustering, or load balancing is used.

### 3. Exact deployable meta identification

- Implement a new 0047 identifier; do not import the old folded 0045 mapping or the 15-class predictor.
- Maintain per-game public opponent-card evidence. Filter the immutable registered opponent-deck catalog by observed public card IDs; lock a meta only when every still-compatible exact deck has the same 29-class meta ID. Never unlock or relabel after confirmation.
- Store the routing state at every decision boundary (`UNKNOWN=-1` or confirmed ID), identification decision index, and public evidence version in the rollout transition. Never reconstruct earlier decisions from a later/final meta.

### 4. Exact policy mixture

- Decode every decision from the active gate support: E0 only for UNKNOWN/non-core states, or E0 plus the identified core meta's Specialist. At each token, form the effective conditional distribution from the posterior expert weights, sample/argmax that distribution, then Bayes-update expert posterior using the chosen token.
- For allocation actions, combine each expert's allocation distribution using its posterior after the root action sequence. Store the full effective joint log-probability.
- PPO recomputation uses the saved decision-boundary routing ID and computes `logsumexp(log(g_k)+log pi_k(action|state))`, including allocation. Ratio, behavior KL, specialist-reference KL, sampling, and greedy evaluation all consume this effective policy.
- Freeze a same-architecture U0 reference snapshot. Critic and opponent remain outside the Router and expert mixture.

### 5. Checkpoint, evaluation, and logging

- Save model-only checkpoints containing shared base, Critic, all seven Actors, six Router logits/probabilities, phase, and identifier config/version. Atomically write a lightweight `router_table.json` beside each checkpoint.
- Add fixed seeded CUDA-512 pools against Frozen Policy-0814: `old_three=02/03/05`, `new_three=00/01/27`, and `remain_meta=all remaining IDs`. True meta stays outside focal inference.
- Log pool win rates, each meta win rate and game count, effective expert usage, six `router/alpha_MM` values, unknown-decision fraction, identification timing, and a 29×7 sparse Router heatmap. Keep `training_metrics.jsonl` canonical and mirror to TensorBoard then W&B online.

## Minimal gates before launch

1. U0 fixed-observation parity against raw Policy-0814 for root action probabilities/actions. Allocation is separately audited because it is absent from raw 0814.
2. Equal initial expert values and independent objects/storage.
3. Hard-routing gradient ownership for 00/01/02/03/05/27/other/UNKNOWN.
4. Soft-routing PPO smoke: Router updates, all effective probabilities normalize, stored/recomputed old log-probabilities agree before an update, and no NaN/Inf appears after an update.
5. Tiny official-engine CUDA evaluation across all three pool definitions, with Policy-0814 opponent identity and W&B/heatmap logging verified.

## Launch

- Start `V8_deck070_policy0814_moe7_sparse_512lane_micro256` with focal deck 070, 512-game rollout, existing 0045 PPO settings, 5 hard-routing updates, true sparse Actor execution, 512 CUDA lanes, PPO forward microbatch 256, and the 256 core-deck + 256 Meta-balanced opponent split.
- Run fixed three-pool CUDA-512 evaluation at the inherited periodic cadence, retain every model-only update checkpoint, and continue indefinitely unless a hard failure (NaN, routing/effective-policy error, no expert updates, or CUDA/eval crash) occurs.

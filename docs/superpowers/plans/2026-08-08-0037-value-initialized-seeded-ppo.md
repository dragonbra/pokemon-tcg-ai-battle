# 0037 Value-Initialized Seeded PPO Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with review gates. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch the first exact-007 PPO baseline initialized from the selected 0036 Value Network, using turn-clock GAE and fresh 256-pair/512-Episode seeded Frozen-0806 rollouts.

**Architecture:** Freeze the admitted friend-0806 semantic Encoder, fine-tune only the original action decoder and the 0036 latent-query Value trunk/scalar head, and retain terminal `-1/0/+1` rewards. Each update consumes every committed Frozen-0806 scenario slot once with a fresh engine seed and its opposite-seat mate; worker-local stateless compilation feeds centralized CUDA inference. Fixed validation and final holdout seed namespaces never enter PPO updates.

**Tech Stack:** Python 3.11, PyTorch PPO, 0031 semantic Transformer, 0036 latent-query critic, seeded official CPU engine runtime 0002, multiprocessing worker-local compilation, CUDA batched inference, TensorBoard, W&B online, unittest.

## Global Constraints

- Never modify `engine/source/`; use the hash-committed seeded runtime under `engine/build/seeded_official/0002/`.
- Project roots are `train/0037_dragapult_value_initialized_rl/`, `experiments/0037_dragapult_value_initialized_rl/`, and `rl_runs/0037_dragapult_value_initialized_rl/`.
- 0037 is self-contained and may not import executable code from another numbered project; copied source and weight provenance must be hash-recorded.
- Focal deck is exact Frozen-0806 deck 007, `dragapult_ex_07bedfffbfad`, SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`.
- Actor and opponent source checkpoint is friend-0806 epoch 11, SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`.
- Critic initialization is 0036 V2 epoch 5; load the latent-query trunk and scalar head with strict tensor/hash parity. Auxiliary archetype and final-diff outputs do not enter PPO loss or optimizer.
- Freeze every Encoder tensor. Train only `actor.action_decoder.*` plus the latent-query Value trunk/scalar head.
- Reward remains terminal official outcome only: win `+1`, draw `0`, loss `-1`. Do not add PBRS or Value-delta reward shaping.
- Use `gamma=1.0`, turn-clock GAE, `gae_lambda=0.95`, four PPO epochs, decision minibatch 1024, actor LR `1e-5`, critic LR `1e-4`, clip ratio `0.10`, value coefficient `0.5`, entropy coefficient `0.01`, reference KL coefficient `0.02`, target behavior KL `0.02`, max gradient norm `0.5`, and zero weight decay.
- Each PPO update uses 256 unique base scenario/engine-seed pairs and exactly two opposite-seat Episodes per pair, totaling 512 completed Episodes. The same pair shares physical deck slots, engine seed and Search seed; focal stochastic policy RNG is distinct per Episode and independent of batching order.
- Training, repeated validation and final holdout seeds use disjoint derivation domains. Validation outcomes may select checkpoints; final holdout outcomes are opened only after candidate checkpoints are locked.
- Canonical `training_metrics.jsonl` writes before TensorBoard and W&B. Formal W&B target is private `dragon_bra/pokemon-tcg-policy-learning`, with run display name beginning `0037`.
- Save model-only checkpoints after every update with `checkpoint_retention: all`; never save optimizer, scheduler, scaler, RNG, rollout, replay, or DataLoader state.

---

### Task 1: Freeze source assets and project identity

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/`
- Create: `experiments/0037_dragapult_value_initialized_rl/manifest.json`
- Create: `experiments/0037_dragapult_value_initialized_rl/DECISIONS.md`
- Create: `experiments/0037_dragapult_value_initialized_rl/DESIGN.md`
- Create: `experiments/0037_dragapult_value_initialized_rl/DESIGN.html`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_project_identity.py`

**Interfaces:**
- Produces strict source/deck/critic/runtime identities and rejects any hash or metadata mismatch before creating a run directory.

- [ ] Copy only required semantic policy, rollout, PPO, logging, catalog and test assets into the self-contained 0037 package; remove caches and obsolete historical runners.
- [ ] Snapshot the exact source actor checkpoint and selected 0036 V2 epoch-5 model-only checkpoint under Git-ignored `rl_runs/0037_dragapult_value_initialized_rl/source/`, with tracked hash manifests only.
- [ ] Test exact project ID, 60-card deck, source actor hash, critic checkpoint hash/schema/config, Frozen-0806 schedule hash and seeded-runtime manifest.
- [ ] Run the identity test and require PASS before model construction.

### Task 2: Integrate the latent-query critic without changing the actor

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/policy/actor_critic.py`
- Create: `train/0037_dragapult_value_initialized_rl/policy/action_distribution.py`
- Create: `train/0037_dragapult_value_initialized_rl/policy/value_network.py`
- Create: `train/0037_dragapult_value_initialized_rl/training/checkpoints.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_value_initialization.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_actor_critic.py`

**Interfaces:**
- Produces one shared Encoder pass returning action-decoder inputs and `2*sigmoid(value_logit)-1`.
- Produces model-only checkpoints containing decoder and critic state plus immutable identities.

- [ ] Test bit-identical update-0 actor logits, greedy actions and sampled log probabilities versus the source actor.
- [ ] Test critic logit/value parity against independently loaded 0036 V2 epoch 5 on real and synthetic batches.
- [ ] Exclude auxiliary output heads from the deployed critic and optimizer while preserving the trained query/block/value tensors.
- [ ] Assert trainable names contain only action decoder and critic; assert the frozen representation hash before and after backward/optimizer steps.
- [ ] Reject optimizer/RNG/replay fields in saved checkpoints and retain every numbered update.

### Task 3: Implement paired fresh-seed schedules and variance accounting

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/rollout/protocol.py`
- Create: `train/0037_dragapult_value_initialized_rl/rollout/schedule.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_schedule.py`

**Interfaces:**
- Produces `build_training_jobs(source_policy_update, run_seed) -> list[RolloutJob]` with 512 jobs.
- Produces immutable pre-rollout schedule manifests and pair-aware outcome metrics.

- [ ] Test 256 unique base pairs, 512 Episodes, all 55 Frozen-0806 opponent identities, exact scenario allocation, fixed physical deck slots and opposite requested seats.
- [ ] Test shared engine/Search seed within each pair, distinct per-Episode policy seeds, deterministic reconstruction, disjoint update windows and disjoint train/validation/holdout namespaces.
- [ ] Hash and atomically persist every update schedule before rollout, including deck, opponent slot, engine seed, Search seed, policy seed, seat and source-policy checkpoint hash.
- [ ] Report paired WW/WL/LW/LL/draw outcomes and bootstrap/statistical grouping by 256 pairs; never treat 512 mates as independent evidence.

### Task 4: Bring the admitted high-throughput rollout topology into 0037

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/rollout/worker.py`
- Create: `train/0037_dragapult_value_initialized_rl/rollout/collector.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_rollout.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_feature_parity.py`

**Interfaces:**
- Engine-owning workers compile causal stateless canonical records locally; the central process performs collation, CUDA transfer, actor/critic inference and stochastic sampling.
- Produces complete focal-only action-level trajectories with features, ordered action, stop flag, behavior log probability, entropy, Value, engine turn and all seed identities.

- [ ] Copy the admitted worker-local stateless compiler and frozen-prototype GPU cache implementation into 0037 with source commitments; do not enable the rejected incremental compiler path.
- [ ] Test all 39 canonical tensors, actor logits, Value logits and actions against the independent central reference on complete official trajectories.
- [ ] Keep one causal encoder per battle side and fail closed on actor/deck/session/chronology mismatch, illegal action, worker error or incomplete Episode.
- [ ] Benchmark a paired official-engine smoke and record games/s, selections/s, mean inference batch, compile/collate/H2D/model timing, CPU RSS and GPU memory.

### Task 5: Turn-clock GAE and PPO

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/training/batch.py`
- Create: `train/0037_dragapult_value_initialized_rl/training/ppo.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_gae.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_ppo.py`

**Interfaces:**
- Applies lambda once when consecutive focal decisions cross an official engine-turn boundary and not between decisions inside the same turn.
- Produces clipped PPO updates over decoder and critic only.

- [ ] Test same-turn transitions use trace factor `1.0`, cross-turn transitions use `0.95`, terminal bootstrap is zero, gamma remains `1.0`, and long action chains preserve credit.
- [ ] Keep one parameter-update pass per configured PPO epoch without re-running greedy decode; replay behavior log probabilities before optimization with maximum error `1e-4`.
- [ ] Log original terminal return, old/new Value, explained variance, Value loss, advantage standard deviation, KL, clip fraction, entropy, gradient norm and frozen-representation parity.
- [ ] Run a one-update canary and require 512/512 valid Episodes, finite metrics, unchanged Encoder hash, changed decoder/critic hashes and a model-only update-1 checkpoint.

### Task 6: Formal V1 orchestration, evaluation separation and launch

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/training/run.py`
- Create: `train/0037_dragapult_value_initialized_rl/monitor_training.py`
- Create during launch: `rl_runs/0037_dragapult_value_initialized_rl/versions/V1_value_initialized_turn_clock_seeded512/`

**Interfaces:**
- Produces canonical JSONL/TensorBoard/W&B metrics, per-update schedule manifests and all model-only checkpoints.
- Uses sampled `rollout/*` only as training diagnosis and fixed greedy `eval/*` as checkpoint validation.

- [ ] Create the W&B run at process startup with stable ID and display name beginning `0037`, and persist its URL/status locally.
- [ ] Run a fixed 512-Episode greedy validation suite at update 0 and every ten updates; keep its seed namespace disjoint from training.
- [ ] Record `rollout/source_policy_update=k-1`, then save/log `checkpoint/update=k`; never relabel sampled rollout win rate as checkpoint-k strength.
- [ ] Lock candidate updates using training plus validation evidence before opening the separate final seeded holdout suite.
- [ ] Launch V1 only after all gates pass, then inspect the first complete rollout/update/evaluation cycle for throughput, W&B synchronization and Value/GAE health.

## Deferred 0037 research branches

- Unfreeze additional Encoder blocks or add LoRA to the first one or two Transformer blocks only in a new strictly incremented version with a new trainable/hash contract.
- Use the Value Network for action-conditioned search, Q-style reranking or model-based lookahead only in a separate version; V1 does not use Value deltas as reward or alter greedy action selection.

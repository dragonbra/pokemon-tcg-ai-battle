# 0026 Raging Bolt Canonical Decoder RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune only the 0025 best-greedy canonical action decoder against all 51 immutable 0019 Foundation zero-shot deck opponents using official-engine PPO.

**Architecture:** A heterogeneous parent collector batches canonical observations for the focal Raging Bolt actor and legacy 0019 observations for the frozen opponents. The full 0025 semantic representation is loaded, but only `action_decoder.*` and a new value head are trainable; terminal win/loss/draw rewards drive PPO with behavior-policy clipping and KL anchoring to the immutable best-greedy decoder.

**Tech Stack:** Python 3.11+, PyTorch, official engine runtime, multiprocessing spawn workers, TensorBoard, W&B online, repository `rl_environment` logging/storage utilities.

## Global Constraints

- Project ID is `0026_raging_bolt_canonical_decoder_rl`; runtime code under `train/0026_raging_bolt_canonical_decoder_rl/` must not import another numbered training project.
- `engine/source/` is read-only; all capability evidence comes from the unmodified official engine runtime.
- Focal deck is exact-deck James Cox / James Cox & Henry Chao Raging Bolt, SHA-256 `f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a`.
- Initialization is 0025 V4 best-greedy-exact epoch 14, checkpoint SHA-256 `adc4eaeca1e62a28bbd762e8e513212c94941044bbc85673f02bbeadc1865aa3`.
- Trainable actor parameters are exactly `action_decoder.*`; `prototypes.*`, `state_encoder.*`, and `option_encoder.*` remain bit-identical to initialization. The new value head is trainable.
- Opponents are the 51 exact decks from the immutable 0019 Foundation Frozen view; opponent weights never update and opponent identity is not actor-visible.
- PPO checkpoint payloads are model-only and exclude optimizer, scheduler, scaler, RNG, DataLoader, rollout, replay, and engine state.
- Formal runs use W&B online project `dragon_bra/pokemon-tcg-policy-learning` and a foreground watchdog with structured heartbeats/alerts.

---

### Task 1: Self-Contained Project And Provenance

**Files:**
- Create: `train/0026_raging_bolt_canonical_decoder_rl/__init__.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/focal/`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/opponents/foundation/`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/league/decks/`
- Create: `experiments/0026_raging_bolt_canonical_decoder_rl/manifest.json`
- Create: `experiments/0026_raging_bolt_canonical_decoder_rl/decisions/2026-08-02-initial-contract.md`

**Interfaces:**
- Consumes: immutable 0025 checkpoint and archived 0019 Epoch 13 model data.
- Produces: project-local Python source, `verify_initialization()`, and `load_frozen_opponent()` without numbered-project imports.

- [x] Copy and namespace the 0025 canonical feature/model/knowledge code into `focal/`; copy and namespace the frozen 0019 model/codec code into `opponents/foundation/`.
- [x] Copy the 51 exact deck manifests into `league/decks/` and write a loader that rejects any count other than 51, any non-60 deck, or a changed catalog digest.
- [x] Write tests that scan AST imports and fail on `train.00xx` cross-project dependencies.
- [x] Strict-load both immutable checkpoints and assert their expected hashes, schema, parameter counts, and deck identity.
- [x] Run the project-contract tests and record the frozen asset hashes in the experiment manifest.

### Task 2: Decoder-Only Actor-Critic Contract

**Files:**
- Create: `train/0026_raging_bolt_canonical_decoder_rl/policy/actor_critic.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/policy/action_distribution.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/checkpoint.py`
- Test: `train/0026_raging_bolt_canonical_decoder_rl/tests/test_policy_contract.py`

**Interfaces:**
- Consumes: `CanonicalSemanticPolicy` and canonical batches.
- Produces: `CanonicalActorCritic`, `sample_actions()`, `greedy_actions()`, `evaluate_actions()`, `save_model_checkpoint()`, and `load_model_checkpoint()`.

- [x] Write a failing test that expects trainable names to equal `actor.action_decoder.*` plus `value_head.*` and verifies all other tensors are frozen.
- [x] Implement `freeze_representation()` by first disabling every actor parameter, then enabling only `actor.action_decoder.parameters()` and the value head.
- [x] Implement categorical sampling/evaluation over unique ordered legal options plus STOP using `OrderedOptionDecoder.logits()` and `consume()`.
- [x] Add a state-summary value head ending in `tanh`, initialized with a zero final layer.
- [x] Add model-only atomic checkpoints containing the action decoder, value head, initialization hash, representation hash, update, and schema; reject forbidden training-state keys.
- [x] Verify one optimizer step changes decoder/value tensors and leaves the representation hash unchanged.

### Task 3: Heterogeneous Official-Engine Rollout

**Files:**
- Create: `train/0026_raging_bolt_canonical_decoder_rl/rollout/protocol.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/rollout/worker.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/rollout/collector.py`
- Test: `train/0026_raging_bolt_canonical_decoder_rl/tests/test_rollout.py`

**Interfaces:**
- Consumes: focal canonical actor, frozen 0019 actor, exact focal/opponent decks, official runtime path.
- Produces: `EpisodeTrajectory` containing only focal-policy decisions and terminal actor-relative rewards.

- [x] Define immutable jobs with opponent ID, balanced seat, source policy update, seed, exact decks, runtime root, and 1,000-step cap.
- [x] Run each official-engine game in an isolated spawn worker; worker sends observations to the GPU parent and never mutates engine source.
- [x] In the parent, split ready requests by role: canonical encoder/collator and sampled focal action for focal requests; legacy encoder/collator and greedy frozen action for opponent requests.
- [x] Store only focal decisions with behavior log-prob, entropy, value, action indices, STOP flag, and source policy update.
- [x] Fail closed on encoder exceptions, illegal selection bounds, worker EOF, timeout, step limit, or incomplete terminal reward.
- [x] Run a four-game two-opponent/two-seat official-engine smoke and assert all games finish without errors.

### Task 4: Decoder-Only PPO

**Files:**
- Create: `train/0026_raging_bolt_canonical_decoder_rl/training/batch.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/training/ppo.py`
- Test: `train/0026_raging_bolt_canonical_decoder_rl/tests/test_ppo.py`

**Interfaces:**
- Consumes: focal-only trajectories from Task 3.
- Produces: `PreparedBatch`, terminal-reward GAE returns, and `PPOTrainer.update()` metrics.

- [x] Prepare actor-relative terminal returns with `gamma=1.0`, `gae_lambda=0.95`, normalized advantages, and equal episode weighting.
- [x] Freeze an update-local behavior snapshot and the immutable best-greedy reference decoder.
- [x] Optimize clipped PPO policy loss, value MSE, entropy bonus, and reference-log-prob KL surrogate over decoder/value parameters only.
- [x] Use actor LR `1e-5`, value LR `1e-4`, clip `0.10`, entropy coefficient `0.01`, reference KL coefficient `0.02`, target behavior KL `0.02`, and grad norm `0.5`.
- [x] Assert finite losses/gradients, nonzero decoder update, unchanged representation hash, and exact rollout/source checkpoint chronology.

### Task 5: Versioned Training And Monitoring

**Files:**
- Create: `train/0026_raging_bolt_canonical_decoder_rl/training/run.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/monitor_training.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/cli.py`
- Create: `train/0026_raging_bolt_canonical_decoder_rl/__main__.py`
- Test: `train/0026_raging_bolt_canonical_decoder_rl/tests/test_training.py`

**Interfaces:**
- Consumes: all prior task interfaces.
- Produces: immutable `rl_runs/0026.../versions/V<n>_<tag>/` artifacts, checkpoints, TensorBoard, W&B logs, and watchdog evidence.

- [x] Schedule 512 sampled games per update across all 51 Frozen identities with balanced seats; omit no opponent.
- [x] Before update 0 and every fifth update, run a 102-game balanced-seat greedy Frozen diagnostic and record `eval/checkpoint_update` separately from sampled rollout metrics.
- [x] Save bounded model-only latest/best checkpoints and enforce free-space low-water marks.
- [x] Log canonical `training_metrics.jsonl` first, then TensorBoard, then W&B with `rollout/source_policy_update`, `checkpoint/update`, throughput, KL, gradients, resource use, and frozen evaluation.
- [x] Implement foreground watchdog heartbeats checking process, status, metrics, checkpoints, W&B sync, GPU/CPU/RAM/swap/SSD, worker EOF, timeout, and OOM signals.
- [x] Test version collision refusal and model-only payloads. Monitoring remains exercised by the formal foreground run.

### Task 6: Authority Documentation

**Files:**
- Create: `experiments/0026_raging_bolt_canonical_decoder_rl/DESIGN.md`
- Create: `experiments/0026_raging_bolt_canonical_decoder_rl/DESIGN.html`

**Interfaces:**
- Consumes: verified code contracts and smoke evidence.
- Produces: synchronized human-readable authority documents.

- [x] Document exact input tensors, heterogeneous focal/opponent data flow, frozen/trainable modules, PPO objective, reward boundary, checkpoint schema, run version, and current/next project stage.
- [x] Distinguish official rules, official runtime/card facts, and project hypotheses; state that RL improvement tests usability of the representation but cannot by itself prove semantic completeness.
- [x] Parse HTML, validate manifest JSON, and cross-check every parameter/hash/count against code and checkpoint metadata.

### Task 7: Official-Engine Smoke Gate

**Files:**
- Create: `rl_runs/0026_raging_bolt_canonical_decoder_rl/versions/V1_best_exact_decoder_only_smoke/artifact/`
- Create: `.tmp/evaluation/0026_decoder_rl_smoke/<run_id>/report.html`

**Interfaces:**
- Consumes: complete 0026 package.
- Produces: an auditable readiness decision before a formal long run.

- [x] Run all 0026 unit/contract tests and `git diff --check`.
- [x] Run an official-engine sampled rollout plus one PPO update and prove representation SHA is unchanged while decoder SHA changes.
- [x] Export the smoke checkpoint as a self-contained candidate, validate exact 60-card/cg/package contracts, and run a small official-engine evaluation report under `.tmp/evaluation/0026_decoder_rl_smoke/`.
- [x] Record smoke throughput, GPU peak memory, decisions, rewards, KL, gradient norm, checkpoint hashes, and all errors in V1 status/summary.
- [ ] Start formal V2 only if V1 has zero engine/encoder/decode errors and all frozen/trainable invariants hold.

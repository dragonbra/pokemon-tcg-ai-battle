# 0033 Full Semantic CPU PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task in the current session. Subagent execution is unavailable for this repository turn.

**Goal:** Train the exact PT0805 Epoch 13 0031 SemanticPolicy decoder against all 51 Frozen 0019 decks using only official CPU engine observations and outcomes, without dropping or substituting any actor-visible field.

**Architecture:** Each official-engine game runs in an isolated CPU worker. The parent owns one complete PT0805 SemanticPolicy on CUDA, one actor-local `OnlineCausalEncoder` per live game, and the immutable 0019 opponent policy; it batches inference while preserving per-game chronological state. PPO freezes every original 0031 tensor except `action_decoder.*`, adds a separate critic over the original state summary, and fails before any optimizer step unless update-0 features, logits, probabilities, and greedy actions match the evaluated PT0805 package.

**Tech Stack:** Python 3.11, PyTorch, official CPU `cg` runtime, multiprocessing spawn workers, 0031 SemanticPolicy, terminal-reward PPO, TensorBoard, W&B online.

## Global Constraints

- Never modify `engine/source/`; official CPU runtime is the only rollout and outcome authority.
- `train/0033_dragapult_third_ptcg_club_rl/` must not import executable code from another numbered project.
- The focal actor schema is exactly `0031_rule_faithful_semantic_decision_v2`; missing, extra, default-degraded, or substituted fields fail closed.
- Preserve full global/card/resource/event/option tensors, registered deck, causal ledger, known-hand state, event chronology, and card/attack/skill/effect prototypes.
- Use all 51 Frozen 0019 exact decks twice in both seat orders every training update: 204 complete Episodes.
- Update only the original `action_decoder.*` plus a non-deployed value head; hash all other original tensors before and after every update.
- Use a fresh V3 repository version, fresh optimizer, W&B online, model-only checkpoint per update, and retention `all`.
- Use terminal-only reward with `gamma=1.0` and `GAE lambda=1.0`; every action in a complete Episode receives the same undiscounted terminal-return target and per-decision weight `1/T`.
- Sampled rollout metrics are diagnostics; checkpoint strength requires later fixed-seed official-engine greedy evaluation.
- Do not commit automatically in the shared dirty worktree.

---

### Task 1: Close And Record Invalid V2

**Files:**
- Modify: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V2_third_ptcg_club_cuda_ppo_200u/artifact/status.json`
- Create: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V2_third_ptcg_club_cuda_ppo_200u/artifact/training_summary.json`
- Modify: `experiments/0033_dragapult_third_ptcg_club_rl/DECISIONS.md`

**Interfaces:**
- Consumes: six existing model-only checkpoints and canonical metric rows.
- Produces: immutable `stopped_by_user_invalid_feature_contract` evidence.

- [ ] Verify no V2 training/watchdog process remains and stop the W&B run.
- [ ] Record update 6, 233,472 decisions, 942 episodes, and the missing full-semantic input contract.
- [ ] Mark all V2 checkpoints invalid for selection or continuation.

### Task 2: Freeze Exact 0031 Runtime And 51-Deck League

**Files:**
- Create: `train/0033_dragapult_third_ptcg_club_rl/semantic_policy/`
- Create: `train/0033_dragapult_third_ptcg_club_rl/opponents/`
- Create: `train/0033_dragapult_third_ptcg_club_rl/league/frozen_catalog.json`
- Create: `train/0033_dragapult_third_ptcg_club_rl/league/decks/<deck_id>/`

**Interfaces:**
- Consumes: evaluated PT0805 package implementation and immutable 0019 model/deck assets.
- Produces: `PortableSemanticPolicy`, `OnlineCausalEncoder`, `DecisionBatch`, `load_foundation()`, and `load_frozen_catalog()` entirely inside 0033.

- [ ] Mechanically freeze the evaluated package's assets, contracts, deployment, domain, features, knowledge, and model modules.
- [ ] Freeze the 51 exact Frozen deck manifests and 0019 opponent implementation.
- [ ] Add a test that rejects any `train.00xx` executable import and verifies all 51 multiset hashes.

### Task 3: Exact PT0805 Actor-Critic Contract

**Files:**
- Create: `train/0033_dragapult_third_ptcg_club_rl/policy/actor_critic.py`
- Create: `train/0033_dragapult_third_ptcg_club_rl/policy/action_distribution.py`
- Create: `train/0033_dragapult_third_ptcg_club_rl/policy/batching.py`
- Test: `train/0033_dragapult_third_ptcg_club_rl/tests/test_full_semantic_policy.py`

**Interfaces:**
- Produces: `SemanticActorCritic`, `sample_actions()`, `greedy_actions()`, and `evaluate_actions_encoded()`.

- [ ] Write a failing test requiring strict checkpoint load of all 293 tensors and exact 56,352,322 actor parameters.
- [ ] Load `SemanticPolicy` using checkpoint metadata and both official prototype assets with `strict=True`.
- [ ] Freeze all actor parameters, then enable only `actor.action_decoder.*`; initialize a separate value head from the original state summary.
- [ ] Hash the complete frozen representation and reject any mutation.

### Task 4: Full Online Feature And Package-Equivalence Gate

**Files:**
- Create: `train/0033_dragapult_third_ptcg_club_rl/parity.py`
- Test: `train/0033_dragapult_third_ptcg_club_rl/tests/test_pt0805_runtime_parity.py`

**Interfaces:**
- Produces: `assert_pt0805_runtime_parity(observations, deck, checkpoint)` and a JSON parity report.

- [ ] Assert all 39 exact `DecisionBatch` keys and field widths: global 12/24, card 9/7, resource 4/15, event 31/4, option 19/2 plus all relation/prototype tensors.
- [ ] Feed the same ordered official observation sequence into the evaluated package encoder and V3 encoder.
- [ ] Compare every tensor element, actor state digest, logits, ordered-action log probabilities, and greedy action.
- [ ] Reject all mismatch, fallback, reinitialized encoder, or missing event/ledger evidence before optimizer construction.

### Task 5: Official CPU Rollout With Per-Game State Isolation

**Files:**
- Create: `train/0033_dragapult_third_ptcg_club_rl/rollout/protocol.py`
- Create: `train/0033_dragapult_third_ptcg_club_rl/rollout/worker.py`
- Create: `train/0033_dragapult_third_ptcg_club_rl/rollout/collector.py`
- Test: `train/0033_dragapult_third_ptcg_club_rl/tests/test_official_cpu_rollout.py`

**Interfaces:**
- Produces: `RolloutJob`, `EpisodeTrajectory`, and `FullSemanticRolloutCollector.collect()`.

- [ ] Spawn one isolated official CPU engine process per game; never mock outcomes or legal options.
- [ ] Create one focal and one opponent encoder per live game and keep each encoder until terminal result.
- [ ] Batch parent-process inference without sharing causal state between games.
- [ ] Store complete CPU feature batches and exact behavior log probability for focal decisions only.
- [ ] Fail the whole update on worker error, schema mismatch, illegal action, timeout, or incomplete episode.

### Task 6: Decoder-Only PPO And Model-Only Storage

**Files:**
- Create: `train/0033_dragapult_third_ptcg_club_rl/training/batch.py`
- Create: `train/0033_dragapult_third_ptcg_club_rl/training/ppo_full_semantic.py`
- Create: `train/0033_dragapult_third_ptcg_club_rl/training/storage_full_semantic.py`
- Test: `train/0033_dragapult_third_ptcg_club_rl/tests/test_full_semantic_ppo.py`

**Interfaces:**
- Produces: episode-balanced GAE, `PPOTrainer.update()`, and atomic model-only checkpoints.

- [ ] Replay behavior log probabilities before the first gradient and require MAE below `1e-4`.
- [ ] Optimize PPO clipped policy loss, value loss, entropy, and fixed PT0805 decoder reference KL.
- [ ] Hash and verify every non-decoder PT0805 tensor after each optimizer step.
- [ ] Save only `action_decoder.*`, `value_head.*`, update, source hashes, full schema ID, and reconstruction metadata.

### Task 7: V3 Formal Runner, Logging, And Watchdog

**Files:**
- Create: `train/0033_dragapult_third_ptcg_club_rl/training/run_full_semantic.py`
- Modify: `train/0033_dragapult_third_ptcg_club_rl/monitor_training.py`
- Modify: `experiments/0033_dragapult_third_ptcg_club_rl/DESIGN.md`
- Modify: `experiments/0033_dragapult_third_ptcg_club_rl/DESIGN.html`

**Interfaces:**
- Produces: `V3_full_semantic_cpu_ppo_200u` with canonical JSONL, TensorBoard, W&B, status, snapshots, and all checkpoints.

- [ ] Build exactly 204 jobs per update: every Frozen deck twice in each seat.
- [ ] Record `rollout/source_policy_update=k-1`, `checkpoint/update=k`, all opponent hashes, engine runtime hash, feature parity hash, and W&B stable ID/URL.
- [ ] Refuse an occupied version and retain every update checkpoint.
- [ ] Update both DESIGN documents to make V2 invalid and V3 the only valid RL path.

### Task 8: Gates And Formal Launch

**Files:**
- Create: `.tmp/evaluation/0033_full_semantic_cpu_smoke/`
- Create: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V3_full_semantic_cpu_ppo_200u/`

**Interfaces:**
- Consumes: all prior gates.
- Produces: a continuously monitored formal V3 run.

- [ ] Run all 0033 and shared evaluation/versioning/W&B tests.
- [ ] Run update-0 package-equivalence parity on real official observation sequences.
- [ ] Benchmark 8/12/16 official CPU workers on one fixed 24-game contract and select the fastest zero-error setting.
- [ ] Run a zero-gradient 204-game all-51/two-per-seat official CPU rollout and require 204/204 terminal, zero errors.
- [ ] Run one decoder-only PPO canary and require frozen hash unchanged, decoder/value hash changed, and finite metrics.
- [ ] Launch V3 under a detached watchdog with W&B online.
- [ ] Wait for update 1; verify metrics, checkpoint plus SHA-256, status heartbeat, full opponent snapshot, zero errors, and active W&B URL.

## Self-Review

- Spec coverage: original policy, complete fields, chronological state, official CPU engine, all 51 decks, decoder-only update, parity, logging, retention, and persistent launch each have an explicit gate.
- Placeholder scan: no deferred implementation or unspecified error-handling step remains.
- Type consistency: the collector emits `EpisodeTrajectory`; batch preparation consumes it; PPO consumes the prepared batch; storage reconstructs the exact actor plus decoder/value delta.

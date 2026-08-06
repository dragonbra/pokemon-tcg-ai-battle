# 0033 Dragapult Third PTCG Club RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the PT0805 Epoch 13 foundation zero-shot on the exact Third PTCG Club Dragapult deck, then start a versioned decoder/value-only PPO run against the frozen 0019 opponent snapshot.

**Architecture:** V1 packages the immutable PT0805 Epoch 13 `best_validation_loss` policy with the supplied exact 60-card deck and evaluates it against all 51 CPU Frozen opponents through the official engine. V2 is a self-contained copy of the accepted 0032 resident-CUDA pathway: the 0031-derived learner is initialized from the audited PT0805 checkpoint transfer, all encoder tensors remain frozen, only the action decoder and value head train, and 38 CUDA-supported Frozen decks use the shared immutable 0019 Epoch 13 opponent head.

**Tech Stack:** Python 3.11, PyTorch, official CPU engine evaluation, official resident CUDA engine, PPO, TensorBoard, W&B online.

## Global Constraints

- Do not modify `engine/source/`; official engine runtime results are the only policy-strength evidence.
- Project ID is `0033_dragapult_third_ptcg_club_rl`; numbered-project executable code must be self-contained.
- V1 and V2 artifacts are immutable and use strictly increasing version directories.
- Formal PPO writes canonical JSONL, TensorBoard, W&B online metadata, and one atomic model-only decoder/value checkpoint per update with retention `all`.
- Source identity is provenance only and must not enter actor-visible inputs.
- Sampled rollout rates are diagnostics, not frozen greedy checkpoint-strength claims.

---

### Task 1: Freeze the exact deck and zero-shot candidate

**Files:**
- Create: `evaluation/arena/candidates/0033_dragapult_third_ptcg_club_zero_shot/`
- Create: `train/0033_dragapult_third_ptcg_club_rl/league/decks/dragapult_third_ptcg_club/deck.csv`
- Create: `train/0033_dragapult_third_ptcg_club_rl/league/decks/dragapult_third_ptcg_club/manifest.json`

**Interfaces:**
- Consumes: PT0805 checkpoint `.tmp/model_loading/0031_friend_0805/checkpoint/best_validation_loss.pt` with SHA-256 `285b88f5e30c40ad07ad025b429c5bd1594f0359cc0d44849c368056222d63ae`.
- Produces: a standard candidate whose initialization observation returns the exact supplied 60-card multiset.

- [ ] Materialize the supplied card IDs and counts as exactly 60 newline-delimited IDs.
- [ ] Copy the already exported PT0805 portable runtime and replace only deck/provenance metadata.
- [ ] Assert card counts, deck SHA-256, no symlinks, initialization return, and `evaluation validate` success.

### Task 2: Run formal zero-shot evaluation

**Files:**
- Create: `experiments/0033_dragapult_third_ptcg_club_rl/evaluation/V1_pt0805_epoch13_zero_shot_frozen0019.html`
- Create: `experiments/0033_dragapult_third_ptcg_club_rl/evaluation/index.html`
- Create: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V1_pt0805_epoch13_zero_shot_frozen0019/artifact/evaluation.json`
- Create: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V1_pt0805_epoch13_zero_shot_frozen0019/artifact/status.json`

**Interfaces:**
- Consumes: Task 1 candidate and the immutable 51-deck Frozen 0019 pool.
- Produces: 510 official-engine games, ten balanced games per opponent, with completion/error and matchup metrics.

- [ ] Run `python3 -m evaluation run` with `--workers 8 --worker-cpu-threads 1`, shared CUDA inference, and `league_deck_quality` metrics.
- [ ] Fail closed unless all 510 games finish without errors.
- [ ] Record the authoritative report link, run ID, exact candidate/deck/checkpoint hashes, and refresh the evaluation index.

### Task 3: Implement the self-contained 0033 training unit

**Files:**
- Create: `train/0033_dragapult_third_ptcg_club_rl/`
- Create: `experiments/0033_dragapult_third_ptcg_club_rl/manifest.json`
- Create: `experiments/0033_dragapult_third_ptcg_club_rl/DECISIONS.md`
- Create: `experiments/0033_dragapult_third_ptcg_club_rl/DESIGN.md`
- Create: `experiments/0033_dragapult_third_ptcg_club_rl/DESIGN.html`

**Interfaces:**
- Consumes: the accepted 0032 POD-native actor, transfer map, resident rollout, PPO, checkpoint, and watchdog contracts as copied source.
- Produces: imports only from 0033, `rl_environment`, `evaluation`, `engine_cuda`, official assets, and immutable checkpoints.

- [ ] Copy the focused 0032 implementation and tests into 0033, then rewrite project/schema identifiers and focal deck constants.
- [ ] Change source initialization to the audited PT0805 Epoch 13 checkpoint hash and keep the 0019 opponent foundation hash distinct.
- [ ] Add tests proving exact deck identity, frozen encoder gradients/weights, decoder/value-only checkpoint payloads, source hash refusal, and 38-opponent snapshot identity.
- [ ] Update both DESIGN formats with tensor shapes, transfer boundary, trainable parameter boundary, PPO/reward/value contracts, stages, and evidence limitations.

### Task 4: Verify official-engine PPO readiness

**Files:**
- Create: `.tmp/evaluation/0033_dragapult_third_ptcg_club_canary/ppo_one_update.json`

**Interfaces:**
- Consumes: Task 3 implementation, official CUDA rules binary, built extension, 38-deck support audit, deck snapshot, PT0805 Epoch 13 checkpoint, and 0019 foundation.
- Produces: one update with legal resident actions, finite PPO metrics, frozen encoder proof, and decoder/value change proof.

- [ ] Run all 0033 unit tests and repository contract tests affected by copied code.
- [ ] Run a 152-lane official-engine PPO canary and require zero engine/legality errors.
- [ ] Verify source checkpoint and frozen opponent hashes, GPU memory headroom, W&B authentication, disk headroom, and fresh V2 output paths.

### Task 5: Start and supervise formal V2 PPO

**Files:**
- Create: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V2_third_ptcg_club_cuda_ppo_200u/{artifact,checkpoint,tensorboard,wandb}/`
- Create: `.tmp/training_monitor/0033_dragapult_third_ptcg_club_rl/V2_third_ptcg_club_cuda_ppo_200u/`

**Interfaces:**
- Consumes: all Task 4 gates.
- Produces: a live, watchdog-owned W&B online PPO process using 152 lanes, 256 rollout steps, 200 updates, and all-checkpoint retention.

- [ ] Launch through the 0033 monitor with a stable W&B run ID and no reuse of prior version paths.
- [ ] Wait for the first completed update, atomic checkpoint plus SHA-256, canonical JSONL row, TensorBoard event, W&B URL/sync status, and heartbeat.
- [ ] Confirm nonzero throughput, completed Episodes, finite losses/KL, zero engine/illegal/OOM errors, unchanged frozen encoder commitment, and sufficient disk.
- [ ] Leave the watchdog and training process running and report exact PID, version, status, W&B URL, and local artifact paths.

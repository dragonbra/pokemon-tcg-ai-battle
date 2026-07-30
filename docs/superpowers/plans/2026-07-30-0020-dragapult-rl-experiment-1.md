# 0020 Dragapult RL Experiment 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start the first deck-specific 0020 RL experiment from the frozen universal epoch-13 checkpoint using the Dragapult exact deck and complete official-engine terminal-reward rollouts.

**Architecture:** Physically freeze the proven 0018 terminal-RL implementation inside project 0020, adapted to the 0019 epoch-13 source schema, neutral persona, and Dragapult deck. Calibrate a new value head in immutable V7, preserve the Raging Bolt zero-shot evaluation in immutable V8, then warm-start decoder-only PPO in immutable V9 with a fresh optimizer and freshly collected on-policy Episodes.

**Tech Stack:** Python 3.11, PyTorch 2.11 CUDA, official engine worker processes, TensorBoard, W&B online.

## Global Constraints

- Never modify `engine/source/`; `engine_cuda/` is research-only and rollout-inadmissible.
- Use checkpoint SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.
- Use exact deck SHA-256 `00468e64b7c5eefb1cc4586ba9351a7f2a5bce223ef209d2b9e4b0bb0ca42e17`.
- Use neutral `source_id=0`; do not import executable code from another numbered project.
- Reward is only terminal win `+1`, loss `-1`, draw `0`; invalid Episodes are discarded.
- Training metrics flush to JSONL, TensorBoard, then private W&B `dragon_bra/pokemon-tcg-policy-learning`.
- Checkpoints are model-only and must exclude optimizer, scheduler, scaler, RNG, replay, and rollout state.
- Repository versions are V7 value calibration, V8 Raging Bolt zero-shot evaluation, and V9 initial PPO; “RL experiment 1” is the research label, not a reused V1 directory.

---

### Task 1: Freeze the 0020 terminal-RL runtime

**Files:**
- Create: `train/0020_pluggable_deck_rl/rl/`
- Create: `train/0020_pluggable_deck_rl/tests/test_rl_policy.py`
- Create: `train/0020_pluggable_deck_rl/tests/test_rl_training.py`
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.md`
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.html`

**Interfaces:**
- Consumes: 0020 foundation checkpoint, ontology, exact Dragapult deck, current Arena catalog.
- Produces: `load_source_actor_critic() -> (DragapultActorCritic, metadata)` and `python3 -m train.0020_pluggable_deck_rl.rl.run`.

- [ ] Write failing tests for project identity, hashes, 60-card deck, 17,756,162 actor parameters, neutral source ID, zero value head, decoder-only trainability, model-only checkpoints, terminal reward, and no cross-project imports.
- [ ] Copy the 0018 terminal-RL runtime into the 0020 project and rename every project/deck/schema identity.
- [ ] Adapt checkpoint loading to accept only `0019_model_only_checkpoint_v1`, epoch 13, global step 337194, strict weights, vocabulary size 509, and actor parameter count 17,756,162.
- [ ] Replace Alakazam diagnostics with Dragapult/Dusknoir diagnostics without changing rewards.
- [ ] Run `python3 -m unittest -v train.0020_pluggable_deck_rl.tests.test_rl_policy train.0020_pluggable_deck_rl.tests.test_rl_training`.

### Task 2: Prove official-engine rollout and W&B contracts

**Files:**
- Create: `.tmp/evaluation/0020_dragapult_rl_smoke/`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V7_dragapult_value_calibration/`

**Interfaces:**
- Consumes: self-contained runtime and frozen 30-opponent snapshot.
- Produces: successful short official-engine sampled rollout with legal terminal Episodes and online W&B initialization.

- [ ] Run a 30-opponent greedy probe only if the training worker contract cannot be validated from the existing V2 formal report.
- [ ] Run a disposable two-Episode training smoke outside formal version paths with W&B disabled.
- [ ] Verify actor weights remain byte-identical during value calibration and every Episode reward comes from official terminal state.
- [ ] Verify W&B credentials and the training interpreter can import W&B.

### Task 3: Run V7 value calibration

**Files:**
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V7_dragapult_value_calibration/artifact/training_config.json`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V7_dragapult_value_calibration/artifact/training_metrics.jsonl`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V7_dragapult_value_calibration/artifact/training_summary.json`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V7_dragapult_value_calibration/checkpoint/update-000000-value.pt`

**Interfaces:**
- Consumes: 512 sampled official-engine Episodes, balanced over the frozen pool.
- Produces: calibrated model-only actor-critic checkpoint suitable for a fresh V9 PPO optimizer.

- [ ] Allocate the unused V7 artifact, checkpoint, TensorBoard, and W&B staging paths.
- [ ] Run value calibration with `--episodes 512 --epochs 4 --batch-size 1024 --learning-rate 1e-4 --workers 16 --device cuda --allow-gpu --coalesce-ms 2 --wandb-mode online`.
- [ ] Verify 512 valid Episodes, no persisted rollout, actor unchanged, finite value metrics, model-only checkpoint, W&B URL/sync state, and completed status.

### Task 4: Start V9 decoder-only PPO

**Files:**
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V9_dragapult_ppo_initial/`
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.md`
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.html`

**Interfaces:**
- Consumes: V7 model-only checkpoint with a fresh optimizer and new on-policy Episodes.
- Produces: initial decoder-only PPO checkpoints and live canonical/TensorBoard/W&B curves.

- [ ] Allocate V9 with 100 updates, 512 Episodes/update, GAE lambda 1, 2 epochs, batch 1024, actor LR 1e-5, value LR 1e-4, entropy 0, reference-KL coefficient 0.02, behavior-KL guard 0.01, checkpoint every 10 updates plus update 1/final, retention 8.
- [ ] Start V9 and inspect update 1 for legal completion, finite losses/gradients, KL, clip fraction, decoder movement, canary flips, throughput, storage, and W&B sync.
- [ ] Continue the same frozen V9 run when update-1 guards pass; otherwise preserve failed/interrupted status and allocate a new version for changed semantics.
- [ ] Synchronize DESIGN.md and DESIGN.html with actual V7/V8/V9 status and measured facts.

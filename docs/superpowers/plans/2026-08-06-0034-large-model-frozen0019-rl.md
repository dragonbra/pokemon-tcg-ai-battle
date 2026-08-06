# 0034 Large Model Frozen 0019 RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained 0034 experiment that initializes the selected 0033 V6 PPO design from `large-model-0806.tar.gz`, establishes its Frozen 0019 zero-shot baseline, and runs 200 PPO updates with auditable win-rate monitoring.

**Architecture:** Physically freeze the full 0033 semantic-policy, rollout, PPO, checkpoint, parity, and watchdog implementation into a new numbered project with no executable cross-project imports. Export the new 0031-compatible checkpoint into a new exact THIRD candidate, validate/parity-test it, record a 510-game official-engine V1 baseline, then launch a fresh-optimizer V2 long run over the immutable 51-opponent Frozen 0019 snapshot. Sampled rollout and periodic 102-game frozen greedy results remain separate metric namespaces and evidence classes.

**Tech Stack:** Python 3.11, PyTorch CUDA inference/PPO, official CPU engine workers, repository evaluation CLI, TensorBoard, W&B online.

## Global Constraints

- Project ID is `0034_dragapult_third_large_model_rl`; executable files under `train/0034_dragapult_third_large_model_rl/` must not import another numbered training project.
- The source checkpoint is `best_validation_loss_0806.pt` from `large-model-0806.tar.gz`, SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`.
- The exact THIRD deck remains 60 cards with file SHA-256 `5db1e0d52fc723e8b2f688d76f780715fd726171f4d804f1d4d55673ba9b0ac2`; source/team identity is provenance only.
- All strength evidence must use the unmodified official engine runtime and immutable Frozen 0019 pool of 51 opponents with balanced focal seats.
- V2 starts fresh from the new pretrained checkpoint with a new optimizer; it may not continue any 0033 checkpoint or W&B run.
- PPO uses terminal-only `+1/0/-1`, `gamma=1.0`, turn-clock same-turn lambda `1.0`, boundary lambda `0.97`, `episode_equal_decisions`, 204 Episodes/update, two PPO epochs, batch 512, actor LR `1e-5`, critic LR `1e-4`, clip `0.1`, entropy `0.01`, reference KL `0.02`, and target behavior KL `0.02`.
- The 200-update run uses 24 isolated CPU workers, 5 ms inference coalescing, a frozen greedy 102-game probe every 10 updates, all model-only checkpoints retained, and W&B online project `dragon_bra/pokemon-tcg-policy-learning`.
- No optimizer, scheduler, RNG, replay, or rollout-buffer state is saved in checkpoints; no checkpoint retention or deletion is allowed.
- `experiments/0034_dragapult_third_large_model_rl/DESIGN.md` and `DESIGN.html` must match the implemented schema, architecture, objectives, and current stage before launch.

---

### Task 1: Freeze the 0034 executable project and identity contract

**Files:**
- Create: `train/0034_dragapult_third_large_model_rl/`
- Create: `train/0034_dragapult_third_large_model_rl/tests/test_project_identity.py`
- Create: `experiments/0034_dragapult_third_large_model_rl/manifest.json`
- Create: `experiments/0034_dragapult_third_large_model_rl/DECISIONS.md`

**Interfaces:**
- Consumes: the physical 0033 source tree, the new checkpoint, the exact THIRD deck, and shared `rl_environment/`, `evaluation/`, official-card, and engine infrastructure.
- Produces: a self-contained `train.0034_dragapult_third_large_model_rl` package whose constants, paths, schemas, hashes, run IDs, and tests name only 0034 and the new source asset.

- [ ] **Step 1: Copy the executable source without caches or generated weights**

Run a mechanical copy of `train/0033_dragapult_third_ptcg_club_rl/` to `train/0034_dragapult_third_large_model_rl/`, excluding `__pycache__`, `*.pyc`, and the old `semantic_policy/model.bin`; physically retain the exact deck and immutable runtime assets.

- [ ] **Step 2: Update the identity contract and expected source hash**

Set `PROJECT = "0034_dragapult_third_large_model_rl"`, source checkpoint to `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/model.pt`, candidate to `evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot`, source SHA to `0ca395...5df8`, and long-run default version to `V2_turn_clock_lambda097_200u`. Rename PT0805-specific parity/schema labels to neutral 0034 large-model labels while preserving tensor and action contracts.

- [ ] **Step 3: Add identity and cross-project-import assertions**

Make `test_project_identity.py` assert the strict project ID, source/candidate paths, exact deck hash, W&B `0034-...` run ID, and absence of executable `train.0033_` imports in every 0034 Python file.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m unittest -v train.0034_dragapult_third_large_model_rl.tests.test_project_identity`

Expected: all identity, source-hash, deck, and self-containment assertions pass.

### Task 2: Export and validate the new zero-shot policy

**Files:**
- Create: `evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot/`
- Create: `.tmp/evaluation/0034_large_model_parity/parity.json`
- Modify: `train/0034_dragapult_third_large_model_rl/semantic_policy/model.bin`

**Interfaces:**
- Consumes: the new model-only checkpoint and 0034 semantic runtime.
- Produces: a self-contained FP16-storage/FP32-runtime candidate and training-side portable checkpoint with strict actor parity.

- [ ] **Step 1: Strict-load the source checkpoint**

Assert schema `0031_model_only_checkpoint_v1`, 293 source tensors, 56,352,322 actor parameters, exact key/shape compatibility, and source SHA before creating any runtime package.

- [ ] **Step 2: Export the candidate and local portable checkpoint**

Use the established `_portable_checkpoint()` contract to remove registered prototype aliases exactly once, use FP16 storage with FP32 runtime, copy the exact THIRD deck and physical `cg/`, and write manifest hashes and new checkpoint provenance.

- [ ] **Step 3: Validate the candidate package**

Run: `python3 -m evaluation validate evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot`

Expected: exact 60-card initialization, imports, engine compatibility, and package validation all pass with no symlink or external runtime dependency.

- [ ] **Step 4: Run exact parity**

Collect official-engine observations and assert all 39 actor input keys, exact compiled feature tensors, bitwise actor tensor identity, maximum logit error no greater than `2e-6`, and exact greedy actions. Persist source, portable, and representation hashes in `.tmp/evaluation/0034_large_model_parity/parity.json`.

### Task 3: Verify PPO and monitoring gates

**Files:**
- Create: `.tmp/evaluation/0034_large_model_ppo_gate/full_204_ppo.json`
- Test: `train/0034_dragapult_third_large_model_rl/tests/`

**Interfaces:**
- Consumes: the validated source model, exact deck, Frozen 0019 catalog, official engine, and V6 PPO configuration.
- Produces: evidence that one complete on-policy update is finite, legal, and changes only the decoder/value modules.

- [ ] **Step 1: Run the complete 0034 test suite**

Run: `python3 -m unittest discover -v train/0034_dragapult_third_large_model_rl/tests`

Expected: all model, rollout, credit, storage, and identity tests pass.

- [ ] **Step 2: Run a full 204-Episode PPO gate**

Run the 0034 full-semantic gate with 204 games, 24 workers, CUDA, 5 ms coalescing, turn clock, lambda `0.97`, and `episode_equal_decisions`.

Expected: 204 valid Episodes, all 51 opponents and both seats present, zero engine/illegal/incomplete errors, finite PPO metrics, changed decoder/value hash, unchanged representation hash, and behavior log-prob parity within the existing contract.

- [ ] **Step 3: Verify model-only storage and watchdog behavior**

Assert the gate checkpoint payload excludes optimizer/scheduler/RNG/rollout state, the monitor detects fatal log patterns and stale heartbeats, and version preflight rejects any occupied artifact/checkpoint/TensorBoard/W&B path.

### Task 4: Establish the formal V1 zero-shot baseline

**Files:**
- Create: `experiments/0034_dragapult_third_large_model_rl/evaluation/V1_large_model_zero_shot_frozen0019.html`
- Create: `experiments/0034_dragapult_third_large_model_rl/evaluation/index.html`
- Create: `rl_runs/0034_dragapult_third_large_model_rl/versions/V1_large_model_zero_shot_frozen0019/artifact/evaluation.json`

**Interfaces:**
- Consumes: the validated 0034 zero-shot candidate and fixed opponent catalog.
- Produces: the immutable pre-RL 510-game comparison point and authoritative report linkage.

- [ ] **Step 1: Run the formal evaluation**

Run: `python3 -m evaluation run --candidate evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot --opponents all --games 10 --workers 8 --worker-cpu-threads 1 --output experiments/0034_dragapult_third_large_model_rl/evaluation/V1_large_model_zero_shot_frozen0019.html`

Expected: 510/510 official-engine games finish, each of 51 opponents contributes 10 games, seats are balanced, and errors equal zero.

- [ ] **Step 2: Refresh the project evaluation index and reverse link**

Generate the standard clickable index row with candidate, run ID, 510 games, W-L-D, error count, win rate, and completion rate; write `artifact/evaluation.json` pointing back to the authoritative V1 HTML without duplicating traces.

- [ ] **Step 3: Record the evidence boundary**

Update the 0034 manifest and decisions with the exact W-L-D and state explicitly that V1 is the pre-RL baseline, not a guarantee of the 200-update run's final strength.

### Task 5: Synchronize design documents and launch V2

**Files:**
- Create: `experiments/0034_dragapult_third_large_model_rl/DESIGN.md`
- Create: `experiments/0034_dragapult_third_large_model_rl/DESIGN.html`
- Create: `rl_runs/0034_dragapult_third_large_model_rl/versions/V2_turn_clock_lambda097_200u/{artifact,checkpoint,tensorboard,wandb}/`

**Interfaces:**
- Consumes: all passed gates and the V1 baseline.
- Produces: an active, supervised 200-update PPO run with canonical JSONL, TensorBoard, W&B, heartbeats, periodic frozen probes, and retained model-only checkpoints.

- [ ] **Step 1: Cross-check the authoritative design pages**

Document the 39-field semantic input contract and tensor shapes, 56,352,322-parameter pretrained actor, frozen representation, trainable original decoder plus critic, terminal reward, turn-clock GAE, PPO loss, Frozen 0019 schedule, V1/V2 boundary, checkpoint contract, and sampled-rollout versus frozen-greedy evidence boundary in both Markdown and HTML.

- [ ] **Step 2: Run launch preflight**

Verify V2 paths are all unused, CUDA is available, W&B imports in the training interpreter, the private project configuration is online, at least 5 GiB GPU and 5 GiB disk remain, no old 0033/0034 trainer is active, and the new source/candidate hashes still match.

- [ ] **Step 3: Start the watchdog-supervised 200-update run**

Launch `train.0034_dragapult_third_large_model_rl.monitor_training` in a persistent tmux session with version `V2_turn_clock_lambda097_200u`, 200 updates, 204 games/update, 24 workers, CUDA, 5 ms coalescing, eval every 10, turn clock, lambda `0.97`, `episode_equal_decisions`, and W&B online.

- [ ] **Step 4: Verify the first completed update**

Wait for update 1, then verify `training_metrics.jsonl` contains `rollout/source_policy_update=0` and `checkpoint/update=1`, checkpoint `update-000001.pt` exists with a recorded SHA, TensorBoard has an event file, W&B has a stable run URL/sync state, heartbeat is current, representation hash is unchanged, and there are no engine or mirror errors.

- [ ] **Step 5: Hand off monitoring endpoints**

Report the tmux session/process ID, W&B URL, TensorBoard command/path, metrics/status/checkpoint locations, V1 baseline, first-update sampled rollout metrics, estimated completion window, and the rule that 102-game periodic probes are diagnostics while any final strength claim requires a formal 510-game official-engine evaluation.

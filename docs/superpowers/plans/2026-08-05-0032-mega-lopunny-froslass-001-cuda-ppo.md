# 0032 Mega Lopunny ex / Mega Froslass ex 001 CUDA PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one `mega_lopunny_ex_mega_froslass_ex_001` actor from the 0031 epoch-2 checkpoint with fully GPU-resident official-engine rollouts, maximizing end-to-end PPO throughput while preserving auditable frozen-opponent and policy-strength boundaries.

**Architecture:** One POD-native encoder serves every lane. Player 0 uses the trainable 0032 actor head; player 1 uses a frozen deck-indexed head initialized from the matching 0022 V11 decoder checkpoint. Rollouts stay on CUDA, retain complete Episode trajectories for terminal-return GAE, and copy one compact immutable batch to the learner only after collection; PPO trains the focal decoder/value path and separately measures rollout, update, and combined throughput.

**Tech Stack:** Python 3.11, PyTorch 2.11/CUDA 12.8, official `engine_cuda` extension on SM 120, TensorBoard, W&B, unittest.

## Global Constraints

- Never modify `engine/source/`.
- Keep `train/0032_dragapult_pod_native_cuda_rl/` self-contained; it may copy and record provenance from 0022/0023 but may not import their executable code.
- Preserve `V1_pod_native_adapter`; formal optimization uses a new monotonically increasing version.
- Source checkpoint is 0031 epoch 2 SHA-256 `05f98e2a562f0037713ebd4b96a5a324151c91b187d7edaa879cc1c024134a51`.
- Focal exact deck is `evaluation/arena/frozen/mega_lopunny_ex_mega_froslass_ex_001/deck.csv`.
- Opponent deck snapshot is the 38-deck admitted CUDA subset from `evaluation/arena/combat_mat/0031_latest_frozen_test/cuda_support.json`.
- Frozen heads come from 0022 V11 catalog SHA-256 `d16488e6bb191ddffd626bbdfdaf6b978b1d1d5184922a61c64ec179e2bab97b`; copied decoder weights are an adapter initialization and never a claim of old-encoder behavior parity.
- Source/team identity is provenance only and never actor-visible.
- Use a fresh optimizer, model-only atomic checkpoints, and retain every update.
- Write canonical metrics to JSONL, then TensorBoard, then W&B; sampled rollout and frozen greedy evaluation use separate namespaces.
- Formal long training runs under a foreground watchdog emitting heartbeat/alert/complete events at no more than 60-second intervals.

---

### Task 1: Freeze The Focal Selection And Opponent Snapshot

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/selection_audit.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_selection_audit.py`
- Create during execution: `rl_runs/0032_dragapult_pod_native_cuda_rl/versions/V2_mega_lopunny_froslass_001_preflight/artifact/selection_audit.json`
- Modify: `experiments/0032_dragapult_pod_native_cuda_rl/DECISIONS.md`

**Interfaces:**
- Consumes: 0031 epoch-2 `validation_breakdown.jsonl`, winner catalog Episode records, frozen deck manifests, and CUDA support JSON.
- Produces: `build_selection_audit(...) -> dict[str, object]` with exact deck identity, exact-action/loss/decision counts, Episode/source coverage, support status, and deterministic ranking.

- [ ] Write a test with two synthetic exact-deck hashes that rejects file-hash joins and ranks by exact action after the minimum-evidence gate.
- [ ] Run `python3 -m unittest -v train.0032_dragapult_pod_native_cuda_rl.tests.test_selection_audit` and verify the test fails.
- [ ] Implement the count-manifest deck hash and joined ranking, requiring epoch 2, CUDA support, nonzero validation decisions, and exact 60-card identity.
- [ ] Run the test and materialize the audit; assert Mega Lopunny/Froslass 001 reports exact action `0.657664974619` within floating tolerance and `4,925` validation decisions, and record that business-value selection overrides the higher Marnie offline metric.

### Task 2: Load And Vectorize Frozen 0022 Deck Heads

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/model/frozen_heads.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_frozen_heads.py`
- Create during execution: `rl_runs/0032_dragapult_pod_native_cuda_rl/versions/V3_mega_lopunny_froslass_001_cuda_ppo/artifact/opponent_snapshot.json`

**Interfaces:**
- Consumes: 38 ordered deck IDs, 0022 model-only decoder checkpoints, POD option tokens, state summary, and lane deck indices.
- Produces: `FrozenDeckHeads.from_checkpoints(...)`, `FrozenDeckHeads.greedy(...) -> OrderedActions`, and a hash-complete provenance record.

- [ ] Write tests that reject the wrong schema/deck/hash/foundation, map every 0022 tensor by explicit semantic name, and compare the vectorized single-head calculation with the scalar GRUCell equations.
- [ ] Run the frozen-head tests and verify failure before implementation.
- [ ] Implement stacked batched linear/GRU operations without host routing, `.item()`, or per-lane Python dispatch.
- [ ] Verify all 38 supported deck checkpoints load, remain frozen, and produce legal unique selections under synthetic min/max contracts.

### Task 3: Add Device-Resident Action Evaluation And Player Routing

**Files:**
- Modify: `train/0032_dragapult_pod_native_cuda_rl/model/action_decoder.py`
- Modify: `train/0032_dragapult_pod_native_cuda_rl/model/actor_critic.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/model/routed_policy.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_routed_policy.py`

**Interfaces:**
- Produces: `ActionDecoder.evaluate(...) -> ActionEvaluation` and `RoutedResidentPolicy.act(batch, opponent_ids, stochastic_focal) -> RoutedActions`.
- `RoutedActions` contains `[B,64]` indices, `[B]` lengths/log-prob/value, focal masks, and legality, all on the input CUDA device.

- [ ] Test sampled-action log probabilities by replaying the same sequences, invalid prefix rejection, focal/opponent routing from actor ID, and zero gradients through frozen heads.
- [ ] Implement a fixed 64-step evaluator and select the focal sampled head only for player 0 while player 1 uses the frozen deck head.
- [ ] Capture greedy routing in a CUDA Graph and verify eager/capture/replay equality with no host synchronization in the hot path.
- [ ] Run all model and routing tests on CPU plus the CUDA capture smoke.

### Task 4: Implement Complete-Episode CUDA Rollout Storage

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/training/rollout.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/training/batch.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_rollout_batch.py`

**Interfaces:**
- Consumes: official engine statuses/game results, borrowed codec tensors, routed actions, and fixed `[steps, lanes]` storage.
- Produces: immutable `RolloutBatch` containing only focal decisions from complete Episodes, terminal returns, values, log probabilities, Episode weights, and source policy update.

- [ ] Test player-0 win/loss/draw reward mapping (`+1/-1/0`), reward assignment to the last focal decision, Episode-ID rollover, incomplete Episode exclusion, error-lane rejection, and per-Episode weighting.
- [ ] Add a tested GAE helper supporting both terminal and explicitly bootstrapped truncation; configure the formal collector to exclude incomplete trajectories so no implicit bootstrap is used.
- [ ] Store borrowed codec tensors by device clone only on focal decision rows and perform a single bulk host transfer after collection, never a per-step transfer.
- [ ] Run a 152-lane official CUDA canary and require nonzero completed Episodes, zero errors, legal actions, and internally consistent decision counts.

### Task 5: Implement POD-Native PPO And Model-Only Storage

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/training/ppo.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/training/storage.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_ppo.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_storage.py`

**Interfaces:**
- Produces: `PPOTrainer.update(batch) -> dict[str,float]`, `save_model_only(...) -> sha256`, and strict checkpoint validation.

- [ ] Test clipped policy loss, value loss, entropy, reference KL, target-KL early stop, finite gradients, and rollout/evaluation log-prob equality before the first optimizer step.
- [ ] Train only the focal decoder and value head initially; use a fresh AdamW optimizer, BF16 autocast, gradient clipping, Episode-balanced weights, and no serialized optimizer/RNG/buffer state.
- [ ] Test atomic checkpoints and reject optimizer/scheduler/scaler/RNG/DataLoader/rollout fields.
- [ ] Benchmark minibatch sizes 256/512/1024 and preserve the fastest non-OOM choice in training config.

### Task 6: Build The Formal Runner, Greedy Evaluation, And Watchdog

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/training/run.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/monitor_training.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_run_contract.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_monitor_training.py`

**Interfaces:**
- Produces CLI `python3 -m train.0032_dragapult_pod_native_cuda_rl.training.run` and watchdog CLI `python3 -m train.0032_dragapult_pod_native_cuda_rl.monitor_training`.

- [ ] Test fresh-version refusal, stable W&B run ID, metric namespaces, source-policy-update accounting, checkpoint retention, balanced frozen greedy seeds/seats, and watchdog stale/error detection.
- [ ] Log `rollout/*`, `ppo/*`, `checkpoint/update`, and end-to-end decisions/Episodes per second; log frozen greedy results separately under `eval/*` and `eval/checkpoint_update`.
- [ ] Persist status, training config, summary, exact opponent hashes/weights/seats/seeds, W&B state, GPU memory, disk capacity, and all official-engine errors.
- [ ] Run a two-update canary under the foreground watchdog and require one checkpoint per update plus canonical JSONL/TensorBoard/W&B-or-audited-offline evidence.

### Task 7: Synchronize Design, Benchmark, Start, And Commit

**Files:**
- Modify: `experiments/0032_dragapult_pod_native_cuda_rl/DESIGN.md`
- Modify: `experiments/0032_dragapult_pod_native_cuda_rl/DESIGN.html`
- Modify: `experiments/0032_dragapult_pod_native_cuda_rl/manifest.json`
- Modify: `evaluation/arena/combat_mat/0031_latest_frozen_test/THROUGHPUT.md`
- Create during execution: `rl_runs/0032_dragapult_pod_native_cuda_rl/versions/V3_mega_lopunny_froslass_001_cuda_ppo/artifact/training_config.json`

**Interfaces:**
- Consumes: code/test evidence, measured rollout/update/end-to-end throughput, selection/opponent audits, and canary status.
- Produces: synchronized authoritative design, a running formal V2, and a scoped Git commit.

- [ ] Cross-check every documented tensor shape, parameter count, hash, reward/value/PPO contract, current stage, and checkpoint boundary against code and artifacts.
- [ ] Benchmark 38/76/152/304 lanes with learner update included; select the highest stable end-to-end Episode throughput that leaves learner headroom.
- [ ] Preserve V2 as completed selection/performance preflight; start formal V3 training in a foreground watchdog session, poll at no more than 60 seconds, and intervene on any alert before continuing.
- [ ] Stage only 0032, required CUDA runtime, report, and plan files; inspect the staged diff for unrelated user changes; commit after formal training has demonstrably started.

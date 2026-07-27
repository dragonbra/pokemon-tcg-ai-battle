# 0017 Dragapult Terminal RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable official-engine RL loop for the 0015 Dragapult ex + Dusknoir R15 policy, then improve its frozen-catalog win rate using terminal win/loss/draw reward only.

**Architecture:** Freeze the exact 0015 V2 R15 feature, codec, actor, and portable-runtime implementation into self-contained 0017 modules. Add a scalar value head, a training-only stochastic autoregressive legal-action sampler, bounded in-memory complete-episode rollout, episode-balanced value calibration, and KL-anchored PPO. Keep official-engine processes isolated from the centralized GPU policy service; keep formal greedy evaluation separate from sampled training rollout.

**Tech Stack:** Python 3.11, PyTorch CUDA BF16, official engine runtime, multiprocessing, unittest, TensorBoard, W&B online.

## Global Constraints

- Project is `0017_dragapult_terminal_rl`; never import executable code from another numbered training project.
- Actor initialization is the model-only 0015 V2 exact-best checkpoint with SHA-256 `7f7589427098418682f3a53e0763f3fd24850d641a0a664955f6094f90530532`.
- Reward is terminal outcome only: win `+1`, loss `-1`, draw `0`; `gamma=1.0`. Diagnostics never alter reward.
- Never modify `engine/source/`, submit to Kaggle, admit an opponent, commit, or push while executing this plan unless separately authorized.
- Never save optimizer state or raw rollout traces by default. A restart from weights is a new immutable version, fresh optimizer, and fresh rollout.
- Stop gracefully before either `/` or `/mnt/c` falls below 50 GiB free; warn below 80 GiB; cap each version at 20 GiB.
- Training does not start until the user approves `experiments/0017_dragapult_terminal_rl/DESIGN.md` and `DESIGN.html`.

---

### Task 1: Freeze the 0015 R15 actor contract into 0017

**Files:**
- Create: `train/0017_dragapult_terminal_rl/model/`
- Create: `train/0017_dragapult_terminal_rl/features/`
- Create: `train/0017_dragapult_terminal_rl/runtime/`
- Create: `train/0017_dragapult_terminal_rl/tests/test_source_checkpoint.py`

**Interfaces:**
- Consumes: immutable 0015 V2 checkpoint and official card ontology provenance.
- Produces: self-contained `DragapultActorCritic`, exact actor-logit parity, scalar value prediction, and strict metadata loader.

- [ ] Copy and namespace the required R15 model, codec, feature compiler, card semantics, causal runtime, and source-conditioned policy code without cross-project imports.
- [ ] Add a scalar value head over the final conditioned state representation; initialize it independently without changing actor parameters or logits.
- [ ] Strictly load the source actor weights, require source ID 1 and the registered Dragapult + Dusknoir deck contract, and record source hashes.
- [ ] Test exact actor tensor parity against 0015 on fixed observations and require zero missing/unexpected actor keys.
- [ ] Test value shape `[batch]`, finite outputs, and unchanged actor parameter count of 17,416,642.

### Task 2: Implement stochastic legal-action sampling and log-probability

**Files:**
- Create: `train/0017_dragapult_terminal_rl/policy/action_distribution.py`
- Create: `train/0017_dragapult_terminal_rl/tests/test_action_distribution.py`

**Interfaces:**
- Consumes: R15 encoded state/options, legal option mask, `minCount`, and `maxCount`.
- Produces: ordered simulator action, summed autoregressive behavior log-probability, entropy, and replayable token choices including STOP.

- [ ] Implement temperature-1 categorical sampling at each pointer step over remaining legal options plus STOP when count bounds permit.
- [ ] Make action log-probability the sum of conditional option/STOP token log-probabilities under the exact sampled prefix.
- [ ] Enforce no duplicate option, `minCount`, `maxCount`, 16-token bound, and engine-valid ordered output.
- [ ] Implement batched teacher replay that reproduces stored old/new/reference log-probabilities for PPO and KL metrics.
- [ ] Test forced minimum, forced maximum, empty STOP, duplicate prevention, probability normalization, deterministic seeds, and sampled-action legality.

### Task 3: Build a bounded complete-episode official-engine collector

**Files:**
- Create: `train/0017_dragapult_terminal_rl/rollout/collector.py`
- Create: `train/0017_dragapult_terminal_rl/rollout/worker.py`
- Create: `train/0017_dragapult_terminal_rl/rollout/protocol.py`
- Create: `train/0017_dragapult_terminal_rl/tests/test_rollout_contract.py`

**Interfaces:**
- Consumes: frozen opponent snapshot, balanced seat schedule, immutable policy version, and official engine runtime.
- Produces: completed in-memory trajectories containing compact actor tensors, sampled token actions, old log-probabilities, old values, terminal outcome, and diagnostics.

- [ ] Run each game in an isolated worker process while one centralized GPU service batches policy inference.
- [ ] Retain only candidate decisions; attach terminal reward to a complete episode and discard engine errors or artificial truncations.
- [ ] Bound the queue and rollout buffer; release observations immediately after tensorization and release trajectories after an update.
- [ ] Freeze opponent IDs, package hashes, sampling weights, engine hash, seat schedule, seed, and policy hash in each version manifest.
- [ ] Add signal-aware graceful shutdown and disk guards for `/`, `/mnt/c`, and the 20 GiB version cap.
- [ ] Run a no-update official-engine probe and verify terminal attribution, seat attribution, no raw trace persistence, and bounded memory.

### Task 4: Implement value calibration and KL-anchored PPO

**Files:**
- Create: `train/0017_dragapult_terminal_rl/training/value.py`
- Create: `train/0017_dragapult_terminal_rl/training/ppo.py`
- Create: `train/0017_dragapult_terminal_rl/training/schedules.py`
- Create: `train/0017_dragapult_terminal_rl/tests/test_returns_and_ppo.py`

**Interfaces:**
- Consumes: complete on-policy episodes, frozen BC reference actor, and current actor-critic.
- Produces: episode-balanced value updates and clipped PPO actor updates with reference-KL protection.

- [ ] Compute terminal Monte Carlo returns with `gamma=1.0`; give each episode equal total value-loss weight so long games do not dominate.
- [ ] Calibrate the value head with the actor frozen and report outcome/phase/seat/opponent calibration.
- [ ] Compute GAE with `gamma=1.0`, initial `lambda=0.95`, normalize advantages, and never bootstrap artificial truncations.
- [ ] Implement PPO ratio clipping, value loss, entropy bonus, gradient clipping, fixed-BC reference KL, finite guards, and target-KL early stop.
- [ ] Start by unfreezing only the pointer decoder; keep representation and conditioned actor blocks frozen, and make broader unfreezing an explicit later-version change.
- [ ] Test return signs, draw handling, episode weighting, ratio identity, clipping, KL identity, nonfinite skip, and actor freeze schedules.

### Task 5: Add canonical logging, W&B dashboards, and model-only checkpoints

**Files:**
- Create: `train/0017_dragapult_terminal_rl/observability/logger.py`
- Create: `train/0017_dragapult_terminal_rl/observability/rolling.py`
- Create: `train/0017_dragapult_terminal_rl/checkpoint.py`
- Create: `train/0017_dragapult_terminal_rl/tests/test_observability.py`

**Interfaces:**
- Consumes: rollout, value, PPO, evaluation, performance, GPU, disk, and Dragapult diagnostic events.
- Produces: `training_metrics.jsonl`, TensorBoard, W&B online scalars/tables, status, summary, and model-only checkpoint metadata.

- [ ] Define `env/episodes`, `env/decisions`, and `trainer/update` axes and write local JSONL before TensorBoard and W&B mirrors.
- [ ] Log rolling 100/500/2,000 win rates, Wilson intervals, W/L/D, seat, opponent, archetype, outcome, duration, and policy hashes.
- [ ] Log PPO/value health, throughput/latency/backpressure, GPU memory/utilization, disk free space, and checkpoint bytes.
- [ ] Log Dragapult/Dusknoir evolution, Cursed Blast, relay, Phantom Dive, damage-counter, resource-budget, and attack-timing diagnostics as non-reward metrics.
- [ ] Save actor + value weights and metadata only; keep 6-8 checkpoints and prove optimizer keys are absent.
- [ ] Record stable private W&B run ID/URL and mirror failures without invalidating canonical local metrics.

### Task 6: Execute staged versions after design approval

**Files:**
- Create: `rl_runs/0017_dragapult_terminal_rl/versions/V1_contract_probe/`
- Create: `rl_runs/0017_dragapult_terminal_rl/versions/V2_value_calibration/`
- Create: `rl_runs/0017_dragapult_terminal_rl/versions/V3_ppo_pilot/`
- Create: `experiments/0017_dragapult_terminal_rl/evaluation/V1_contract_probe.html`
- Create: `experiments/0017_dragapult_terminal_rl/evaluation/V3_ppo_pilot.html`

**Interfaces:**
- Consumes: approved implementation, 20-opponent training snapshot, and 6-opponent unseen holdout snapshot.
- Produces: frozen current-catalog baseline, calibrated critic, PPO learning curves, model-only candidates, and comparable official-engine reports.

- [ ] V1: run 10 greedy games per each of the current 26 opponents, seat-balanced, with no updates; freeze packages and hashes.
- [ ] V2: collect fresh complete training-pool episodes and calibrate only the value head; discard each rollout after learning.
- [ ] V3: run a conservative PPO pilot with 256 completed episodes per update, 4 epochs, actor LR `1e-5`, value LR `1e-4`, clip `0.10`, entropy coefficient `0.01`, max grad norm `0.5`, and target behavior KL `0.02`.
- [ ] Evaluate frozen greedy snapshots every 20 updates with scalar-only reports; do not select from sampled training win rate alone.
- [ ] Stop or branch a new version on legality below 100%, repeated nonfinite updates, KL breach, holdout collapse, error/truncation excess, disk guard, or user stop.
- [ ] Run final 10-game-per-opponent official-engine confirmation against the same 26-opponent snapshot and publish immutable backlink/index records.

### Task 7: Synchronize authoritative design and audit evidence

**Files:**
- Modify: `experiments/0017_dragapult_terminal_rl/DESIGN.md`
- Modify: `experiments/0017_dragapult_terminal_rl/DESIGN.html`
- Modify: `experiments/0017_dragapult_terminal_rl/manifest.json`
- Create: `experiments/0017_dragapult_terminal_rl/decisions/`

**Interfaces:**
- Consumes: code contracts, checkpoint metadata, run manifests, W&B state, and official-engine reports.
- Produces: synchronized current-stage model/data/reward documentation and immutable decision history.

- [ ] Update both DESIGN documents in the same change whenever input schema, actor/value structure, action contract, loss, reward, rollout, or stage changes.
- [ ] Cross-check all tensor shapes, parameter counts, hashes, opponent membership, metrics, and stage claims against current artifacts.
- [ ] Run focused 0017 tests, evaluation asset tests when applicable, scoped compile, JSON parsing, link audit, and `git diff --check`.

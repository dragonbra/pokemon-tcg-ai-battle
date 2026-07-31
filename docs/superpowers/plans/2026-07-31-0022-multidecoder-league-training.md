# 0022 Multi-Decoder League Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue from the immutable Dragapult update-39 model branch and train all 48 deck-local decoder/value policies with isolated on-policy PPO while retaining frozen anchors and auditable matchup trends.

**Architecture:** One immutable 0019 encoder serves every request. Each Live deck owns a decoder/value state, fresh optimizer, policy update counter, trajectory buffer, reference snapshot, and model-only checkpoint. Official-engine games collect both players' trajectories; Frozen views never update, Live views route to the matching deck head, and updates publish only after the entire collection snapshot closes.

**Tech Stack:** Python 3.11, PyTorch PPO, multiprocessing official-engine workers, TensorBoard/W&B, model-only `.pt` checkpoints, HTML/Markdown design records.

## Global Constraints

- Never modify `engine/source/`; all rollout and strength evidence uses the official engine runtime.
- The 0019 Epoch-13 encoder and `source_id=0` remain frozen.
- Historical update 39 initializes only `dragapult_ex_001`; every optimizer is fresh and no old rollout/RNG state is reused.
- Every deck's loss consumes only decisions made by that exact policy version and reward sign.
- Frozen and Live evaluation are distinct; sampled rollout win rate is never reported as checkpoint strength.
- Checkpoints remain model-only, atomic, bounded, hash-addressed, and SSD guarded.
- Formal runs use new monotonically increasing 0022 versions and W&B online logging.

---

### Task 1: V2 Failure Audit And Multi-Policy Contracts

**Files:**
- Create: `experiments/0022_league_training/decisions/004_v2_outcome_and_multidecoder_boundary.md`
- Modify: `train/0022_league_training/rollout/protocol.py`
- Test: `train/0022_league_training/tests/test_rollout.py`

**Interfaces:**
- Consumes: V2 status/metrics, selected update-39 checkpoint.
- Produces: `TrajectoryDecision.policy_deck_id`, `TrajectoryDecision.policy_update`, and per-side trajectories with actor-relative rewards.

- [ ] Add failing protocol tests proving both sides retain separate policy identity, action history, and opposite terminal reward.
- [ ] Run `python3 -m unittest -v train.0022_league_training.tests.test_rollout` and confirm failure.
- [ ] Extend the trajectory protocol and worker/collector boundary without changing official-engine behavior.
- [ ] Re-run the rollout tests and record the V2 CUDA error evidence without claiming OOM.

### Task 2: Shared Encoder And Routed Live Heads

**Files:**
- Create: `train/0022_league_training/policy/league_pool.py`
- Modify: `train/0022_league_training/rollout/collector.py`
- Modify: `train/0022_league_training/policy/batching.py`
- Test: `train/0022_league_training/tests/test_policy.py`
- Test: `train/0022_league_training/tests/test_rollout.py`

**Interfaces:**
- Consumes: `dict[deck_id, decoder checkpoint]`, immutable foundation actor, request role/view.
- Produces: `LeaguePolicyPool.encode_and_route(batch, routes, mode)` and two-sided decision records.

- [ ] Add failing tests showing two deck heads produce independently routable logits while encoder parameters remain shared/frozen.
- [ ] Implement a policy pool that stores deck-local trainable heads and routes encoded states/options by deck ID.
- [ ] Update the collector so Frozen requests use the immutable foundation decoder and Live requests use the catalogued deck head.
- [ ] Verify a Live opponent action is recorded under the opponent deck, while Frozen opponent actions are not trainable trajectories.

### Task 3: Isolated Per-Deck PPO Updates

**Files:**
- Create: `train/0022_league_training/training/league_ppo.py`
- Modify: `train/0022_league_training/training/batch.py`
- Modify: `train/0022_league_training/training/run.py`
- Test: `train/0022_league_training/tests/test_training.py`

**Interfaces:**
- Consumes: two-sided `EpisodeTrajectory` objects grouped by exact deck/policy update.
- Produces: one fresh `PPOTrainer` per deck, `ppo/deck/<deck_id>/*` metrics, and atomic deck-local model-only checkpoints.

- [ ] Add failing tests that reject mixed deck IDs or mixed source policy updates in one optimizer step.
- [ ] Initialize Dragapult from selected update 39 and all other decks from their V2 update-0 materialized decoder checkpoints.
- [ ] Build fresh per-deck optimizers/references and update only decks meeting the configured minimum episode/decision threshold.
- [ ] Publish all updated heads after the collection snapshot closes; never mutate a behavior head during rollout.
- [ ] Bound per-deck checkpoint retention and preserve explicit best/branch snapshots.

### Task 4: Frozen, Live, And Fixed In-League Matchup Gates

**Files:**
- Create: `train/0022_league_training/evaluation.py`
- Modify: `train/0022_league_training/training/run.py`
- Test: `train/0022_league_training/tests/test_training.py`

**Interfaces:**
- Consumes: immutable Frozen pool and published Live pool snapshot.
- Produces: `eval/frozen/*`, `eval/live/*`, `eval/probe/<probe>/<view>/*`, balanced seat counts, and policy-version metadata.

- [ ] Add balanced greedy gates for focal versus all Frozen and all Live deck identities.
- [ ] Register the in-League identities `alakazam_dudunsparce_001` and `marnies_grimmsnarl_ex_froslass_001`, failing closed if either exact deck is missing. These are not external BC packages.
- [ ] For each identity, record focal-vs-Foundation-Frozen, focal-vs-deck-local-Live, and deck-local-Live-vs-focal-Live results with separate W-L-D, seats, policy versions, and Wilson intervals.
- [ ] Add per-matchup rolling sampled diagnostics without relabeling them as frozen greedy strength.

### Task 5: Documentation, Smoke, And CUDA Canary

**Files:**
- Modify: `experiments/0022_league_training/DESIGN.md`
- Modify: `experiments/0022_league_training/DESIGN.html`
- Modify: `train/0022_league_training/README.md`
- Modify: `train/0022_league_training/cli.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: self-contained documented CLI for multi-decoder smoke/canary/train.

- [ ] Synchronize tensor shapes, routing, loss ownership, update timing, checkpoint storage, and current phase in both DESIGN files.
- [ ] Run the complete 0022 unit suite plus evaluation/W&B regression suites.
- [ ] Run CPU official-engine smoke with two Live decks and verify both receive finite isolated gradients.
- [ ] Run a short CUDA canary with allocator diagnostics and a complete-update shutdown to investigate the V2 illegal access boundary.
- [ ] Refuse formal launch on any worker error, non-finite metric, mixed-policy batch, missing SHA, or failed Frozen/Live gate.

### Task 6: Formal 20-Hour League Run And Monitoring

**Files:**
- Create: `rl_runs/0022_league_training/versions/V3_multidecoder_league_20h/` through the version allocator.
- Create: `experiments/0022_league_training/decisions/005_v3_multidecoder_launch.md`

**Interfaces:**
- Consumes: validated V3 code, selected update-39 branch, 48-deck catalog.
- Produces: bounded per-deck checkpoints, canonical JSONL/TensorBoard/W&B metrics, Frozen/Live/probe trends, and candidate snapshots.

- [ ] Preflight free disk, package/runtime hashes, W&B connectivity, catalog count, and selected update-39 SHA.
- [ ] Launch the formal run with explicit duration, worker count, batch/coalescing, gate intervals, minimum per-deck data, and retention settings.
- [ ] Monitor process health, errors, throughput, CUDA allocation, disk free space, per-deck sample balance, KL/entropy/value health, and Frozen/Live/probe trends.
- [ ] At complete update boundaries, protect meaningful focal and League snapshots; never select a checkpoint solely from sampled rollout peaks.
- [ ] Continue until the time budget completes or a reproducible guardrail failure requires a new version.

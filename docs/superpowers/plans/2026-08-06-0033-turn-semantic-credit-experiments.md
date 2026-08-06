# 0033 Turn-Semantic Credit Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute this plan task-by-task with an analysis gate between every formal 20-update experiment. Do not pre-launch a later experiment.

**Goal:** Implement a deck-general turn-semantic credit clock, run three sequential 20-update official-engine PPO experiments with evidence-based redesign between them, then launch the selected design as a fresh 200-update run.

**Architecture:** Preserve the exact 0031 39-key actor contract, terminal official outcome, Frozen51 pool, decoder-only actor boundary and separate critic. Add the official observation turn to every focal trajectory decision, compute GAE with lambda applied only when the semantic turn changes, and keep all experiment variables explicit in version config and metrics. Each 20-update version starts independently from PT0805 Epoch 13, receives periodic frozen greedy evaluation, curve analysis and one 510-game formal official-engine evaluation before the next version is designed.

**Tech Stack:** Python 3.11, PyTorch PPO, unmodified official CPU engine workers, CUDA batched policy inference/training, unittest, JSONL/TensorBoard/W&B online, repository Evaluation CLI.

## Global Constraints

- Never modify `engine/source/`; all strength evidence uses the official engine runtime.
- Every run uses all 51 Frozen 0019 decks with balanced seats and recorded snapshot hashes.
- Actor input remains the complete 0031 semantic schema with 39 keys; hidden opponent identity never enters actor or critic forward.
- Only `actor.action_decoder.*` and `value_head.*` train; all model-only checkpoints are retained.
- Every experiment gets a fresh strictly increasing version and stable private W&B run ID.
- JSONL is canonical, followed by TensorBoard and W&B mirroring.
- A later experiment cannot start until the previous version has a completed analysis artifact and formal evaluation.
- The final 200-update run starts from PT0805 with a fresh optimizer; a 20-update trial curve is never extended or relabeled as the formal run.

---

### Task 1: Turn-Semantic Trajectory Contract

**Files:**
- Modify: `train/0033_dragapult_third_ptcg_club_rl/rollout/protocol.py`
- Modify: `train/0033_dragapult_third_ptcg_club_rl/rollout/collector.py`
- Modify: `train/0033_dragapult_third_ptcg_club_rl/training/batch_full_semantic.py`
- Modify: `train/0033_dragapult_third_ptcg_club_rl/tests/test_full_terminal_credit.py`

**Interfaces:**
- `TrajectoryDecision.turn: int` records the official observation `current.turn` at the focal decision.
- `_episode_gae(values, reward, turns, gamma, gae_lambda, credit_clock)` supports `selection` and `turn` clocks.
- Under `credit_clock="turn"`, the recursive multiplier is `1.0` when adjacent focal decisions have the same official turn and `gae_lambda` when it changes.

- [ ] Add failing tests proving same-turn decisions do not decay and cross-turn decisions decay exactly once.
- [ ] Add tests rejecting missing, negative or length-mismatched turn metadata.
- [ ] Record `current.turn` in the collector and implement variable-lambda GAE.
- [ ] Log semantic boundary counts and first-decision terminal residual weights.
- [ ] Run `python3 -m unittest -v train.0033_dragapult_third_ptcg_club_rl.tests.test_full_terminal_credit`.

### Task 2: Reusable Versioned Experiment Runner

**Files:**
- Modify: `train/0033_dragapult_third_ptcg_club_rl/training/ppo_full_semantic.py`
- Modify: `train/0033_dragapult_third_ptcg_club_rl/training/run_full_semantic.py`
- Modify: `train/0033_dragapult_third_ptcg_club_rl/tests/test_full_terminal_credit.py`
- Modify: `train/0033_dragapult_third_ptcg_club_rl/tests/test_project_identity.py`

**Interfaces:**
- CLI accepts `--credit-clock {selection,turn}`, `--gae-lambda FLOAT`, `--games-per-update {204,408}`, and `--eval-every N`.
- W&B run ID is a deterministic ASCII slug derived from the repository version and never reuses V3.
- Training config records reward, gamma, lambda, clock, episode weighting, exact schedule and precision contract.

- [ ] Add failing validation tests for unsupported gamma, lambda, clock, schedule and reused versions.
- [ ] Generalize `RunConfig` without weakening fresh-version checks.
- [ ] Make PPO accept `gamma=1.0` with `0 < lambda <= 1` and reject discounted gamma.
- [ ] Pass the configured clock into batch preparation and log it numerically and textually.
- [ ] Run all 0033 unit tests and the existing full-semantic PPO gate.

### Task 3: Experiment 1 - Turn Clock Lambda 0.97

**Files:**
- Create at runtime: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V6_turn_clock_lambda097_20u/`
- Modify after completion: `experiments/0033_dragapult_third_ptcg_club_rl/DESIGN.md`
- Modify after completion: `experiments/0033_dragapult_third_ptcg_club_rl/DESIGN.html`
- Modify after completion: `experiments/0033_dragapult_third_ptcg_club_rl/DECISIONS.md`

**Contract:** terminal-only `+1/0/-1`; `gamma=1.0`; same-turn lambda `1.0`; cross-turn lambda `0.97`; 204 Episodes/update; 20 updates; frozen greedy evaluation every 5 updates; existing episode-equal decision weighting.

- [ ] Run a 204-Episode one-update canary with zero errors and unchanged representation hash.
- [ ] Start V6 under the watchdog with W&B online and retain every checkpoint.
- [ ] After 20 updates, summarize rollout, eval, explained variance, return/advantage variance, KL, entropy, throughput and seat split into `artifact/analysis.json`.
- [ ] Select the best periodic frozen checkpoint by eval win rate, latest on ties.
- [ ] Export FP16-storage/FP32-runtime candidate and run a 510-game Frozen51 formal evaluation on GPU inference.
- [ ] Record whether V6 supports keeping the clock and whether variance improved relative to V3.

### Task 4: Experiment 2 - Evidence-Selected Variance Control

**Files:**
- Create at runtime: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V7_<recorded_hypothesis>_20u/`
- Update: the three authoritative experiment documents.

**Decision rule:** If V6 frozen eval improves without rising return variance, keep lambda 0.97 and test turn-equal loss weights. If V6 variance remains high with stable KL, keep the clock and test lambda 0.95. If V6 under-credits early decisions, test lambda 1.0 on the turn clock. The selected branch and rejected alternatives must be recorded before V7 starts.

- [ ] Write the V7 hypothesis and exact one-variable delta to `DECISIONS.md` before allocation.
- [ ] Add the required implementation test for that single delta.
- [ ] Run the gate, then 20 updates with frozen eval every 5 updates.
- [ ] Produce the same analysis and 510-game formal evaluation as V6.
- [ ] Compare V6/V7 under identical metric definitions and record the V8 branch decision.

### Task 5: Experiment 3 - Evidence-Selected Generalization Test

**Files:**
- Create at runtime: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V8_<recorded_hypothesis>_20u/`
- Update: the three authoritative experiment documents.

**Decision rule:** If credit timing is stable but sampling noise dominates, test 408 Episodes/update. If critic calibration is the bottleneck, add a frozen actor-visible potential delta or a critic calibration phase without direct Prize/damage/attack rewards. If V7 regresses, revert its delta and test the next-best V6-derived hypothesis. Record the exact selected path before allocation.

- [ ] Write the V8 hypothesis, evidence and exact delta before starting.
- [ ] Test fail-closed actor visibility and reward telescoping if potential shaping is selected.
- [ ] Run the gate, 20 updates, periodic frozen eval, curve analysis and 510-game formal evaluation.
- [ ] Rank V6/V7/V8 using zero-error operation, frozen official win rate, stability, early-credit retention and critic quality; rollout peaks are diagnostic only.

### Task 6: Conditional Legacy Action-Clock Control

**Files:**
- Create only if triggered: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V9_action_clock_lambda095_control_20u/`
- Update: the three authoritative experiment documents.

**Trigger:** Run this control if none of V6/V7/V8 has a credible advantage under the shared update-20 frozen seeds, 510-game official evaluation, critic variance and zero-error stability gates. A small point-estimate difference within sampling noise is not a credible advantage.

**Contract:** exact full 0031 semantic policy and official CPU engine; terminal-only reward; `gamma=1.0`; legacy per-selection `lambda=0.95`; 204 Episodes/update; 20 updates; frozen greedy evaluation every 5 updates. The control never uses the invalid V2 reduced CUDA/POD feature path.

- [ ] Record the failed superiority evidence that triggered the control.
- [ ] Run the same gate, 20-update schedule, analysis and 510-game formal evaluation.
- [ ] Add the control to the final selection matrix without giving it preferential treatment.

### Task 7: Selected 200-Update Run

**Files:**
- Create at runtime: `rl_runs/0033_dragapult_third_ptcg_club_rl/versions/V9_<selected_design>_200u/` when the control is not triggered, otherwise `V10_<selected_design>_200u/`.
- Update: `experiments/0033_dragapult_third_ptcg_club_rl/manifest.json`
- Update: `experiments/0033_dragapult_third_ptcg_club_rl/DESIGN.md`
- Update: `experiments/0033_dragapult_third_ptcg_club_rl/DESIGN.html`
- Update: `experiments/0033_dragapult_third_ptcg_club_rl/DECISIONS.md`

- [ ] Record the selection matrix and why the two rejected designs lost.
- [ ] Allocate a fresh version, optimizer and W&B run from PT0805 Epoch 13.
- [ ] Run the selected contract to 200 updates under the watchdog, with all checkpoints retained and periodic frozen greedy evaluation.
- [ ] Verify monitoring, disk projection, W&B health and official worker fail-closed behavior.
- [ ] At completion, export the selected checkpoint and run the full formal Frozen51 evaluation before any candidate promotion decision.

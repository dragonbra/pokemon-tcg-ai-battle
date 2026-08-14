# 0044 Training Agent-Owned First Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 0044 PPO rollout's fixed 128/128 seat assignment with deterministic seeded toss winners followed by the winning complete policy's real official context-41 choice, and persist the actual resulting seat.

**Architecture:** Reuse the already-valid Benchmark V2 CUDA path: schedules own only reproducible seeds and toss-winner identity; the resident CUDA runner places the toss winner as the context-41 chooser; focal or frozen opponent policy then selects first/second. PPO preparation and telemetry consume the recorded choice, never a preassigned seat. Historical V10 remains immutable and explicitly documented as fixed-seat training.

**Tech Stack:** Python 3.12, PyTorch PPO, CUDA Engine 2.0 resident collector, pytest, Markdown/HTML design documents.

## Global Constraints

- Do not modify `engine/source/`.
- Do not rewrite V10 artifacts or metrics; the correction belongs to a new training version.
- A seed may determine only the toss winner. The harness must not assign, alternate, or balance actual first/second seats.
- The toss-winning focal or complete immutable Champion-G2 policy must process official context 41.
- Missing toss seed, toss-winner identity, Agent choice, or actual seat is fatal before PPO optimization.
- Opponent policy identity, exact deck binding and CUDA resident routing remain unchanged.
- Update both `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md` and `DESIGN.html`.

---

### Task 1: Seeded toss schedule contract

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/league/sampler.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/tests/test_sampler.py`

**Interfaces:**
- Produces: `LeagueLane.coin_winner_seed: int` and `LeagueLane.focal_won_toss: bool`.
- Removes schedule authority over actual first player.

- [ ] Write tests asserting unique deterministic coin-winner seeds and the absence of an exact 128/128 actual-seat contract.
- [ ] Run `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_sampler.py` and confirm the old fixed-seat implementation fails.
- [ ] Derive `coin_winner_seed` with the existing namespaced `_seed` function and derive only `focal_won_toss` from it.
- [ ] Run the sampler tests and confirm PASS.

### Task 2: Agent-owned context 41 in formal training

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/run_v1.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/rollout/protocol.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/rollout/cuda_collector.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_v10_g3_generalist.py`

**Interfaces:**
- Consumes: `LeagueLane.coin_winner_seed` and `LeagueLane.focal_won_toss`.
- Produces: every formal training `RolloutJob` has a seeded toss winner and every terminal episode has `diagnostics.first_player_choice.focal_first`.

- [ ] Write a failing job-construction test requiring seeded toss metadata and forbidding fixed-seat claims.
- [ ] Set training `RolloutJob.focal_won_toss` from the lane schedule; keep the legacy `focal_first` constructor field only as resident chooser placement, not as actual-seat evidence.
- [ ] Construct every formal training collector with `agent_selects_first_player=True`.
- [ ] Add a collector/result hard gate requiring exactly one valid context-41 choice per terminal episode and a boolean actual focal seat.
- [ ] Record `coin_winner_seed`, toss winner, chooser policy role, action index and actual seat in diagnostics.
- [ ] Run the focused training tests and confirm PASS.

### Task 3: PPO and telemetry actual-seat source

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/batch_full_semantic.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/telemetry.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_training_regression.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_telemetry.py`

**Interfaces:**
- Consumes: `episode.diagnostics["first_player_choice"]["focal_first"]`.
- Produces: `PreparedBatch.candidate_first` and seat metrics bound to actual Agent-selected seats.

- [ ] Write a failing test where toss winner differs from actual first player.
- [ ] Replace every training use of `episode.job.focal_first` as evidence with the validated diagnostic actual seat.
- [ ] Fail closed when training episodes lack toss/choice/actual-seat metadata.
- [ ] Run focused batch and telemetry tests and confirm PASS.

### Task 4: Historical boundary and authoritative design

**Files:**
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md`
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html`
- Modify: `train/0044_g2_dragapult_policy_option_lora/README.md`

**Interfaces:**
- Produces: an auditable statement that V10 used legacy fixed-seat training while Benchmark V2 already used Agent-owned context 41, and that every successor uses the corrected contract.

- [ ] Document the seeded-toss data flow and required manifest fields.
- [ ] Document that actual seat counts are observations, not schedule quotas.
- [ ] Document that opponent deck sampling remains a separate experiment variable.
- [ ] Cross-check Markdown and HTML for identical current-stage facts.

### Task 5: Verification

**Files:**
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/`

**Interfaces:**
- Proves: schedule reproducibility, Agent-owned context 41, actual-seat telemetry, opponent-policy isolation and no CUDA routing regression.

- [ ] Run all focused sampler, rollout, batch, telemetry and V10/V11 contract tests.
- [ ] Run `pytest -q train/0044_g2_dragapult_policy_option_lora/tests`.
- [ ] Run the smallest available CUDA Engine 2.0 context-41 smoke without starting a formal training run.
- [ ] Inspect `git diff --check` and confirm no `engine/source/` changes.

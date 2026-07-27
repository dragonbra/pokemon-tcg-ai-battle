# 0015 Label-Smoothing BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute inline because the repository does not
> expose `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Do not
> delegate or commit.

**Goal:** Test whether a small amount of train-only label smoothing improves the rollout strength
of the unchanged source-conditioned R15 T1 policy under scarce demonstrations.

**Architecture:** Keep model, inputs, dataset, source/deck conditioning, validation, decoder and
candidate runtime byte-compatible with V2. Apply `label_smoothing=0.05` only inside the token-level
training cross-entropy; validation remains ordinary unsmoothed teacher-forced and greedy full-action
evaluation.

**Tech Stack:** Python 3.11, PyTorch BF16/CUDA, unittest, TensorBoard, W&B online, official engine.

## Global Constraints

- V14 is a fresh R15 T1 run with seed `20260723`; no rule reader or warm start.
- Only training CE changes. Rows, order, epochs, optimizer, target validation and selector match V2.
- Record smoothing explicitly in config/checkpoints/W&B; `0.0 <= epsilon < 1.0`.
- Do not modify `engine/source/`, commit, push, submit to Kaggle or admit a candidate.

---

### Task 1: Smoothed train-only objective

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/training/outcome_weighting.py`
- Modify: `train/0015_dragapult_conditioned_bc/training/outcome_weighted_trainer.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_outcome_weighting.py`

- [ ] Add failing tests for invalid epsilon, equality at epsilon zero, padding exclusion and the
  exact PyTorch smoothed CE reference at epsilon 0.05.
- [ ] Thread epsilon through the existing weighted token loss and optimization callback without
  changing accuracy diagnostics or validation code.
- [ ] Preserve the V4/V5 outcome-weighting behavior when epsilon is zero.

### Task 2: Formal CLI and V14 run

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/run.py`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V14_t1_label_smoothing_005/`

- [ ] Add `--label-smoothing`, validate its range and record it in every formal artifact.
- [ ] Run focused tests and a five-batch CUDA BF16 smoke with epsilon 0.05.
- [ ] Preflight all V14 paths, then start formal training immediately after the smoke.
- [ ] Export best target-exact, validate the package and run the unchanged 200-game official-engine
  contract; stop smoothing exploration unless V14 beats V2.

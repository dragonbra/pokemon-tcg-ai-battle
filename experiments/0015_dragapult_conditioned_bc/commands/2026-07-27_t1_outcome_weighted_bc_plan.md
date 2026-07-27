# T1 Outcome-Weighted BC Implementation Plan

> **For agentic workers:** execute inline in the current session. The repository does not expose
> the `superpowers:executing-plans` skill, so track the concrete steps below directly and do not
> delegate them.

**Goal:** Test whether retaining every T1 trajectory while moderately down-weighting failed-game
decisions improves the target Dragapult + Dusknoir policy beyond V2.

**Architecture:** Keep the frozen R15 + source-persona model, T1 dataset membership, target-only
validation and optimizer contract unchanged. Add a local 0015 weighted trainer that applies one
outcome-derived positive weight to every valid action token during the training loss only; all
accuracy, legality, validation, checkpoint selection and engine evaluation remain unweighted.

**Tech Stack:** Python 3.11, PyTorch BF16/CUDA, unittest, TensorBoard, W&B online, official engine.

## Global Constraints

- Do not modify `engine/source/` or any frozen V1–V3 artifact.
- Keep all win and loss trajectories; weighting changes loss contribution, not membership.
- Use T1's 54,764 training decisions and the same 1,000-decision target validation.
- Fresh initialization with seed `20260723`, batch 256, LR `3e-4`, weight decay `0.02`.
- Validation and the best-greedy-exact checkpoint selector remain completely unweighted.
- Allocate strict new versions and W&B runs; write canonical metrics locally before mirrors.
- Official strength claims require the same 20-opponent, 10-game, 8-worker engine contract.
- Candidate admission, Kaggle submission/upload, commit and push remain unauthorized.

---

### Task 1: Outcome-weighted loss primitive

**Files:**
- Create: `train/0015_dragapult_conditioned_bc/training/outcome_weighting.py`
- Create: `train/0015_dragapult_conditioned_bc/tests/test_outcome_weighting.py`

**Interfaces:**
- Consumes: `outcome_id` tensors where `1=win`, `-1=loss`, and `0=draw/unknown`.
- Produces: `outcome_weights(outcome_ids, win_weight, loss_weight, draw_weight) -> Tensor` and
  `weighted_token_cross_entropy(logits, targets, weights) -> (loss, numerator, denominator)`.

- [x] Write tests proving exact weight mapping, invalid-ID rejection, uniform equivalence to normal
  token CE, padding exclusion, and finite backward gradients.
- [x] Run the tests and confirm they fail before implementation.
- [x] Implement positive finite weight validation and token-normalized weighted CE.
- [x] Run the tests and compile the package.

### Task 2: 0015-local weighted trainer and CLI contract

**Files:**
- Create: `train/0015_dragapult_conditioned_bc/training/outcome_weighted_trainer.py`
- Modify: `train/0015_dragapult_conditioned_bc/run.py`

**Interfaces:**
- Consumes the existing arm batch iterator, which already includes `outcome_id`.
- Produces the same checkpoint/status/summary contract as the frozen trainer plus
  `bc/optimization/weighted_loss`, mean decision weight and per-outcome decision counts.

- [x] Copy the frozen trainer locally and change only the optimization loss and weighted
  diagnostics; preserve unweighted online token/exact metrics and the existing evaluation code.
- [x] Add `--win-weight`, `--loss-weight`, and `--draw-weight` flags with positive finite validation.
- [x] Dispatch to the weighted trainer only when any weight differs from 1.0; uniform defaults must
  retain the original V1–V3 path.
- [x] Record the exact weight contract in config, model contract, checkpoint metadata and W&B tags.
- [x] Run a one-epoch CUDA smoke on T1 loss-weight 0.75 and verify finite gradients, 1,000-decision
  unweighted validation, legality 1.0, checkpoint creation and W&B disabled smoke behavior.

### Task 3: Formal bounded sweep

**Files:**
- Create immutable versions under `rl_runs/0015_dragapult_conditioned_bc/versions/`.

**Interfaces:**
- V4: T1 with win weight 1.0 and loss weight 0.75.
- V5: T1 with win weight 1.0 and loss weight 0.50.

- [x] Train V4 from the same fresh seed and wait through early stopping/W&B sync.
- [x] Train V5 from the same fresh seed and wait through early stopping/W&B sync.
- [x] Select each candidate by the same best target greedy exact rule, never by engine win rate.
- [x] Compare target curves with V2 and verify validation stays unweighted and 100% legal.

### Task 4: Candidate export and official-engine evaluation

**Files:**
- Create candidates in `evaluation/arena/candidates/`.
- Create formal reports in `experiments/0015_dragapult_conditioned_bc/evaluation/`.

**Interfaces:**
- Uses the frozen THIRD PTCG Club target deck/persona and existing portable R15 exporter.
- Produces V4/V5 immutable report backlinks and a refreshed project evaluation index.

- [x] Export and validate both best-greedy-exact checkpoints.
- [x] Run 200 official-engine games for each candidate under the frozen catalog/worker contract.
- [x] Record W/L/error, completion, both seats and relative changes versus V2 and V1.
- [x] Do not promote any candidate into the formal opponent pool.

### Task 5: Documentation and completion audit

**Files:**
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`
- Modify: `experiments/0015_dragapult_conditioned_bc/RUNBOOK.md`
- Create: `experiments/0015_dragapult_conditioned_bc/decisions/006_t1_outcome_weight_results.md`

**Interfaces:**
- Consumes canonical local metrics, W&B status and official reports.
- Produces one honest decision: improvement, no gain, or inconclusive.

- [x] Sync tensor/loss contracts, actual version results and current project phase in both DESIGN
  formats.
- [x] Verify report SHA backlinks, W&B sync state, candidate packages, relevant tests, compileall
  and `git diff --check`.
- [x] Preserve known unrelated baseline failures separately; do not claim a full-repository pass.

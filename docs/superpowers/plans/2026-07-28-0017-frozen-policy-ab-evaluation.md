# 0017 Frozen Policy A/B Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the exact pre-RL BC policy and the selected V14 RL policy, then determine whether RL changed official-engine win rate beyond plausible sampling noise.

**Architecture:** Reuse the self-contained 0017 candidate exporter and the repository evaluation runner. Run equal-size, seat-balanced frozen-policy evaluations against one immutable 26-opponent catalog, then compare the embedded per-game records with stratified statistical summaries. The official binary uses `std::random_device` internally and exposes no seed API, so the result is an independent, balanced A/B comparison rather than a shared-seed paired experiment.

**Tech Stack:** Python 3.11, PyTorch CPU greedy inference, official `cg` runtime, repository `evaluation` package, standard-library statistical analysis.

## Global Constraints

- Do not modify `engine/source/`.
- Use the exact 0017 source checkpoint SHA256 `7f7589427098418682f3a53e0763f3fd24850d641a0a664955f6094f90530532` as the pre-RL baseline.
- Use V14 update 15 as the primary post-RL policy; retain update 30 only as a declared secondary checkpoint.
- Evaluate every enabled opponent in the frozen 26-package catalog with 10 games per opponent and alternating candidate seat, as explicitly requested by the user after stopping the original 100-game attempt.
- Use `--workers 8 --worker-cpu-threads 1` and the official engine runtime.
- Do not call the comparison paired: the official runtime does not expose deterministic battle seeding.
- Preserve unrelated dirty 0015, 0016, and environment-daily worktree changes.

---

### Task 1: Export and validate frozen candidate packages

**Files:**
- Create: `evaluation/arena/candidates/0017_bc_source_r15/`
- Create: `evaluation/arena/candidates/0017_v14_update15_rl/`

**Interfaces:**
- Consumes: `train.0017_dragapult_terminal_rl.export_candidate.export_candidate(checkpoint, cg_source, output)`
- Produces: two standard submission packages accepted by `python3 -m evaluation validate`

- [ ] **Step 1: Export the BC source package**

Run:

```bash
python3 -m train.0017_dragapult_terminal_rl.export_candidate \
  --checkpoint rl_runs/0015_dragapult_conditioned_bc/versions/V2_t1_plus_pure_dragapult/checkpoint/epoch-0010-7f75894270984186.pt \
  --cg-source evaluation/arena/opponents/dragapult_ex_02/cg \
  --output evaluation/arena/candidates/0017_bc_source_r15
```

- [ ] **Step 2: Export the V14 update 15 package**

Run:

```bash
python3 -m train.0017_dragapult_terminal_rl.export_candidate \
  --checkpoint rl_runs/0017_dragapult_terminal_rl/versions/V14_ppo_lambda097_lr3e6/checkpoint/update-000015.pt \
  --cg-source evaluation/arena/opponents/dragapult_ex_02/cg \
  --output evaluation/arena/candidates/0017_v14_update15_rl
```

- [ ] **Step 3: Validate both package contracts**

Run:

```bash
python3 -m evaluation validate evaluation/arena/candidates/0017_bc_source_r15
python3 -m evaluation validate evaluation/arena/candidates/0017_v14_update15_rl
```

Expected: both report `60-card valid: true` with identical deck and `cg` hashes.

### Task 2: Allocate immutable formal evaluation versions

**Files:**
- Create: `rl_runs/0017_dragapult_terminal_rl/versions/V17_bc_source_eval_10g/`
- Create: `rl_runs/0017_dragapult_terminal_rl/versions/V18_rl_update15_eval_10g/`

**Interfaces:**
- Consumes: `rl_environment.runs.initialize_version(project_id, version_name)`
- Produces: valid formal evaluation destinations and immutable status records

- [ ] **Step 1: Record V15/V16 as interrupted and allocate V17**

Record the stopped 100-game attempts without publishing reports. Create V17 with `initialize_version`, then write `training_config.json` identifying the BC source package, checkpoint hash, 26-opponent catalog hash, 10 games per opponent, alternating seats, and no W&B training run.

- [ ] **Step 2: Allocate V18 and record its evaluation-only contract**

Create the version with the identical 10-game evaluation contract except for the V14 update 15 checkpoint and package hash.

- [ ] **Step 3: Verify neither formal report nor backlink already exists**

Run:

```bash
test ! -e experiments/0017_dragapult_terminal_rl/evaluation/V17_bc_source_eval_10g.html
test ! -e experiments/0017_dragapult_terminal_rl/evaluation/V18_rl_update15_eval_10g.html
```

### Task 3: Run the pre-RL frozen-policy evaluation

**Files:**
- Create: `experiments/0017_dragapult_terminal_rl/evaluation/V17_bc_source_eval_10g.html`
- Create: `rl_runs/0017_dragapult_terminal_rl/versions/V17_bc_source_eval_10g/artifact/evaluation.json`

**Interfaces:**
- Consumes: validated BC package and current frozen catalog
- Produces: 260 official-engine game records embedded in the formal report

- [ ] **Step 1: Run the complete evaluation**

Run:

```bash
python3 -m evaluation run \
  --candidate evaluation/arena/candidates/0017_bc_source_r15 \
  --opponents all --games 10 \
  --workers 4 --worker-cpu-threads 1 \
  --metric-profile core \
  --output experiments/0017_dragapult_terminal_rl/evaluation/V17_bc_source_eval_10g.html
```

- [ ] **Step 2: Verify coverage**

Expected: 260 records, 130 candidate-first games, 130 candidate-second games, zero package/runtime errors.

### Task 4: Run the post-RL frozen-policy evaluation

**Files:**
- Create: `experiments/0017_dragapult_terminal_rl/evaluation/V18_rl_update15_eval_10g.html`
- Create: `rl_runs/0017_dragapult_terminal_rl/versions/V18_rl_update15_eval_10g/artifact/evaluation.json`

**Interfaces:**
- Consumes: validated RL package and the same current frozen catalog
- Produces: 260 independently randomized official-engine game records

- [ ] **Step 1: Run the complete evaluation**

Run the same command as Task 3, replacing the candidate and output with `0017_v14_update15_rl` and `V18_rl_update15_eval_10g.html`.

- [ ] **Step 2: Verify the evaluation contracts match**

Compare both embedded manifests. Opponent names, package hashes, game counts, seat schedule, runtime hash, max steps, workers, worker CPU threads, and metric profile must match; only candidate identity and wall-clock fields may differ.

### Task 5: Quantify strength change and sampling uncertainty

**Files:**
- Create: `train/0017_dragapult_terminal_rl/compare_frozen_evaluations.py`
- Modify: `train/0017_dragapult_terminal_rl/tests/test_training.py`
- Create: `experiments/0017_dragapult_terminal_rl/decisions/007_frozen_policy_ab_result.md`

**Interfaces:**
- Consumes: two formal HTML reports containing `<script id="report-data" type="application/json">`
- Produces: JSON/Markdown summary for overall, train-pool, holdout, seat, and opponent strata

- [ ] **Step 1: Add failing tests for report parsing and two-proportion inference**

Test exact game-count validation, Wilson intervals, an unpooled two-sided difference test, and rejection of mismatched opponent/seat contracts.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
python3 -m unittest -v train.0017_dragapult_terminal_rl.tests.test_training
```

- [ ] **Step 3: Implement the minimum comparison module**

Parse embedded report data, verify equal opponent and seat allocation, compute win counts and rates, Wilson 95% intervals, absolute difference, unpooled 95% interval for the difference, and a two-sided score-test p-value. Group by all games, the frozen 20-opponent training pool, the six-opponent holdout pool, seat, and individual opponent.

- [ ] **Step 4: Run tests and analyze the reports**

Run:

```bash
python3 -m unittest -v train.0017_dragapult_terminal_rl.tests.test_training
python3 -m train.0017_dragapult_terminal_rl.compare_frozen_evaluations \
  --baseline experiments/0017_dragapult_terminal_rl/evaluation/V17_bc_source_eval_10g.html \
  --candidate experiments/0017_dragapult_terminal_rl/evaluation/V18_rl_update15_eval_10g.html \
  --output-json rl_runs/0017_dragapult_terminal_rl/versions/V18_rl_update15_eval_10g/artifact/comparison.json \
  --output-markdown experiments/0017_dragapult_terminal_rl/decisions/007_frozen_policy_ab_result.md
```

- [ ] **Step 5: State the evidence threshold honestly**

Call RL improved only if the overall 95% difference interval excludes zero in the positive direction, completion/legal-action behavior remains intact, and the gain is not solely one seat or one opponent. Otherwise report the result as inconclusive or regressed, regardless of the training rolling-window peak.

### Task 6: Synchronize authoritative project documentation

**Files:**
- Modify: `experiments/0017_dragapult_terminal_rl/DESIGN.md`
- Modify: `experiments/0017_dragapult_terminal_rl/DESIGN.html`

**Interfaces:**
- Consumes: immutable V15/V16 reports and Task 5 statistics
- Produces: current project-stage and evidence-boundary documentation

- [ ] **Step 1: Add the frozen-policy A/B result and evidence boundary**

Record that official game randomness is not seed-controllable, distinguish sampled rollout win rate from frozen greedy strength, and link both reports and Decision 007.

- [ ] **Step 2: Cross-check documentation against package manifests and report payloads**

Verify checkpoint hashes, game counts, opponent split, seats, completion, confidence intervals, and decision wording.

- [ ] **Step 3: Run final verification**

Run:

```bash
python3 -m unittest -v \
  train.0017_dragapult_terminal_rl.tests.test_policy \
  train.0017_dragapult_terminal_rl.tests.test_training
python3 -m compileall -q train/0017_dragapult_terminal_rl evaluation rl_environment
git diff --check
```

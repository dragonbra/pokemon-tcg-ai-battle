# Opponent Invalid-Action Forfeit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count a policy-side invalid selection rejected by the official runtime as a forfeit for that policy, then publish separate full-pool reports for Marnie and `alakazam_dudunsparce_03`.

**Architecture:** `cg.game.battle_select()` reports a nonzero official-engine selection result as `IndexError`; the worker still knows which policy supplied the immediately preceding action. Catch only that exception at the policy-to-engine boundary and serialize it as `candidate_error` or `opponent_error`, with the non-failing side as winner. Leave start, native-runtime, finish, and non-`IndexError` game failures unchanged.

**Tech Stack:** Python 3.11, standard-library unittest, official `cg` engine runtime, evaluation CLI.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve `cg` mismatch fail-fast behavior.
- Do not convert engine/runtime failures into wins.
- Formal arena reports use the current fixed opponent catalog and ten games per opponent.
- Temporary reports and traces stay under `.tmp/evaluation/`.

---

### Task 1: Capture invalid-action attribution in the worker

**Files:**
- Modify: `tests/test_evaluation_worker.py`
- Modify: `evaluation/runner/worker.py`

**Interfaces:**
- Consumes: `game_module.battle_select(action)` raising `IndexError` for an official-runtime rejected action.
- Produces: `GameResult(status="candidate_error"|"opponent_error", winner=1|0, finished=True)`.

- [ ] **Step 1: Write failing worker tests**

Add a fake-game option that raises `IndexError` during `battle_select`. Test candidate-first and candidate-second requests so a rejected candidate action produces `winner == 1`, while a rejected opponent action produces `winner == 0`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_worker`

Expected: new assertions fail because the worker currently emits generic `game_error` with no winner.

- [ ] **Step 3: Implement the narrow exception boundary**

Wrap only the `game_module.battle_select(action)` call in `except IndexError`. Derive the failing role from `current_player == candidate_physical_index`, preserve the selected action in trace, mark the result finished, and normalize the opposite physical player as winner.

- [ ] **Step 4: Verify worker and metric tests**

Run: `python3 -m unittest -v tests.test_evaluation_worker tests.test_evaluation_metrics`

Expected: PASS; generic game exceptions remain `game_error`.

### Task 2: Package and evaluate the two requested candidates

**Files:**
- Create: `evaluation/arena/candidates/alakazam_dudunsparce_03_candidate/`
- Create: `.tmp/evaluation/marnie_forfeit_rule/run-<id>/report.html`
- Create: `.tmp/evaluation/alakazam_dudunsparce_03_candidate/run-<id>/report.html`

**Interfaces:**
- Consumes: the tested worker semantics and baseline-compatible candidate packages.
- Produces: two 10-games-per-opponent official-runtime reports, each with opponent-policy forfeits represented as candidate wins.

- [ ] **Step 1: Create a self-contained candidate copy**

Copy the existing standard package `evaluation/arena/opponents/alakazam_dudunsparce_03/` to `evaluation/arena/candidates/alakazam_dudunsparce_03_candidate/` without changing its deck, model, or `cg/`.

- [ ] **Step 2: Validate packages**

Run: `python3 -m evaluation validate evaluation/arena/candidates/agent_bc_marnie_1100` and `python3 -m evaluation validate evaluation/arena/candidates/alakazam_dudunsparce_03_candidate`

Expected: both are 60-card valid and use the fixed-pool CG manifest.

- [ ] **Step 3: Run requested full-pool reports**

Run each candidate with `--opponents all --games 10 --workers 8 --worker-cpu-threads 1` and a distinct `.tmp/evaluation/` output path.

- [ ] **Step 4: Inspect report summaries**

Verify that each report has 200 planned games, that an `opponent_error` is a candidate win, and report whether `alakazam_dudunsparce_03` fails across unrelated candidate decks.

### Task 3: Regression verification and handoff

**Files:**
- Verify: `tests/test_evaluation_worker.py`
- Verify: `tests/test_evaluation_metrics.py`
- Verify: `tests/test_evaluation_assets.py`

- [ ] **Step 1: Run focused regression suite**

Run: `python3 -m unittest -v tests.test_evaluation_worker tests.test_evaluation_metrics tests.test_evaluation_assets`

Expected: PASS.

- [ ] **Step 2: Review worktree scope**

Run: `git status --short` and confirm that unrelated 0014 work remains untouched.

- [ ] **Step 3: Deliver report links and scope statement**

State the win/loss/error totals for both candidates and distinguish invalid-action forfeits from ordinary game wins.

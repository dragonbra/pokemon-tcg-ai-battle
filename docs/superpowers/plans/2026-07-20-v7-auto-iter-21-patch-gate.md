# V7 AutoIter Iter-21 Wondrous Patch Gate Implementation Plan

> **For agentic workers:** This is a single-variable AutoIter experiment. Keep the fixed deck and the existing best artifact unchanged until the candidate has been evaluated.

**Goal:** Prevent Wondrous Patch from consuming Basic Psychic on an already-energized Bench Abra when no visible Kadabra/Rare Candy route can turn an unenergized Abra into a future attacker.

**Architecture:** Keep the existing option-scoring architecture. Refine only the `valid_target` predicate used by `_bench_handoff_preparation_due`; Patch remains preferred for Bench Kadabra/Alakazam and for an unenergized Bench Abra when the next evolution route is visible. Add a replay-derived regression test and record the candidate as an observe/reject/accept iteration without changing `deck.csv`.

**Tech Stack:** Python 3.11+, unittest, existing evaluator trace analyzer, Bash/LaunchAgent.

## Global Constraints

- `submission/alakazam_v7_auto_iter/deck.csv` is fixed and must not change.
- Attack remains a terminal action for the current turn.
- Do not infer a future card draw; only visible hand, discard, field, and legal options may establish a route.
- Preserve the accepted immutable best marker unless the candidate passes the existing guardrails.
- Keep raw evaluator traces outside this repository; retain compact advisor, decision, and metrics files.

### Task 1: Replay regression first

**Files:**
- Modify: `tests/test_alakazam_v7_auto_iter_strategy.py`

- [ ] Replace the old expectation that Patch should charge an Abra without a visible evolution route with the iter-20 replay case: Active Alakazam has Psychic, Bench contains one energized Abra and one unenergized Abra, discard contains Basic Psychic, hand contains Patch and Alakazam but no Kadabra/Rare Candy, and a non-terminal Powerful Hand is legal.
- [ ] Assert `_main_action()` selects Powerful Hand rather than Patch.
- [ ] Run the focused test and confirm it fails against the current broad gate.

### Task 2: Minimal strategy change

**Files:**
- Modify: `submission/alakazam_v7_auto_iter/main.py`

- [ ] In `_bench_handoff_preparation_due`, make an Abra target valid only when it is unenergized and either Kadabra is visible in hand or Alakazam plus Rare Candy is visible and Items are not locked.
- [ ] Leave Kadabra/Alakazam Patch targets unchanged.
- [ ] Run the focused regression and the existing Patch/handoff strategy tests.

### Task 3: Scheduled submission labeling

**Files:**
- Modify: `scripts/submit_kaggle_v7_best.sh`

- [ ] Include explicit `iteration=<best_iteration>`, label, and archive SHA in the Kaggle submission message and log output.
- [ ] Keep submission pointed at the marker's immutable archive; do not repackage the dirty working tree at trigger time.

### Task 4: Full evaluation and iteration record

**Files:**
- Create: `reports/kaggle/alakazam-v7-auto-iter/iter-21/advisor.md`
- Create: `reports/kaggle/alakazam-v7-auto-iter/iter-21/decision.md`
- Create: `reports/kaggle/alakazam-v7-auto-iter/iter-21/metrics.json`
- Modify: `submission/alakazam_v7_auto_iter/ITER_PROCESS.md`
- Modify: `submission/alakazam_v7_auto_iter/STRATEGY.md` only if the accepted behavior changes the durable strategy description.

- [ ] Run the focused replay/evaluator check for the Patch case.
- [ ] Run the fixed 17-opponent × 10-game full-trace evaluation if disk space permits; otherwise record the exact limitation.
- [ ] Compare against the current control and decide `accept`, `observe`, or `reject` using win rate, Meta weighted win rate, second-turn Powerful Hand, post-KO no-ready event rate, bench-break game rate, and action errors.
- [ ] Promote a new best only if the guardrails justify it; otherwise keep `iter-15-patch-priority` as the scheduled best.

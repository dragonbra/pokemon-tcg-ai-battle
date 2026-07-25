# V7 AutoIter iter-34 Routed Abra Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 `deck.csv`、不放宽规则 gate 的前提下，修正 Poffin 将“无可见进化路线的带能量 Abra”误判为 ready attacker 的策略遗漏。

**Architecture:** 保留生产层对“可直接攻击”与“可见 handoff”的区分，只在 Poffin 的 deck effect 选择中使用已有的 `_has_visible_bench_handoff_route()` 语义。通过一个最小 fixture 验证：有两个带 Psychic 但无 Kadabra/Rare Candy 路线的 Bench Abra 时，Poffin 仍优先找 Abra；有可见路线时，原有 Dunsparce + Enriching draw route 不回归。

**Tech Stack:** Python 3.11、unittest、现有 Alakazam V7 AutoIter runtime、JSON trace analyzer。

## Global Constraints

- 固定 `submission/alakazam_v7_auto_iter/deck.csv`，不得修改卡组。
- 遵守宝可梦进化时序、Item Lock、Supporter/手动填能量次数和牌库保护规则。
- 不把孤立 Stage 1/Stage 2、Dunsparce 或无进化证据的 Abra 当作 ready handoff。
- 不修改终局攻击、第二回合已有 Abra 线的 Powerful Hand 例外。
- 每轮保留 `decision.md`、`advisor.md`、`analysis.md`、`metrics.json` 和 `ITER_PROCESS.md` 摘要；完整逐局 trace 只保留在外部评测目录的最新批次。

### Task 1: Add the failing Poffin regression fixture

**Files:**
- Modify: `tests/test_alakazam_v7_auto_iter_strategy.py` near the existing Poffin effect tests.

**Interfaces:**
- Consumes: `base_obs()`, `player()`, `pokemon()`, `MODULE._select_effect()`.
- Produces: a regression test proving that two energized but unrouted Bench Abra do not authorize the Poffin Dunsparce branch.

- [ ] **Step 1: Write the failing test**

  Build `Active Alakazam + Psychic`, Bench two Abra each with Psychic and no `Kadabra`, `Alakazam + Rare Candy`, or `Telepath` route in hand, and hand `Poffin`. Supply a Poffin deck selection containing Abra and Dunsparce. Use own turn 7, an opponent that is not a terminal KO, and assert `_select_effect()` chooses the Abra option(s), not Dunsparce.

- [ ] **Step 2: Run the focused test**

  Run:

  ```bash
  python3 -m unittest tests.test_alakazam_v7_auto_iter_strategy.V7AutoIterStrategyTests.test_poffin_keeps_abra_when_energized_bench_abra_has_no_visible_evolution
  ```

  Expected result before the production fix: FAIL because the current `_ready_attack_line_count()` counts both unrouted Abra and selects Dunsparce.

### Task 2: Make the smallest production correction

**Files:**
- Modify: `submission/alakazam_v7_auto_iter/main.py:1930-1948`.
- Test: `tests/test_alakazam_v7_auto_iter_strategy.py`.

**Interfaces:**
- Consumes: `_has_visible_bench_handoff_route(current, player)` and the existing `_poffin_dunsparce_draw_route()` exception.
- Produces: Poffin effect selection that reserves Dunsparce only when the Abra-line handoff is visible and the explicit Enriching route is valid.

- [ ] **Step 1: Implement the narrow predicate change**

  Keep the existing explicit Enriching route first. In the ordinary Poffin branch, use the existing visible-handoff predicate as the readiness guard; do not globally redefine `_ready_attack_line_count()` because it is also used by Fezandipiti safety and other setup decisions.

  The resulting decision shape must be equivalent to:

  ```python
  explicit_dunsparce_route = _poffin_dunsparce_draw_route(current, player)
  visible_handoff = _has_visible_bench_handoff_route(current, player)
  wanted = (
      {DUNSPARCE}
      if explicit_dunsparce_route
      else {ABRA}
      if _attack_line_count(player) < 3 or not visible_handoff
      else {DUNSPARCE}
  )
  ```

  Preserve the existing option-count and legal-selection behavior.

- [ ] **Step 2: Run the focused test**

  Re-run the Task 1 command and expect PASS.

- [ ] **Step 3: Run the related Poffin and Bench suite**

  ```bash
  python3 -m unittest tests.test_alakazam_v7_auto_iter_strategy -v
  ```

  Expect all existing Poffin, Bench continuity, direct handoff, Item Lock, and second-turn Powerful Hand tests to pass.

### Task 3: Record iter-34 evidence without claiming an unverified win-rate gain

**Files:**
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-34/decision.md`.
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-34/advisor.md`.
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-34/analysis.md`.
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-34/metrics.json`.
- Modify: `submission/alakazam_v7_auto_iter/ITER_PROCESS.md`.

**Interfaces:**
- Consumes: iter-33 repeat metrics, the two advisor reports, focused unit-test output, and any evaluation output actually produced in this iteration.
- Produces: a transparent iteration record with baseline, change, evidence, acceptance status, and no fabricated full-matrix numbers.

- [ ] **Step 1: Record the advisor conclusion**

  Explain that the trace review found many analyzer false positives, while the code review isolated the Poffin effect mismatch; retain the exact fixture boundary and do not claim the 122 post-KO diagnostics are all fixed.

- [ ] **Step 2: Record before/after metrics**

  Keep iter-33 repeat as the reference: 104/65/1, 61.2% win rate, 61.3% Meta-weighted, 25.9% second-turn Powerful Hand, 68.9% post-KO zero-ready, 34.1% game-level breaks, zero action errors and zero empty-Bench Run Away Draw. If no new full evaluation is run, label these as the baseline and mark candidate outcome as `not_evaluated` rather than inventing a change.

- [ ] **Step 3: Update the process log**

  Add the concise iter-34 change, fixture result, deck unchanged assertion, and promotion state. Keep `BEST_STRATEGY.json` unchanged until a real evaluation satisfies the promotion gates.

### Task 4: Verification gate

- [ ] **Step 1: Run syntax and asset checks**

  ```bash
  python3 -m py_compile submission/alakazam_v7_auto_iter/main.py
  python3 scripts/check_assets.py
  ```

- [ ] **Step 2: Run the complete existing test suite**

  ```bash
  python3 -m unittest discover -s tests -p 'test_*.py'
  ```

- [ ] **Step 3: Verify the deck invariant**

  ```bash
  git diff -- submission/alakazam_v7_auto_iter/deck.csv
  ```

  Expected output: empty.

- [ ] **Step 4: Decide promotion from evidence**

  Do not promote the stable best from a fixture alone. If no full evaluation is available, record `observe`; if a full candidate evaluation is later run, compare it to iter-15 best and iter-33 reference using the existing error, Bench, second-turn, Meta, and win-rate gates.

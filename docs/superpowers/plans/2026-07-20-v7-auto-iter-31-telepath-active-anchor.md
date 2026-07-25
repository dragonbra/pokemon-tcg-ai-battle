# V7 AutoIter 31：Telepath Active 附能与 Bench 锚点 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改固定 `deck.csv` 的前提下，让 Active Abra-line 在同一回合可选 Basic Psychic 与 Telepath Psychic 时，优先利用 Telepath 的额外 Bench 搜索建立接班锚点。

**Architecture:** 只调整 V7 AutoIter 的手动 Energy 选择评分。Telepath 仍然必须附到实际 Psychic Pokémon；仅当存在 Bench 空位且当前 Active 未充能、同时当前场上没有已充能的 Abra-line 接班人时，Telepath Active 路线才严格优于 Basic Psychic Active 路线。没有 Bench 空位、已有 ready handoff、Item Lock 或没有 Telepath option 时保持原有排序。

**Tech Stack:** Python 3.11+, `unittest`, 官方 evaluator trace JSON，固定 `submission/alakazam_v7_auto_iter/deck.csv`。

## Global Constraints

- `submission/alakazam_v7_auto_iter/deck.csv` 必须保持原样且始终为 60 张。
- agent 只能返回 evaluator 当前提供的合法 option index，并保持确定性。
- 当前工作区已有改动属于用户或前序迭代，不得 reset、checkout 或覆盖无关改动。
- 本轮不执行 `git commit`、`git push` 或 Kaggle 正式提交，除非用户另行明确要求。
- 完整 trace 写入外部评测仓库或 `/tmp`；当前 repo 只新增轻量 decision/metrics/advisor 文档。

## Evidence and behavioral hypothesis

- Replay: `/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-27-boss-before-second-turn-attack-20260720/kiyotah_dragapult/game_002.json`, shared turn 7.
- State: Active Kadabra is uncharged; Bench contains uncharged Alakazam, Shaymin and Dunsparce; hand contains Basic Psychic (id 5) and Telepath Psychic (id 19); Bench has one open slot; no ready Abra-line handoff exists.
- Current behavior: `_main_action()` selects Basic Psychic to Active because both Active Basic and Active Telepath options score equally.
- Desired behavior: select Telepath Psychic to Active. Its printed effect can search up to two Basic Psychic Pokémon onto Bench while it also provides the Psychic Energy needed for the Active Kadabra route. This does not skip the next evolution/attack action.

### Task 1: Add a red regression fixture

**Files:**
- Modify: `tests/test_alakazam_v7_auto_iter_strategy.py`

- [ ] **Step 1: Add the failing test**

Add a test with Active Kadabra lacking Energy, Bench containing an uncharged Alakazam and one open slot, hand containing both Basic Psychic and Telepath, and main options containing both Active attachments plus attack/end. Assert that `_main_action()` returns the Active Telepath option index.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v7_auto_iter_strategy.V7AutoIterStrategyTests.test_active_telepath_prepares_bench_anchor_when_basic_is_also_available
```

Expected: failure because the current stable option order chooses Active Basic Psychic.

### Task 2: Implement the smallest scoring change

**Files:**
- Modify: `submission/alakazam_v7_auto_iter/main.py: _main_action()` Energy option scoring

- [ ] **Step 1: Implement the minimal rule**

In the `option_type == 8` branch, give an Active Telepath attachment a strictly better score than an Active Basic Psychic attachment only when all of these are true:

```python
energy_id == TELEPATH_ENERGY
target_area == 4
target_id in ATTACK_LINE
not _has_psychic_energy(target)
bench_space > 0
not _has_visible_bench_handoff_route(current, player)
```

Keep the existing Bench-target Telepath behavior and all Energy/Item/terminal guards unchanged. Do not infer hidden deck contents; the benefit is an available legal Telepath attachment plus an open Bench slot, while the evaluator remains authoritative about the later search options.

- [ ] **Step 2: Run the focused test and verify it passes**

Run the focused unittest from Task 1. Expected: PASS.

- [ ] **Step 3: Run neighboring Energy and handoff regressions**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v7_auto_iter_strategy
```

Expected: all tests pass, including the existing cases that prefer a hand Abra over redundant Active Telepath and that prepare a Bench target from an already-ready Active attacker.

### Task 3: Verify and record the iteration

**Files:**
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-31/decision.md`
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-31/metrics.json`
- Modify: `submission/alakazam_v7_auto_iter/ITER_PROCESS.md`
- Create if advisor returns: `docs/history/kaggle/alakazam-v7-auto-iter/iter-30/advisor.md`

- [ ] **Step 1: Run repository checks**

```bash
python3 scripts/check_assets.py
python3 -m compileall -q scripts submission
```

Expected: exit code 0 and unchanged 60-card deck.

- [ ] **Step 2: Run focused evaluator traces**

Run the smallest useful matchup focus against `kiyotah_dragapult` and `sue_alakazam`, writing full traces under `/tmp/alakazam-v7-auto-iter-31-focus/`. Record wins/losses/draws, Meta weighted win rate, second-turn Powerful Hand, post-KO ready-attacker failure, empty-Bench Run Away Draw and action errors.

- [ ] **Step 3: Add a concise decision record**

Document the exact replay evidence, old and new behavior, test output, focused metrics, whether the target case improved, and `accept`/`observe`/`reject`. Do not claim a global improvement from a small independent focused sample.

- [ ] **Step 4: Update the process ledger**

Append one iteration entry containing the prior baseline, the minimal change, metrics before/after, and promotion decision. Keep `BEST_STRATEGY.json` pointing at iter-15 unless a full guardrail comparison proves promotion.

## Verification checklist

- [ ] The new fixture fails before the production change and passes afterward.
- [ ] Full V7 strategy tests pass.
- [ ] `scripts/check_assets.py` passes and `deck.csv` hash/content is unchanged.
- [ ] The historical turn-7 observation now selects Active Telepath, while the existing Active-Alakazam/hand-Abra fixture still selects the hand Abra route.
- [ ] Focused results are recorded as evidence, with no unsupported claim of full-matrix improvement.

# V7 AutoIter Iter-16 Bench 接力与第二回合启动 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改固定 `deck.csv` 的前提下，修复 Telepath 贴给已充能 Active 的错误优先级，并提高可观察资源确实支持时的 Bench 接力与第二回合 Powerful Hand 准备率。

**Architecture:** 保留现有 V7 AutoIter 的统一 option scoring。把 Bench anchor 判断收紧为真实指向 Bench 的 Abra 线或可放下的 Basic；Active 已有 Psychic 时的重复附能不得冒充接力资源。第二回合只复用已有的真实手牌、进化合法性和手填能量约束，不添加基于结果指标的盲目硬编码。

**Tech Stack:** Python 3.11+, unittest, 本地 `scripts/check_assets.py`，AutoIter trace analyzer。

## Global Constraints

- 固定 `submission/alakazam_v7_auto_iter/deck.csv`，本轮不得修改卡组。
- 攻击仍是回合终止动作；不得在攻击之后安排 Bench 准备。
- 一回合只能手填一次能量；只把实际可用的合法 option 纳入判断。
- 非终局攻击前优先保证至少存在可解释的 Bench 接力；终局奖赏闭环仍可优先攻击。
- 评测结果必须区分同批对照、独立随机批次和 trace 覆盖，不把缺失 trace 当作失败。
- 磁盘剩余空间低于 8 GiB 时，删除最早 iteration 的原始 trace JSON，保留 summary、metrics、decision、advisor 和最近失败 case。

### Task 1: 修复 Bench anchor 与重复 Active 附能的优先级

**Files:**
- Modify: `submission/alakazam_v7_auto_iter/main.py`，`_bench_insurance_due` 与 `_main_action` 的 Energy option scoring
- Test: `tests/test_alakazam_v7_auto_iter_strategy.py`

**Interfaces:**
- Consumes: 当前 observation 的 `inPlayArea`、`inPlayIndex`、真实 hand/field state
- Produces: `_bench_insurance_due(...)` 只在可行 Bench anchor 存在时成立；`_main_action(...)` 在 Kiyotah case 首选放下 Abra，不首选给已充能 Active Alakazam 重复贴 Telepath

- [ ] **Step 1: 写失败测试**

  构造 Active Alakazam 已有 Psychic、Bench 只有 Dunsparce、手牌同时有 Telepath Energy 与 Abra 的 observation；同时提供 Telepath→Active、Telepath→Dunsparce、Basic→Dunsparce 和放下 Abra 的合法 options。断言首步不是 Telepath→Active，而是直接放下 Abra（后续 observation 再完成附能）。

- [ ] **Step 2: 运行测试确认失败**

  Run: `python3 -m unittest -v tests.test_alakazam_v7_auto_iter_strategy.V7AutoIterStrategyTests.test_bench_anchor_does_not_attach_telepath_to_ready_active_first`

  Expected: 当前实现选择 Telepath→Active，断言失败；失败原因必须是策略排序而不是 fixture 加载错误。

- [ ] **Step 3: 最小实现**

  在 `bench_insurance_due` 的 anchor 检查中要求 Energy option 的 `inPlayArea == 5` 且目标属于 Abra 线；在 `_main_action` 的 Telepath 分支中，仅对 Bench Abra/Kadabra/Alakazam 返回 Bench insurance 的负分。Active Alakazam、Active Kadabra 或 Bench Dunsparce 的 Telepath option 回到普通资源排序。

- [ ] **Step 4: 运行测试确认通过**

  Run: `python3 -m unittest -v tests.test_alakazam_v7_auto_iter_strategy.V7AutoIterStrategyTests.test_bench_anchor_does_not_attach_telepath_to_ready_active_first tests.test_alakazam_v7_auto_iter_strategy.V7AutoIterStrategyTests.test_telepath_energy_builds_bench_before_nonterminal_knockout`

  Expected: 两个测试 PASS。

- [ ] **Step 5: 全量验证**

  Run: `python3 -m unittest discover -s tests -p 'test_*.py' && python3 scripts/check_assets.py && python3 -m compileall -q scripts submission && git diff --check`

  Expected: 142 个以上测试通过，资产检查、编译和 diff 检查均返回 0。

### Task 2: 记录 advisor、指标和评测结果

**Files:**
- Create: `docs/reports/kaggle/alakazam-v7-auto-iter/iter-16/advisor.md`
- Create: `docs/reports/kaggle/alakazam-v7-auto-iter/iter-16/decision.md`
- Create: `docs/reports/kaggle/alakazam-v7-auto-iter/iter-16/metrics.json`
- Modify: `submission/alakazam_v7_auto_iter/ITER_PROCESS.md`
- Modify: `submission/alakazam_v7_auto_iter/STRATEGY.md`

**Interfaces:**
- Consumes: iter-15 trace case、advisor 对卡牌/规则的核验、iter-16 本地评测 summary
- Produces: 可复盘的 control→candidate 动作差异、胜率/Meta/第二回合 Powerful Hand/接力断档/action error 指标及保留或回退结论

- [ ] **Step 1: 写 advisor**：说明 Poffin、Abra、Telepath Energy 的合法作用，明确 Telepath→Active 不建立 Bench。
- [ ] **Step 2: 写 decision/metrics**：注明前版本、改动、样本、trace 覆盖、指标变化和独立随机批次限制。
- [ ] **Step 3: 更新 ITER_PROCESS/STRATEGY**：将 iter-16 作为单变量实验记录，不把第二回合比例当成唯一目标。

### Task 3: 运行受控评测与磁盘清理

**Files:**
- Evaluate through existing evaluator; write raw outputs under `/tmp` or the configured report directory.

- [ ] **Step 1:** 先运行 focused Kiyotah/Penguin trace，确认新策略首步不再把 Telepath 贴给已充能 Active。
- [ ] **Step 2:** 磁盘低于 8 GiB 时删除最早 iteration 原始 JSON，保留其 summary-level artifacts。
- [ ] **Step 3:** 运行与当前评测框架一致的 17×10 批次；独立随机时只作方向性 guardrail，并分析具体失败 case。
- [ ] **Step 4:** 若胜率或接力显著回退，保留 case 修复但回退候选控制逻辑；若 guardrail 可接受，记录为 observe，继续下一轮。

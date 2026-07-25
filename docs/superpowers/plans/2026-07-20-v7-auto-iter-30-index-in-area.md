# V7 AutoIter Iter-30 Index-in-Area Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复归一化 option 中 `index` 与 `indexInArea` 混用导致的手牌 Energy 解析错误，并用 focused trace 验证 Dragapult/Sue case。

**Architecture:** 保持 `deck.csv` 和策略边界不变，只让 `submission/alakazam_v7_auto_iter/main.py` 在 `type=7`/手牌区域解析时使用原始手牌位置 `indexInArea`。测试 fixture 必须分别覆盖归一化手牌 option、孤立手牌资源不应阻止攻击，以及现有 V7 接力规则。

**Tech Stack:** Python 3.11+, `unittest`, AutoIter evaluator CLI, JSON full trace。

## Global Constraints

- 固定 `submission/alakazam_v7_auto_iter/deck.csv`，本轮不得修改卡组。
- 只返回 simulator 提供的合法 option index；策略必须保持确定性。
- 先运行针对性失败测试，再修改生产代码；完整 trace 放在隔壁评测仓库，不提交本 repo。
- 不能因为 focused case 改善就自动晋升 BEST；必须比较当前 iter-15 best 的主结果 guardrail。

---

### Task 1: 清理并固定 iter-30 回归 fixture

**Files:**
- Modify: `tests/test_alakazam_v7_auto_iter_strategy.py`

**Interfaces:**
- Consumes: `MODULE._option_card_id`, `MODULE._main_action`, `base_obs`。
- Produces: 两个最小、独立的回归测试：归一化手牌 option 使用 `indexInArea`；只有孤立手牌资源时仍选择 Powerful Hand。

- [ ] **Step 1: 删除误插入到 Enriching Energy 测试中的 parser 断言**

保留该测试自身的合法 option 与主动作断言，不在其中断言归一化 parser。

- [ ] **Step 2: 将孤立资源测试限制为实际可见的攻击/结束选项**

`test_empty_bench_with_isolated_resources_keeps_powerful_hand` 使用：

```python
options = [
    {"index": 0, "type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
    {"index": 1, "type": 14},
]
```

这样测试只验证手牌中的潜在资源不会被错误当作当前已暴露的 Bench route。

- [ ] **Step 3: 运行测试确认回归行为可复核**

Run: `python3 -m unittest tests.test_alakazam_v7_auto_iter_strategy -v`

Expected: `48 tests` 全部通过；若失败，先修正 fixture 或生产解析逻辑，不扩大本轮策略范围。

### Task 2: 验证完整单元测试与资产不变量

**Files:**
- Read: `submission/alakazam_v7_auto_iter/deck.csv`
- Read: `submission/alakazam_v7_auto_iter/main.py`

- [ ] **Step 1: 运行策略和 AutoIter 测试**

Run: `python3 -m unittest discover -s tests -p 'test*.py' -v`

- [ ] **Step 2: 运行语法与固定卡组检查**

Run: `python3 -m py_compile submission/alakazam_v7_auto_iter/main.py`

Run: `python3 scripts/check_assets.py`

Expected: 测试、编译和资产校验均成功；`deck.csv` 仍为 60 张且未被修改。

### Task 3: 记录 iter-30 的 advisor、决策和指标状态

**Files:**
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-30/decision.md`
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-30/advisor.md`
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-30/metrics.json`
- Modify: `submission/alakazam_v7_auto_iter/ITER_PROCESS.md`

- [ ] **Step 1: 记录本轮假设和卡牌/规则核验**

明确：`index` 是归一化 option 的局部位置，`indexInArea` 是原始手牌位置；不因手牌中有 Telepath/Basic Psychic 就禁止攻击，只有 simulator 暴露实际合法 attachment option 才进入接力 gate。

- [ ] **Step 2: 记录 focused 评测前后数值和证据来源**

若 focused trace 只覆盖两名对手，指标必须标注为 focused，不得冒充 170 局完整矩阵；不可用指标写 `null` 或“不可用”，不填 0。

- [ ] **Step 3: 明确采纳状态**

只有完整矩阵不低于 iter-15 best 的 guardrail 且核心 case 被验证，才可以写 `accepted`；否则写 `observe`，`BEST_STRATEGY.json` 保持 iter-15。

### Task 4: 运行 focused full-trace 并分析具体 case

**Files:**
- Create outside repo: `/tmp/alakazam-v7-auto-iter-30-focus/`
- Read: `/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/`

- [ ] **Step 1: 运行 Dragapult/Sue focused evaluator**

```bash
python3 scripts/alakazam_auto_iter.py run \
  --evaluator-root /Users/hejinyu/Documents/repos/ptcg-agent-kaggle \
  --agent submission/alakazam_v7_auto_iter/main.py \
  --cg-path submission/alakazam_v7_auto_iter \
  --label iter-30-index-in-area-handoff \
  --opponents kiyotah_dragapult,sue_alakazam \
  --games 10 \
  --output-dir /tmp/alakazam-v7-auto-iter-30-focus
```

- [ ] **Step 2: 读取 summary/trace，区分三种结论**

记录是：解析错误消失；simulator 未暴露合法 attachment option；或仍有其他策略 gate。每一种结论都附具体 game/step，不以胜率单独推断。

- [ ] **Step 3: 仅在 focused 证据支持时决定是否运行完整 17×10**

完整评测使用外部 evaluator 的 trace 目录；运行前检查空间，旧 trace 只在关键 case 已抽取后清理。

### Task 5: 完成验证报告

- [ ] **Step 1: 对照 iter-15 best**

比较胜率、Meta 加权胜率、第二回合 Powerful Hand、post-KO ready attacker、空 Bench Run Away Draw 和我方 action error。

- [ ] **Step 2: 保持最优版本边界**

若候选没有证据证明全面改善，保留 `BEST_STRATEGY.json` 指向 iter-15，并在 `decision.md` 写明 observe。

- [ ] **Step 3: 报告未完成项**

说明 focused 与 full-matrix 的覆盖差异、trace 保存位置，以及下一轮最小变量建议。

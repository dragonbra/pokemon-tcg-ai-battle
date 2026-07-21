# Alakazam V8 Semantic Recovery Implementation Plan

> **状态：已终止。** 2026-07-21 根据用户决定停止在模块化重构框架上继续恢复语义；
> 本文件只保留为失败尝试的过程记录，不得据此继续运行后续任务。活动候选改回
> `submission/alakazam_v8` 的重构前实现，未来策略迭代从该版本重新开始。

> **For agentic workers:** 在当前会话内逐任务执行；禁止 AutoIteration 生成候选，禁止子代理并行修改，禁止未经用户要求 commit。

**Goal:** 在保留 `work/alakazam_v8_current` 模块化框架的前提下，使胜率和第二回合 Powerful Hand 至少恢复到 `submission/alakazam_v8` 的同口径水平。

**Architecture:** 把历史实现当作只读行为 oracle，沿 `TurnFacts -> RouteAnalysis -> TurnPlan -> ActionIntent -> SemanticOption` 数据流逐项恢复语义。单元测试证明具体决策，原生 17×10 报告验证整体结果；生产代码不导入历史 submission。

**Tech Stack:** Python 3.11、标准库 `unittest`、仓库原生 `evaluation` CLI、官方 `cg` runtime。

## Global Constraints

- 只修改 `work/alakazam_v8_current`、相关 V8 测试、`work/docs/FIX.md` 和本实验目录。
- 不修改 `evaluation/`、`engine/source/`、opponent package、卡表或历史 submission。
- 每个行为修复严格执行 RED -> GREEN -> 回归验证。
- 每次 17×10 的完整原生报告直接写入本目录，不保存额外批量 trace。
- 核心验收线为胜场 `>=112/170`、第二回合 Powerful Hand `>=35/170`、candidate error `=0`。
- 不执行 git commit、git push 或 Kaggle submission。

---

### Task 1: 建立基线和真实 observation 差分入口

**Files:**
- Modify: `work/docs/FIX.md`
- Create: `tests/test_alakazam_v8_semantic_parity.py`
- Create: `work/auto-iteration/history_iterations/v8-fix-sol-refactor-diff/baseline-2026-07-21.md`

**Interfaces:**
- Consumes: `submission/alakazam_v8.main.agent(obs)` 与 `work/alakazam_v8_current.main.agent(obs)` 的公开 agent 契约。
- Produces: 可对同一 fixture observation 运行两套 agent、重置模块状态并显示首个动作分歧的测试帮助函数。

- [x] 写入两轮 17×10 基线的 run id、总体/先后手/逐对手结果和核心指标。
- [x] 在 parity 测试中通过独立模块名加载历史与当前 agent，避免共享模块级状态。
- [ ] 用一个已知等价 fixture 证明 harness 能比较合法 option index。
- [x] 用第二回合缺失路线 fixture 观察 RED，并把差异归到 facts、route、plan、intent 或 effect 中的一层。
- [x] 运行 `python3 -m unittest -v tests.test_alakazam_v8_semantic_parity`。

### Task 2: 恢复第二回合 Powerful Hand 启动闭环

**Files:**
- Modify: `tests/test_alakazam_v8_semantic_parity.py`
- Modify: `tests/test_alakazam_v8_agent.py`
- Modify: `work/alakazam_v8_current/strategy/planner.py`
- Modify: `work/alakazam_v8_current/strategy/policies/setup.py`
- Modify: `work/alakazam_v8_current/strategy/policies/resources.py`
- Modify when evidence requires: `work/alakazam_v8_current/strategy/effects/search.py`

**Interfaces:**
- Consumes: `TurnFacts`, `RouteAnalysis`, `TurnPlan` 和已解码的 `SemanticOption`。
- Produces: 对 Active Abra/Kadabra 的合法进化、当前攻击者附 Psychic、必要检索与 `attackId=1072` 提交顺序。

- [x] 从真实二回合失败 evidence 写一个仅覆盖首个断点的失败测试。
- [x] 运行该测试并确认历史 agent 选择期望动作、当前 agent 选择不同动作。
- [x] 只在产生错误过滤的 planner/policy 层恢复旧版路线条件。
- [x] 运行目标测试和四个 V8 测试模块，确认 GREEN 且无回归。
- [ ] 对新 evidence 重复上述 RED/GREEN 循环，直到已观察的二回合断点不再出现。

### Task 3: 恢复非攻击 Active 的 handoff

**Files:**
- Modify: `tests/test_alakazam_v8_semantic_parity.py`
- Modify: `tests/test_alakazam_v8_agent.py`
- Modify: `tests/test_alakazam_v8_planner.py`
- Modify: `work/alakazam_v8_current/strategy/routes.py`
- Modify: `work/alakazam_v8_current/strategy/planner.py`
- Modify: `work/alakazam_v8_current/strategy/policies/continuity.py`

**Interfaces:**
- Produces: 只有 Bench 存在本回合可攻击 Alakazam、当前 Active 不应继续攻击且 retreat budget 可用时生成 `ActionKind.RETREAT` intent。

- [ ] 写 Active Shaymin/Fezandipiti ex + ready Bench Alakazam 的 Retreat RED 测试。
- [ ] 写 Active Dunsparce 必须区分 Dudunsparce Ability 与普通 Retreat 的 RED 测试。
- [ ] 写换位后仍不能攻击时不 Retreat、永不提交 Trading Places 的保护测试。
- [ ] 在 route/plan 中表达 confirmed retreat handoff，并在 continuity 中生成精确 target intent。
- [ ] 运行目标测试和四个 V8 测试模块，确认全部 GREEN。

### Task 4: 恢复 Supporter、回收和效果目标组合语义

**Files:**
- Modify: `tests/test_alakazam_v8_semantic_parity.py`
- Modify: `tests/test_alakazam_v8_agent.py`
- Modify: `tests/test_alakazam_v8_effects.py`
- Modify: `work/alakazam_v8_current/strategy/planner.py`
- Modify: `work/alakazam_v8_current/strategy/policies/resources.py`
- Modify: `work/alakazam_v8_current/strategy/policies/control.py`
- Modify: `work/alakazam_v8_current/strategy/effects/recovery.py`
- Modify: `work/alakazam_v8_current/strategy/effects/search.py`
- Modify: `work/alakazam_v8_current/strategy/effects/dispatcher.py`

**Interfaces:**
- Produces: Boss KO、Hilda/Dawn 建线、Lana/Night Stretcher/Sacred Ash 恢复和 Xerosic 干扰之间可解释且互斥的优先级。

- [ ] 分别为 Boss、Xerosic、Lana、Night Stretcher、Sacred Ash 写单一条件 RED 测试。
- [ ] 对每个测试确认历史 agent/effect selector 的期望选择。
- [ ] 用具体 route predicate 替代会过早过滤合法动作的单一 `supporter_purpose` 分支。
- [ ] 确保 Supporter budget、Item Lock、deck budget 和 terminal victory gate 仍是硬约束。
- [ ] 运行 effects、agent、planner、facts 和 parity 测试。

### Task 5: 运行正式矩阵并按 evidence 修正

**Files:**
- Create: `work/auto-iteration/history_iterations/v8-fix-sol-refactor-diff/iteration-NNN/run-*/`
- Modify: `work/docs/FIX.md`

**Interfaces:**
- Consumes: revision 2 的 `summary.json`、`metrics.json`、`games.jsonl` 和 `report.html`。
- Produces: 每轮完整可审计报告，以及 FIX.md 中对应的已修复/未解决记录。

- [ ] 运行 package validate 和全部 V8 单元测试。
- [ ] 运行 17 个对手各 10 局，使用 `--no-visualize`，并把 output 指向对应的
      `iteration-NNN/` 目录。
- [ ] 记录总体、实际先后手、逐对手、Powerful Hand、过牌、Post-KO、攻击质量和错误分类。
- [ ] 若未达门槛，只从报告 evidence 提出一个新根因假设，返回对应任务先写 RED 测试。
- [ ] 达到门槛后再运行一轮新鲜 17×10，避免用单次随机高点宣称恢复完成。

### Task 6: 完整验证与交付

**Files:**
- Modify: `work/docs/FIX.md`

- [ ] 运行 `python3 -m unittest -v tests.test_alakazam_v8_agent tests.test_alakazam_v8_effects tests.test_alakazam_v8_facts tests.test_alakazam_v8_planner tests.test_alakazam_v8_semantic_parity`。
- [ ] 运行 `python3 scripts/check_assets.py`。
- [ ] 运行 `python3 -m compileall -q work/alakazam_v8_current evaluation scripts tests`。
- [ ] 检查 `git diff --check` 和 `git status --short`，确认未覆盖 handoff 的用户修改。
- [ ] 在 FIX.md 写明最终指标、已修复语义和仍未解决问题，不执行 commit。

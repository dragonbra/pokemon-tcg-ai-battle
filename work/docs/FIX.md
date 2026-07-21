# Alakazam V8 重构语义修复

## 1. 任务边界

本轮只修改 `work/alakazam_v8_current`，目标是在保留当前模块化策略框架的前提下，
恢复 `submission/alakazam_v8/main.py` 的可观察策略语义。不采用 AutoIteration 自动生成
候选，不修改 `evaluation/`、`engine/source/`、17 个 opponent package 或历史 submission。

评测使用仓库原生入口和 `auto_iteration_v8_setup_relay` revision 2：

```bash
python3 -m evaluation validate work/alakazam_v8_current
python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output work/auto-iteration/history_iterations/v8-fix-sol-refactor-diff/iteration-NNN
```

核心验收指标是胜率和第二回合实际提交 `attackId=1072` 的比例。由于官方引擎内部使用
`random_device` 且当前接口不能注入 seed，不把两轮矩阵解释成逐局配对实验；单元测试负责
证明具体语义，17×10 矩阵负责确认整体性能没有下降。

## 2. 已确认的未对齐语义

### 2.1 整体结果和第二回合启动能力大幅下降

同一卡表、同一 `cg` runtime、同一 17×10 原生评测中：

| 候选 | 胜 / 负 / 平 | 胜率 | 第二回合 Powerful Hand |
| --- | ---: | ---: | ---: |
| `submission/alakazam_v8` | 112 / 55 / 0，另有 3 个外部引擎错误 | 65.9% | 35 / 170（20.6%） |
| `work/alakazam_v8_current` | 83 / 86 / 0，另有 1 个外部引擎错误 | 48.8% | 2 / 170（1.2%） |

两者 deck hash 和 `cg` tree hash 完全一致。当前 52 个 V8 fixture 测试全部通过，说明现有
测试覆盖的是少量局部规则，不足以约束旧版完整的动作排序和跨步骤路线。

### 2.2 Retreat 只有解码和路线识别，没有策略提交

当前框架能够把 option type 12 解码为 `ActionKind.RETREAT`，`routes.py` 也能把 Bench 已
就绪 Alakazam 识别为 handoff，但所有 policy 都没有生成 `RETREAT` intent。因此旧版
`_retreat_improves_attack` 所表达的“非攻击 Active 让位给本回合可攻击 Alakazam”路线
无法真正穿过 orchestrator。

### 2.3 非攻击 Active 的接力语义只恢复了一部分

当前 continuity 主要覆盖 Active Dunsparce/Dudunsparce 及 post-KO Fezandipiti，尚未完整
覆盖旧版中 Shaymin、Fezandipiti ex 等临时 Active 的 Bench 建设、补牌、附能和 Retreat
交接。handoff 的存在目前也没有统一进入 `TurnPlan.must_goals` 和攻击闸门。

### 2.4 资源动作由粗粒度 purpose 过滤，丢失旧版的组合条件

当前 planner 每回合只产生一个 `supporter_purpose`，resources policy 再用 `elif` 只尝试
对应 Supporter。旧版则同时比较 Boss KO、Hilda 进化与能量路线、Dawn 进化链、Lana/Night
Stretcher 恢复、Xerosic 干扰及当前攻击路线。重构后的粗粒度 purpose 会让合法且必要的
组合条件在到达 option 匹配前就被过滤。

### 2.5 Item 和效果选择缺少与主行动相同的路线证明

当前 resources 会在未锁 Item 时依次提出 Rare Candy、Night Stretcher 和 Sacred Ash，
但没有先证明对应路线缺口；effect selector 又使用新的简化优先级。旧版主动作条件与效果
目标选择是一组配套语义，拆开后可能出现过早消耗、拿错阶段或没有完成当回合攻击路线。

### 2.6 单元测试没有把历史实现作为行为 oracle

现有 fixture 主要验证“某动作会发生”，没有系统检查同一 observation 下旧版与重构版的
首选动作差异，也没有用真实 trace observation 覆盖第二回合失败样本。这使局部测试全部
通过时，真实 Powerful Hand 仍可从 35 次下降到 2 次。

## 3. 修复方法

1. 先从真实 trace 和旧版实现提取单一语义差异，每次只提出一个根因假设。
2. 对每个差异先增加失败测试，并确认失败来自缺失语义而非 fixture 错误。
3. 在现有 `TurnFacts -> RouteAnalysis -> TurnPlan -> ActionIntent -> SemanticOption` 数据流中
   修复根因，不让生产代码导入或委托历史 `submission/alakazam_v8`。
4. 优先修复第二回合 Abra 进化、能量和检索闭环；然后修复非攻击 Active 的 handoff；最后
   恢复 Supporter、回收、控制和攻击前准备排序。
5. 每个小修复运行对应 V8 测试；形成完整语义批次后运行 17×10，并把整套原生报告写入
   `work/auto-iteration/history_iterations/v8-fix-sol-refactor-diff/iteration-NNN/<run_id>/`。
6. 只有 correctness 没有 candidate error，且胜率和第二回合 Powerful Hand 同时达到旧版
   同口径水平，才把本轮标记为完成。

## 4. 验收线

- 参考实现：`submission/alakazam_v8/main.py`。
- 正确性：candidate 非法动作、load error、runtime error 为 0；已知 opponent/engine error
  单独记录，不伪装成 candidate loss。
- 胜率：新鲜 17×10 中至少达到本轮旧版基线 `112/170`。
- 第二回合 Powerful Hand：新鲜 17×10 中至少达到本轮旧版基线 `35/170`。
- 辅助审计：同时观察先后手、额外过牌、Post-KO 成功率、空 Bench Run Away Draw、攻击
  质量和零分母状态，但不以这些辅助项替换两个核心指标。

## 5. 当前未解决问题

- 第二回合 Powerful Hand 的单轮最好结果为 `9/170`，但最后一轮只有 `6/170`；始终远低于
  历史 V8 的 `35/170`，恢复尝试已经终止。
- Retreat intent 尚未实现，Shaymin/Fezandipiti ex 临时 Active 路线尚未完整恢复。
- Supporter purpose、回收 Item 和 effect target 的旧版组合语义尚未逐项建立 parity 测试。
- 引擎不能固定 seed，单轮 170 局存在抽样波动；最终结果需要结合明确的语义测试解释。
- `yakitori_raging_bolt` 已知 opponent/engine error 仍可能影响完成对局数，必须在报告中
  与 candidate correctness 分开。

## 6. 修复记录

| 日期 | 状态 | 内容 | 验证 |
| --- | --- | --- | --- |
| 2026-07-21 | 基线 | 确认同卡表、同 runtime 下重构版胜率和第二回合启动能力下降 | 旧版与当前版各 17×10；V8 测试 52 项通过 |
| 2026-07-21 | 已修复 | Item Lock 只屏蔽 Item，不再误伤 Abra/Dunsparce 上场 | parity RED/GREEN；Iteration 001 为 85 胜、二回合 Powerful Hand 3 次 |
| 2026-07-21 | 已修复 | 带 Psychic 的 Active Abra 保留 Rare Candy 直达路线，先自然进化合法 Bench Abra | 两份真实 trace；parity RED/GREEN；V8 测试 54 项通过 |
| 2026-07-21 | 未达标 | Iteration 002 验证启动次数提高但胜率仍未恢复 | 82 胜；二回合 Powerful Hand 8/170；candidate error 0 |
| 2026-07-21 | 已修复 | `SETUP_BENCH_POKEMON` 即使可选也选择一个有用 Basic，不再统一返回空选择 | parity RED/GREEN；V8 测试 55 项通过 |
| 2026-07-21 | 未达标 | Iteration 003 验证开局 Bench 语义正确但主要性能仍未恢复 | 78 胜；二回合 Powerful Hand 9/170；2 个对手侧 engine error |
| 2026-07-21 | 已修复 | Fezandipiti ex/Shaymin 临时 Active 先附 Telepath，为后续撤退 handoff 准备费用 | 两份真实 trace；parity RED/GREEN；V8 测试 56 项通过 |
| 2026-07-21 | 未达标 | Iteration 004 未恢复该动作依赖的后续链路，核心指标下降 | 67 胜；二回合 Powerful Hand 5/170；candidate error 0 |
| 2026-07-21 | 已修复 | Poffin 建线优先于临时 Active 附能；首回合 Poké Pad 优先搜索 Dunsparce | 两项 parity RED/GREEN；V8 测试 58 项通过 |
| 2026-07-21 | 未达标 | Iteration 005 部分回升，但 Dunsparce bridge 和核心指标仍未恢复 | 75 胜；二回合 Powerful Hand 6/170；candidate error 0 |

## 7. 终止与回退决定

本次恢复尝试未达到继续投入的条件，停止在模块化重构框架上追加修复。五轮矩阵的结果为：

| 轮次 | 胜 / 负 / 平 | 第二回合 Powerful Hand | 结论 |
| --- | ---: | ---: | --- |
| Iteration 001 | 85 / 84 / 0，另有 1 个对手侧错误 | 3 / 170 | Item Lock 局部修复不足以恢复性能 |
| Iteration 002 | 82 / 88 / 0 | 8 / 170 | Rare Candy 路线改善启动次数，胜率未恢复 |
| Iteration 003 | 78 / 90 / 0，另有 2 个对手侧错误 | 9 / 170 | 本实验最好启动次数，仍远低于基线 |
| Iteration 004 | 67 / 103 / 0 | 5 / 170 | 配套语义不完整，整体进一步下降 |
| Iteration 005 | 75 / 95 / 0 | 6 / 170 | 部分回升，仍不具备可用性 |

历史 V8 的本轮参考线为 `112/170` 胜和 `35/170` 第二回合 Powerful Hand；所有恢复轮次
均明显未达标。活动候选因此回退到 `submission/alakazam_v8` 的重构前代码，未来 V8 改进
直接以该版本为基础，不再把本实验中的模块化策略当作工作基线。

为控制仓库体积，每轮只保留自包含 `report.html` 和本目录的过程 Markdown；完整 trace、
`metrics.json`、`games.jsonl`、`manifest.json`、`summary.json`、`cases.jsonl` 与 `report.md`
已经删除。如需重新审计原始数据，必须重新运行 evaluation。

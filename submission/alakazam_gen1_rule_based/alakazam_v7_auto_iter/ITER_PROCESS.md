# Alakazam V7 AutoIter 迭代过程

> 本文件记录 AutoIter 的每一轮“为什么改、改了什么、指标怎么变、是否保留”。
> 这里的 `AutoIter V1/V2/V3` 分别对应 `iter-01/iter-02/iter-03`，不要与卡组版本
> V1/V2/V3 混淆。

## 总目标

固定 `deck.csv`，只通过可解释、可回放的策略重构，让以下指标在最终评测对手池上
逐步改善：

1. 胜率和 Meta 加权胜率；
2. 第二回合实际使用 Alakazam 的 `Powerful Hand` 比例；
3. Active 被击倒后的 ready attacker 接力能力；
4. 进化链是否连续、是否出现资源断档；
5. 空 Bench 时错误使用 Dudunsparce `Run Away Draw` 的次数；
6. 我方 action error、异常结束和 deck-out 等正确性指标。

70%、80%、接近 90% 是努力方向，不是每一轮的硬性晋级线。局部 case 得到改善但
胜率连续下降，说明改动伤害了整体游戏计划，不能仅因为 case 通过就保留。

## 固定评测约定

- 卡组：`submission/alakazam_v7_auto_iter/deck.csv`，不换牌、不调整数量。
- 对手矩阵：当前 registry 的 17 个对手，每个对手 10 局；先后手按 evaluator 的
  swap 协议交替。
- control：最近一个已接受的策略版本。
- candidate：只针对一个主要 replay case 假设做的最小改动。
- discovery：保存完整 trace，用于第二回合、接力和具体 case 分析。
- trace 保留：完整 trace 只保留在隔壁评测仓库的最新一轮；本 repo 每轮至少保留
  `decision.md`、必要的 `advisor.md`/`metrics.json` 和精炼 case。清理旧 trace 前，先把
  关键状态与动作链写入本 repo，避免以“删除文件”代替实验记录。
- outcome guardrail：磁盘受限时可以先运行 summary-only 批次，但只能比较胜负、平局、
  异常数量；没有完整 trace 时，第二回合和接力指标必须记为“不可用”，不能填 0
  或推断为失败。
- 每轮必须记录随机性、样本数、trace 覆盖数和是否使用同一批随机样本。独立批次的
  结果只能作为噪声下的 guardrail，不能把单次胜率变化直接解释为策略因果。
- 每轮增加一个独立的 `strategy advisor` agent：只读 `deck.csv`、官方卡牌描述、规则
  和本轮失败 trace，先提出“哪张牌能解决、为什么合法、有哪些边界”；主 agent 必须
  核验建议后才能改策略。advisor 不直接修改生产策略，报告写入对应 iteration 的
  `advisor.md`，避免因不了解卡牌效果而把动作优先级写反。

## 指标记录格式

每轮至少记录：

| 指标 | control | candidate | 变化 | 口径说明 |
|---|---:|---:|---:|---|
| 胜 / 负 / 平 |  |  |  | 全部对局分母 |
| 胜率 |  |  |  | wins / games |
| Meta 加权胜率 |  |  |  | 对手权重加权 |
| 第二回合 Powerful Hand |  |  |  | 实际选择 `attackId=1072` |
| 击倒后无 ready attacker |  |  |  | 事件数 / 事件分母 |
| 打手断档对局率 |  |  |  | 至少一次断档的对局 / games |
| 空 Bench Run Away Draw |  |  |  | 目标硬错误，期望为 0 |
| 我方 action error |  |  |  | 与对手侧异常分开 |

每个 case 还要写明：行动前状态、合法 options、control 实际 action、candidate
实际 action、预期 action、后续行动链，以及 case 是 `pass`、`observe` 还是 `fail`。

## iter-00：V7 AutoIter 基线

### 基线来源

- 原始 V7 详细评测：17 个对手 × 10 局 = 170 局。
- `submission/alakazam_v7/EVAL_RESULT.md`：用于完整矩阵的 V7 参考值。
- `docs/reports/kaggle/alakazam-v7-auto-iter/iter-00-baseline/`：当前实际可读取的 74
  个完整 trace，用于验证分析器口径；不是完整 170 局的替代品。

### 已知基线

| 指标 | V7 170 局参考 | AutoIter 可读取的 74 trace |
|---|---:|---:|
| 胜率 | 106/170 = 62.4% | 32/74 = 43.2% |
| 第二回合 Powerful Hand | 46/170 = 27.1% | 12/74 = 16.2% |
| 击倒后无 ready attacker | 19/64 败局主因摘要 | 56/74 事件 |
| 空 Bench Run Away Draw | 4/64 败局主因摘要 | 2 次 |

这里的 74 trace 是覆盖限制下的诊断样本；不能把它与 170 局参考值当作同批 A/B
结果。V6 的第二回合指标仍没有可靠同口径数据，不在此处补估计值。

## AutoIter V1：iter-01 空 Bench Run Away Draw 修复

### 策略假设

当 Active 是 Dudunsparce 且 Bench 为空时，`Run Away Draw` 会把唯一的 Active 洗回
牌库，造成没有 Active 的可避免错误。该局面应优先选择合法的结束或其它可行选项，
不能把“能抽牌”当成“应该抽牌”。

### 实际改动

只改 `submission/alakazam_v7_auto_iter/main.py` 的策略与 option 解析：

- 增加空 Bench `Dudunsparce + Run Away Draw` 的明确拒绝优先级；
- 修正 field option 的 `indexInArea` 解析，使 evaluator 没有直接提供 `cardId` 时，
  仍能识别 Active Dudunsparce；
- 不修改 `deck.csv`，不调整进化、附能、Supporter 或牌库保护策略。

同时，AutoIter 分析器新增了与策略无关的测量修复：

- 第二回合只统计实际选中的 `attackId=1072`；
- 只有有伤害日志或零 HP 状态证据时才统计 Pokémon KO；
- 能从 `area/indexInArea` 解析 Run Away Draw case。

### Case 验收

| case | control | candidate | 结论 |
|---|---|---|---|
| Active Dudunsparce、Bench 为空、Run Away Draw 与 End 同时合法 | 选择 Run Away Draw | 选择 End | fixture 回归通过 |
| option 使用 `indexInArea` 指向 Active | 解析不到 Dudunsparce | 正确解析并拒绝 Run Away Draw | fixture 回归通过 |

### 结果记录

本轮先后运行了两类批次，不能混为一个严格的同随机 A/B：

| 批次 | 样本 | control | candidate | 说明 |
|---|---:|---:|---:|---|
| 完整矩阵 summary-only | 17×10 = 170 局 | 90/170 = 52.9% | 93/170 = 54.7% | 独立随机批次；只有 outcome 可用 |
| Meta 加权胜率 | 同上 | 52.2% | 54.7% | 仅作方向性 guardrail |
| 第二回合 / 接力 | 同上 | 不可用 | 不可用 | 未保存完整 trace，不能用 0 代替 |
| `yakitori_raging_bolt` 异常 | 各批次 3 次 | 3 | 3 | focused trace 显示异常步骤 role 为 opponent |

另有 focused trace 检查显示该 `IndexError` 出现在对手 role，不应作为我方 action
error 计入策略退化；完整矩阵 summary 没有携带归属信息，因此这里保留原始异常数并
明确标注来源限制。

### 当前决策：observe

fixture 层面 V1 已解决目标 case，但由于本轮完整矩阵没有保存 trace，尚未证明 170
局中的所有空 Bench case 都消失；同时 outcome 是独立随机批次，不能据此宣称胜率因果
提升。下一步应先在磁盘空间允许时，对相关 matchup 运行 focused full-trace，确认：

- `empty_bench_run_away_draw_count == 0`；
- 我方 action error 不增加；
- 第二回合 Powerful Hand 和 post-KO ready attacker 没有不可解释的回退。

在这些证据补齐前，V1 作为候选观察版本，不直接替换 control。

## AutoIter V2：Active Dunsparce 优先接 Enriching Energy

### 迭代假设

当 Active 是 Dunsparce、手牌同时有 Dudunsparce 与 Enriching Energy，且本回合可以
合法进化时，先把 Enriching Energy 贴给这条抽牌引擎。这样遵循已确认的卡组规则：
Enriching Energy 会补充 Dudunsparce 的过牌，过牌机会通常比把一张普通能量提前贴给
Abra 线更有价值。没有 Enriching Energy 时，原有策略仍优先为 Abra 线准备 Psychic
Energy。

### 实际改动

只改动 `submission/alakazam_v7_auto_iter/main.py` 的 `_enriching_draw_route()`：

- Active Dudunsparce 时允许 Enriching Energy 抽牌路线；
- Active Dunsparce 在 Dudunsparce 可进化且已在手牌时，允许先贴 Enriching Energy；
- 不修改 `deck.csv`，不修改 Supporter、Rare Candy 和攻击终止规则。

回归 fixture：
`tests/test_alakazam_v7_auto_iter_strategy.py::test_enriching_energy_precedes_bench_setup_for_active_dunsparce_route`，
要求同一 observation 下从“给 Bench Abra 进化 / 给 Active Dunsparce 贴 Enriching / 结束”
中选择贴 Enriching 的合法 option。

### Case 验收

| case | control | candidate | 结论 |
|---|---|---|---|
| Active Dunsparce、手牌有 Dudunsparce + Enriching Energy、Bench 有 Abra | 先处理其它 setup | 先给 Active Dunsparce 贴 Enriching Energy | fixture 通过；目标 case 改善 |
| Active Dudunsparce、Bench 为空 | 可能错误 Run Away Draw | candidate 仍禁止空 Bench Run Away Draw | repeat 中 6 → 0 |

### 结果：第一批独立完整 trace

评测为 17 个对手 × 10 局 = 170 局，control 使用 V7 基线，candidate 使用 V7
AutoIter 当前代码。两边不是同一发牌随机样本，只作方向性 guardrail。

| 指标 | control | candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 115 / 53 / 2 | 103 / 67 / 0 | 胜 -12 |
| 胜率 | 67.6% | 60.6% | -7.1pp |
| Meta 加权胜率 | 70.5% | 58.8% | -11.7pp |
| 第二回合 Powerful Hand | 44/170 (25.9%) | 51/170 (30.0%) | +4.1pp |
| post-KO 无 ready attacker | 103/156 (66.0%) | 122/166 (73.5%) | +7.5pp |
| 至少一场接力断档 | 57/170 (33.5%) | 64/170 (37.6%) | +4.1pp |
| 空 Bench Run Away Draw | 2 | 0 | -2 |
| 我方 action error | 2（归属限制） | 0 | -2 |

### 结果：独立 repeat 完整 trace

评测协议同上，但使用另一批独立随机发牌；原始 trace 在
`/tmp/alakazam-v7-auto-iter/iter-02-repeat/`，因此不能与第一批逐局配对。

| 指标 | control | candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 94 / 73 / 3 | 97 / 73 / 0 | 胜 +3 |
| 胜率 | 55.3% | 57.1% | +1.8pp |
| Meta 加权胜率 | 54.4% | 56.1% | +1.8pp |
| 第二回合 Powerful Hand | 38/170 (22.4%) | 46/170 (27.1%) | +4.7pp |
| post-KO 无 ready attacker | 110/153 (71.9%) | 129/193 (66.8%) | -5.1pp |
| 至少一场接力断档 | 61/170 (35.9%) | 67/170 (39.4%) | +3.5pp |
| 空 Bench Run Away Draw | 6 | 0 | -6 |
| 我方 action error | 3（归属限制） | 0 | -3 |

### 当前决策：observe，不晋级为稳定 control

目标 case 与空 Bench 硬错误都改善；repeat 中第二回合攻击、Meta 胜率和 post-KO
事件率也改善。但第一批完整结果出现明显的总体胜率、Meta 胜率和接力断档回退，第二
批虽方向相反，仍无法证明 Enriching 路线对整体接力有稳定因果收益。两批的
`games_with_post_ko_break_rate` 还同时上升（33.5%→37.6%、35.9%→39.4%），但该
指标受 candidate 对局中实际发生的 KO 事件数量影响，不能脱离事件率单独解释。

因此 V2 保留为“已验证目标 case、整体结果待观察”的实验记录；下一轮不叠加新的
变量，先针对 post-KO 断档和第二回合 missing trace 找到具体的可替代错误动作。若新
候选无法维持 V1/V2 已解决的空 Bench 硬错误或连续两批胜率下降，则回退到上一个版本。

## AutoIter V3：恢复路线不把孤立 Stage 1/Stage 2 当作 Basic 来源

### 迭代假设

弃牌区只有 Kadabra/Alakazam 时，不能把手牌中的 Stage 1/Stage 2 当作可直接恢复的
Basic 来源；恢复第一只打手仍需要 Abra，除非场上已经有 Abra 线宝可梦。该改动不改
卡组数量，只修正恢复路线的资源语义。

### 评测前后

前一版本：`iter-02`；本轮 candidate：`iter-03`。17 个对手 × 10 局，独立随机批次。

| 指标 | iter-02 control | V3 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 91 / 78 / 1 | 102 / 68 / 0 | 胜 +11 |
| 胜率 | 53.5% | 60.0% | +6.5pp |
| Meta 加权胜率 | 51.7% | 59.4% | +7.7pp |
| 第二回合 Powerful Hand | 25.3% | 25.3% | 持平 |
| post-KO 无 ready attacker | 72.9% | 70.6% | -2.3pp |
| 打手断档对局率 | 37.1% | 37.6% | +0.6pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |

### 决策

`observe`，并将 V3 作为下一轮 control。接力断档率基本持平，仍需具体 trace 分析。

## AutoIter V4：Active Alakazam 的可见接力线优先 Psychic 资源

### 迭代假设

Active Alakazam 已有 Psychic Energy，Bench 有可接力的无能量 Kadabra/Alakazam，且
当前 options 暴露 Psychic 附能或 Lana's Aid 回收 Basic Psychic 的路径时，先准备接
力资源；不先把唯一附能机会给 Dudunsparce 的 Enriching Energy，也不在 Lana's Aid
能够完成回收时直接攻击。

### 评测前后

前一版本：`iter-03`；本轮 candidate：`iter-04`。两批均为 17 个对手 × 10 局完整
trace，control/candidate 为独立随机样本。

| 批次 | 指标 | iter-03 control | V4 candidate | 变化 |
|---|---|---:|---:|---:|
| 第一批 | 胜率 | 64.1% | 62.4% | -1.8pp |
| 第一批 | Meta 加权胜率 | 63.7% | 62.3% | -1.4pp |
| 第一批 | 第二回合 Powerful Hand | 21.8% | 27.6% | +5.9pp |
| 第一批 | post-KO 无 ready attacker | 72.5% | 67.1% | -5.4pp |
| 第一批 | 接力准备遗漏 | 12 | 0 | -12 |
| repeat | 胜率 | 56.5% | 62.4% | +5.9pp |
| repeat | Meta 加权胜率 | 55.9% | 61.8% | +5.8pp |
| repeat | 第二回合 Powerful Hand | 20.0% | 21.2% | +1.2pp |
| repeat | post-KO 无 ready attacker | 72.8% | 67.2% | -5.6pp |
| repeat | 接力准备遗漏 | 6 | 0 | -6 |

### 决策

`observe`，不替换 V3 control。两批都解决了接力准备遗漏，但胜率方向不一致；先保留
规则并验证 Bench 保险 case。

## AutoIter V5：Active Alakazam 的 Bench 保险

### 迭代假设与 advisor 任务

当 Active 只有 Alakazam、Bench 为空、手牌有可合法使用的 Buddy-Buddy Poffin 且手牌
有 Abra/Dunsparce 时，除非当前攻击能完成最后奖赏闭环，否则先使用 Poffin 建立
Bench。`decision.md` 记录该局面实际的 legal options 与选择；本轮 advisor 需核对
Poffin 的 70 HP Basic 限制、Bench 容量和终局例外。

### 评测前后

前一版本：`iter-04` candidate；本轮 candidate：`iter-05`。两批均为 17 个对手 × 10
局完整 trace，control/candidate 为独立随机样本。

| 批次 | 指标 | V4 control | V5 candidate | 变化 |
|---|---|---:|---:|---:|
| 第一批 | 胜率 | 62.4% | 62.4% | 持平 |
| 第一批 | Meta 加权胜率 | 62.3% | 63.7% | +1.4pp |
| 第一批 | 第二回合 Powerful Hand | 27.6% | 24.1% | -3.5pp |
| 第一批 | post-KO 无 ready attacker | 67.1% | 64.4% | -2.7pp |
| 第一批 | 打手断档对局率 | 35.3% | 32.9% | -2.4pp |
| 第一批 | Bench 保险 miss | 8/10 | 0/5 | 解决；触发数受独立样本影响 |
| repeat | 胜率 | 62.4% | 57.1% | -5.3pp |
| repeat | Meta 加权胜率 | 61.8% | 57.8% | -3.9pp |
| repeat | 第二回合 Powerful Hand | 21.2% | 28.8% | +7.6pp |
| repeat | post-KO 无 ready attacker | 67.2% | 74.9% | +7.7pp |
| repeat | 接力断档对局率 | 40.0% | 37.1% | -2.9pp |
| repeat | Bench 保险 miss | 12/13 | 0/6 | 解决；触发数受独立样本影响 |

### 决策

`observe`，不晋级为稳定 control。Bench 保险的具体 case 在两批中都从有 miss 降为
0，且没有我方 action error；但独立 repeat 的胜率、Meta 和 post-KO 指标回退，说明
当前实现仍需检查它是否在某些资源紧张局面过早占用主行动。Poffin 的硬规则保留，下一
轮应由 advisor 优先复核“建立 Bench 与当回合进化/能量/终局攻击的冲突”，而不是简单
放宽保险门。

## AutoIter V6：Bench 接力连续性与 Rock Fighting Energy

### 迭代假设与 advisor 任务

本轮不改变固定 `deck.csv`。重点是把“Bench 有 Pokémon”细化为真正的 Abra-line
handoff：Dunsparce 只能算缓冲/抽牌引擎；Kadabra/Alakazam 需要 Psychic Energy，
Abra 需要可见的进化路线。advisor 同时核对用户新增的 Rock Fighting Energy 是否与
Mist Energy 一样阻挡 Alakazam 的 `Powerful Hand`。

### 实际改动

- Poffin 不再要求 Abra/Dunsparce 在手牌中，因为它搜索牌库；没有直接 handoff 时，
  优先 Poffin、Telepath 或直接放下手牌 Basic。
- analyzer 不再把已暴露直接附能/进化选项的状态误报成 Bench insurance miss；将
  Dunsparce-only Bench 继续视为未建立 Abra 接力。
- Rock Fighting Energy（ID `20`）与 Mist Energy（ID `11`）统一进入
  `PROTECTIVE_DAMAGE_ENERGIES`，攻击伤害、Boss、终局 KO 与 Enhanced Hammer 共用
  该模型。
- 新增 Mist/Rock Fighting 两个策略 fixture，并修复 analyzer 缺少
  `_field_pokemon()` 的回归错误。

### 评测前后

前一版本：`iter-05` candidate；本轮 candidate：`iter-06`。17 个对手 × 10 局，均为
完整 trace 的独立随机批次。

| 指标 | iter-05 candidate | iter-06 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 106 / 62 / 2 | 104 / 64 / 2 | 胜 -2 |
| 胜率 | 62.4% | 61.2% | -1.2pp |
| Meta 加权胜率 | 63.7% | 62.5% | -1.2pp |
| 第二回合 Powerful Hand | 24.1% | 25.3% | +1.2pp |
| post-KO 无 ready attacker | 64.4% | 61.2% | -3.2pp |
| 打手断档对局率 | 32.9% | 33.5% | +0.6pp |
| Bench insurance miss | 0 个 fail | 0 个 fail | 保持解决 |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

### 验收与决策

- 127 个单元测试全部通过；其中包括两种保护能量都会优先 Enhanced Hammer、不会被
  当成 `Powerful Hand` 有效 KO 的测试。
- 30 局 focused trace 重新分析后，`bench_insurance_missed` 误报为 0；完整 170
  局中记录到 252 个触发点且全部为 `pass`，没有 `fail`。
- 完整批次 post-KO 无 ready attacker 和第二回合指标方向改善，但胜率、Meta 加权
  胜率和打手断档对局率没有同时改善；原始 2 个 evaluator error 均为对手侧。

决策：**observe**，不把 iter-06 晋级为稳定 control。下一轮应继续分析击倒前的
上游铺场和资源接力，而不是在 KO 后假设可以临时补救。

## AutoIter V7：Rock Fighting Energy 的 Fighting 属性边界

### 迭代假设与 advisor 任务

本轮不改变固定 `deck.csv`。用户要求将 Rock Fighting Energy 按 Mist Energy 处理；核对
卡表后补充精确边界：Rock 的保护只对 Fighting 属性宝可梦生效，Mist 对任意属性生效。
由于 observation 不直接暴露 Pokémon 属性，advisor 核对了提交包内同版本引擎卡表的
可用性，并采用未知 Rock 目标保守视为“可能受保护”。

### 实际改动

- 使用 `cg.api.all_card_data()` 缓存 Pokémon 的打印属性。
- `_has_protective_damage_energy()` 对 Mist 无条件保护，对 Rock 仅在 Fighting 或属性
  未知时保护。
- 攻击、KO、Boss/终局与 Enhanced Hammer 继续共享该函数。
- 新增 Fighting Rock 正例和非 Fighting Rock 反例。

### 评测前后

前一版本：`iter-06` candidate；本轮 candidate：`iter-07`。两批为独立随机完整 trace。

| 指标 | iter-06 candidate | iter-07 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 104 / 64 / 2 | 97 / 71 / 2 | 胜 -7 |
| 胜率 | 61.2% | 57.1% | -4.1pp |
| Meta 加权胜率 | 62.5% | 57.2% | -5.3pp |
| 第二回合 Powerful Hand | 25.3% | 20.0% | -5.3pp |
| post-KO 无 ready attacker | 61.2% | 78.9% | +17.7pp |
| 打手断档对局率 | 33.5% | 42.4% | +8.9pp |
| action error / 空 Bench Run Away Draw | 0 / 0 | 0 / 0 | 持平 |

### 决策

**observe**。整体指标下降来自独立随机批次，不能反推规则修正有害；该规则正确性修正
保留，不晋级为稳定 control。下一轮继续处理可由 trace 证明的上游接力错误。

## AutoIter V8：Active Kadabra 的接力能量

### 迭代假设与 advisor 任务

本轮针对击倒前的具体 trace，而不是只看 post-KO 汇总。advisor 找到两个资源明确可见
但被错误消耗的 case：Active Kadabra 已有 Psychic，Telepath 却贴回 Active，Bench
Kadabra/Abra 未获得接力能量。Active Kadabra 的正确路线是自然进化并继承已有 Psychic，
不能因为它还不是 Alakazam 就把 Bench 接力保护关闭。

### 实际改动

- `_bench_handoff_preparation_due()` 从仅接受 Active Alakazam 扩展到已带 Psychic 的
  Active Kadabra 或 Alakazam。
- 只在合法 Bench Abra/Kadabra/Alakazam 目标和实际 Psychic/Telepath/Lana's Aid 选项
  可见时触发；终局闭环和自然进化优先级保留。
- 新增 `Active Kadabra + Telepath(Active/Bench)` 的失败 fixture，旧策略先失败、新策略
  通过。

### 评测前后

前一版本：`iter-07` candidate；本轮 candidate：`iter-08`。两批为独立随机完整 trace。

| 指标 | iter-07 candidate | iter-08 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 97 / 71 / 2 | 101 / 68 / 1 | 胜 +4 |
| 胜率 | 57.1% | 59.4% | +2.3pp |
| Meta 加权胜率 | 57.2% | 59.6% | +2.4pp |
| 第二回合 Powerful Hand | 20.0% | 17.1% | -2.9pp |
| post-KO 无 ready attacker | 78.9% | 70.9% | -8.0pp |
| 打手断档对局率 | 42.4% | 38.2% | -4.2pp |
| action error / 空 Bench Run Away Draw | 0 / 0 | 0 / 0 | 持平 |

### 决策

**observe**，不晋级为稳定 control。具体 fixture 与接力指标方向改善，但第二回合指标
回退，且仍低于 iter-06 独立 control；继续以具体 trace 为下一轮输入。

## AutoIter V9：swap 回合计数与 Active Abra 的确定 KO 进化

### 迭代假设与 advisor 任务

本轮对应工作区迭代 `iter-18-turn-count`。advisor 复核了
`kiyotah_dragapult/game_008.json`：对手先使用 Budew 的 Itchy Pollen 后，shared turn 3
已经是我方交换先手后的第二个己方回合；Active Abra 有 Psychic、手牌有 Kadabra，
自然进化后可以用 Super Psy Bolt 击倒 30 HP 的 Budew。原策略把该回合误判为第一个己方
回合并选择 END。

### 实际改动

- `_own_turn_number()` 使用 `firstPlayer` 与 `yourIndex` 计算己方回合序号，修复 swap
  对局中第二个己方回合的进化合法性。
- 增加“Active Abra → Kadabra 后可确定 KO”判断；只有合法进化、已有 Psychic、手牌有
  Kadabra 且进化后确实可以 KO 时，才在 Bench Abra 进化之前选择 Active 进化。
- `deck.csv`、攻击终止规则、Supporter/手填能量次数均未修改。

### Case 验收

`kiyotah_dragapult/game_008.json`、shared turn 3、trace[27]：

| 版本 | 首步动作 | 解释 |
|---|---|---|
| iter-15/control | `[6] END` | 错误把 swap 后的己方第二回合当作第一回合 |
| iter-18/candidate | `[1]` | Active Abra → Kadabra；随后 Psychic Draw + Super Psy Bolt 可 KO |

对应 fixture 已写入 `tests/test_alakazam_v7_auto_iter_strategy.py`，并覆盖 Item Lock 不
阻止自然进化的边界。

### 评测前后

control 是 `iter-15-patch-priority` 的 170 局 full-trace 批次；candidate 是独立随机的
17 个对手 × 10 局 full-trace 批次，不能视为逐局 A/B。

| 指标 | iter-15 control | iter-18 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 109 / 60 / 1 | 胜率下降 |
| 胜率 | 68.2% | 64.1% | -4.1pp |
| Meta 加权胜率 | 70.4% | 64.2% | -6.2pp |
| 第二回合 Powerful Hand | 27.1% | 28.8% | +1.8pp |
| post-KO 无 ready attacker | 68.5% | 64.0% | -4.5pp |
| 打手断档对局率 | 36.5% | 34.1% | -2.4pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

candidate 有一个 `yakitori_raging_bolt/game_002` 的 evaluator `IndexError`，trace role
为 `opponent`，没有计入我方错误。

### 决策

**observe**，不覆盖 `BEST_STRATEGY.json`。目标 case 与节奏指标得到局部修复，但总体
胜率和 Meta 加权胜率低于当前 best。iter-18 只作为工作区候选继续分析；08:05 定时任务
仍提交不可变的 `iter-15-patch-priority` 归档。

## 后续追加规则

## AutoIter V10：非终局 KO 前的 Bench Kadabra → Alakazam 接力

### 迭代假设与 advisor 任务

本轮对应工作区迭代 `iter-19-bench-alakazam`。advisor 复核
`romanrozen_v9/game_003.json`：Active Alakazam 已有 Psychic，Bench 有两只已附
Psychic 的 Kadabra，手中有 Alakazam；当前攻击可以 KO，但对方仍有 Bench，故这不是
最后奖赏闭环。原策略直接 Powerful Hand，浪费了攻击前完成接力的窗口。

### 实际改动

- 增加非终局确定 KO 的 Bench Kadabra → Alakazam 接力 gate。
- 只有当 Active Alakazam 已有 Psychic、手牌有 Alakazam、目标 Bench Kadabra 已有
  Psychic 且进化选项合法时，才将该进化排在攻击前。
- `deck.csv`、终局攻击、牌库保护和每回合资源上限不变。

### Case 验收

| 版本 | 首步动作 | 解释 |
|---|---|---|
| iter-18/control | `[10]` Powerful Hand | 非终局 KO 前没有准备 Bench 接力 |
| iter-19/candidate | `[4]` | 先把已充能 Bench Kadabra 进化为 Alakazam |

该 case 已加入 `tests/test_alakazam_v7_auto_iter_strategy.py`，并用原始 replay 的
option index 重放验证。

### 评测前后

为了保持与当前 best 一致，表格使用 iter-15 的独立 full-trace 170 局作为 control；
candidate 为另一批独立 170 局，因此变化只作 guardrail。

| 指标 | iter-15 control | iter-19 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 106 / 63 / 1 | 胜率下降 |
| 胜率 | 68.2% | 62.4% | -5.9pp |
| Meta 加权胜率 | 70.4% | 60.7% | -9.7pp |
| 第二回合 Powerful Hand | 27.1% | 28.2% | +1.2pp |
| post-KO 无 ready attacker | 68.5% | 62.8% | -5.7pp |
| 打手断档对局率 | 36.5% | 38.8% | +2.4pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

原始 summary 的 1 个 `yakitori_raging_bolt/game_002` evaluator error 属于 opponent
role，不计入我方 action error。

### 决策

**observe**，不更新 `BEST_STRATEGY.json`。目标 replay case 已修复，但独立 full matrix
的胜率和 Meta 加权胜率明显低于 iter-15，且打手断档对局率上升。下一轮必须先分析
candidate 的具体败局，再决定收紧 gate、拆分迭代，还是回退工作区候选。

## AutoIter V11：无可见 Kadabra 时也允许 Wondrous Patch 先充 Bench Abra

### 迭代假设与 advisor 任务

本轮对应工作区迭代 `iter-20-patch-abra`。advisor 复核
`penguin_915/game_010.json`：Active Alakazam 已有 Psychic，Bench 有 Abra，弃牌区有
Basic Psychic，手牌有 Wondrous Patch；即使手牌暂时没有 Kadabra，本回合仍可以先把
能量贴给 Abra，为下一回合接力保留可能性。

### 实际改动

- `_bench_handoff_preparation_due()` 不再要求 Abra 的下一阶段进化牌已经在手中，允许
  可见的 Wondrous Patch/Basic Psychic/Lana's Aid 先完成能量准备。
- 终局奖赏闭环、Item Lock、进化时间和牌库保护不变。

### Case 验收

| 版本 | 首步动作 | 解释 |
|---|---|---|
| iter-19/control | `[5]` | 未选择可见的 Wondrous Patch |
| iter-20/candidate | `[4]` | Patch 将弃牌区 Basic Psychic 贴到 Bench Abra |

fixture 还覆盖“手牌没有下一阶段进化牌”这一边界，终局 Patch 测试保持通过。

### 评测前后

control 为 iter-15 的 full-trace 170 局；candidate 为独立随机的 full-trace 170 局。

| 指标 | iter-15 control | iter-20 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 103 / 64 / 3 | 胜率下降 |
| 胜率 | 68.2% | 60.6% | -7.6pp |
| Meta 加权胜率 | 70.4% | 60.2% | -10.2pp |
| 第二回合 Powerful Hand | 27.1% | 28.2% | +1.2pp |
| post-KO 无 ready attacker | 68.5% | 69.0% | +0.5pp |
| 打手断档对局率 | 36.5% | 40.0% | +3.5pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

### 决策

**observe**，不覆盖 `BEST_STRATEGY.json`。具体 case 已修复，但放宽到所有“下一阶段
不可见”的 Abra 后，整体指标没有改善。下一轮需要辨别“先充能确实能形成接力”和
“只增加了资源消耗但仍无法接班”的差别；当前 08:05 仍提交 iter-15 immutable artifact。

## AutoIter V12：收紧 Wondrous Patch 的 Abra gate（iter-21）

### 改动与 advisor 结论

本轮相对 iter-20 只收紧 `Wondrous Patch` 的提前执行条件：Bench Abra 未附 Psychic
时，只有手牌可见 Kadabra，或可见 `Rare Candy + Alakazam` 路线，Patch 才能排在非终局
攻击前。Bench Kadabra/Alakazam 的 Patch 路线不变；`deck.csv` 不变。

advisor 核验了 `penguin_915/game_010.json` 的 trace 79--82：原策略把 Psychic 贴给
已经带 Psychic 的 Abra，属于合法但没有新增接力价值的资源消耗。进化时机、攻击立即
结束回合、Item Lock 和终局奖赏闭环均保持原规则。

### Case 验收

| 观察项 | iter-20 | iter-21 |
|---|---|---|
| 无可见下一阶段的 Bench Abra 被 Patch 抢先处理 | 允许 | 禁止 |
| `bench_insurance_missed` 非 pass case | 需复核 | 0（244 条均为 pass） |
| 我方 action error | 0 | 0 |

### 完整评测

两边均为 17 个对手 × 10 局 full-trace，但使用独立随机样本；control 是 iter-15
immutable best，candidate 是 iter-21 工作区。结果只作为 guardrail，不作逐局因果结论。

| 指标 | iter-15 control | iter-21 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 105 / 63 / 2 | 下降 |
| 胜率 | 68.2% | 61.8% | -6.5pp |
| Meta 加权胜率 | 70.4% | 62.7% | -7.7pp |
| 第二回合 Powerful Hand | 46/170 (27.1%) | 36/170 (21.2%) | -5.9pp |
| post-KO 无 ready attacker | 124/181 (68.5%) | 133/191 (69.6%) | +1.1pp |
| 打手断档对局率 | 36.5% | 38.8% | +2.4pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

### 决策

**observe，不晋升。** 目标 replay case 的局部行为已经修复，但整体胜率、Meta、第二
回合攻击和接力断档率均不满足 guardrail。`BEST_STRATEGY.json` 继续指向
`iter-15-patch-priority` immutable artifact；iter-21 只保留在工作区作为下一轮实验
基础。下一轮不再重复 Patch 变量，改为严格限定的 Poffin 目标分配实验。

## AutoIter V13：Poffin 的 Enriching 目标 gate（iter-22）

### 改动与 advisor 结论

本轮只在完整可见条件下让 Poffin 优先寻找 Dunsparce：场上至少三只 Abra-line、没有
Dunsparce、手中有 Enriching Energy，并且 Active Abra/Kadabra 有合法的直接 Alakazam
路线。其它情形仍优先 Abra，保证 Bench 接力；`deck.csv` 不变。

### 评测前后

candidate 是独立随机的 17×10 full-trace，control 是 iter-15 immutable best，不能
逐局解释为因果变化。

| 指标 | iter-15 control | iter-22 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 102 / 67 / 1 | 下降 |
| 胜率 | 68.2% | 60.0% | -8.2pp |
| Meta 加权胜率 | 70.4% | 58.9% | -11.5pp |
| 第二回合 Powerful Hand | 27.1% | 22.4% | -4.7pp |
| post-KO 无 ready attacker | 68.5% | 71.5% | +3.0pp |
| 打手断档对局率 | 36.5% | 36.5% | 持平 |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

### 决策

**observe，不晋升。** fixture 层面的 Poffin 选择正确，但整体指标没有提升；继续
使用 iter-15 immutable artifact 作为定时提交 best。

## AutoIter V14：Lana's Aid 回收弃牌区接力资源（iter-23）

### 改动与 advisor 结论

当 Active Kadabra/Alakazam 已有 Psychic、Bench 没有 Abra-line、Bench 有空位、弃牌区
同时有 Abra 与 Basic Psychic，且本次攻击不是最后奖赏闭环时，Lana's Aid 排在攻击前，
并在选择阶段同时取回 Abra 与 Basic Psychic。随后分开的主动作负责放下 Abra 和手动
填能量；不在同一回合强行进化。`deck.csv` 不变。

### 评测前后

candidate 是独立随机的 17×10 full-trace，control 是 iter-15 immutable best。

| 指标 | iter-15 control | iter-23 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 103 / 65 / 2 | 下降 |
| 胜率 | 68.2% | 60.6% | -7.6pp |
| Meta 加权胜率 | 70.4% | 59.6% | -10.8pp |
| 第二回合 Powerful Hand | 27.1% | 23.5% | -3.5pp |
| post-KO 无 ready attacker | 68.5% | 68.3% | -0.2pp |
| 打手断档对局率 | 36.5% | 35.3% | -1.2pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

### 决策

**observe，不晋升。** 接力指标方向改善，但胜率、Meta 和第二回合指标仍低于 best。
本批 3 个 `bench_insurance_missed=fail` 需要结合原始 trace 复核；至少一例是已经
选择 Lana's Aid 后 analyzer 未识别完整铺场链，不能把测量误报直接当策略退化。

每完成一轮，在对应小节中追加：

1. 具体改动文件和策略假设；
2. control/candidate 的完整指标与样本口径；
3. 至少一个具体 case 的动作前后对照；
4. correctness、第二回合、接力和胜率 guardrail；
5. `accept`、`observe` 或 `reject` 决策及理由。
6. advisor 的卡牌/规则核查结论，以及主 agent 是否采纳、拒绝或延后。

如果某轮只完成了 summary-only 评测，也必须明确写出“哪些指标不可用”，待有空间
后补跑完整 trace，不能用缺失值制造看似连续的提升曲线。

## AutoIter V15：Night Stretcher 窄接力路径（iter-24）

### 改动与 advisor 结论

本轮只增加一个弃牌区接力分支：Active Kadabra/Alakazam 已附 Psychic、Bench 没有
Abra-line、弃牌区有 Abra、手牌已有 Psychic 且没有更好的 Poffin/手牌 Abra/Lana
路径时，Night Stretcher 可以先取回 Abra，再在本回合下到 Bench 并准备下一回合进化。
Night Stretcher 不能同时取回 Energy，因此没有手牌 Psychic 时不触发。终局攻击和
Budew 的 `Itchy Pollen` Item Lock 保持优先；`deck.csv` 未修改。

advisor 还复核了 iter-23 的三个 `bench_insurance_missed=fail`：实际都是
`Lana's Aid -> Abra + Psychic` 的合法接力，属于 analyzer 误报。分析器增加恢复链
识别后，这些 case 都记录为 `pass`，没有把统计修复冒充策略提升。

### 评测前后

candidate 是独立随机的 17 个对手 × 10 局 full-trace；control 是 iter-15 immutable
best。窄条件在本批实际触发 23 次，均选择 Night Stretcher，action error 为 0。

| 指标 | iter-15 control | iter-24 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 107 / 61 / 2 | 胜率下降 |
| 胜率 | 68.2% | 62.9% | -5.3pp |
| Meta 加权胜率 | 70.4% | 63.7% | -6.7pp |
| 第二回合 Powerful Hand | 27.1% | 21.8% | -5.3pp |
| post-KO 无 ready attacker | 68.5% | 65.4% | -3.1pp |
| 打手断档对局率 | 36.5% | 38.2% | +1.8pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

### 决策

**observe，不晋升。** post-KO 事件级指标局部改善，但对局级打手断档率、胜率、Meta
和第二回合攻击率没有满足 guardrail。`BEST_STRATEGY.json` 仍保持
`iter-15-patch-priority`，下一轮应继续围绕“二回合铺场”和“对局级接力连续性”寻找
更早、可验证的最小变量。

## AutoIter V16：已有 Stage 1/2 Bench 时的二回合攻击优先级（iter-25）

### 改动与 advisor 结论

本轮在 iter-24 基础上增加一条条件式攻击优先级：Active Alakazam 已附 Psychic、
属于自己的第二回合、当前确实有 Powerful Hand（1072）选项，并且 Bench 已有 Kadabra
或 Alakazam 时，不因泛化 Poffin/Hilda setup 阻塞攻击。直接 Psychic 附能、Active
Kadabra 自然进化、空 Bench/只有 Dunsparce 的 insurance、Item Lock 和终局攻击不受
影响；`deck.csv` 未修改。

advisor 的规则核验支持这个边界，但明确不应扩展到没有真正攻击选项、未充能 Active
Kadabra 或没有 Abra-line Bench 的局面。本批 raw trace 有 38 次符合条件且直接使用
Powerful Hand 的动作，说明局部规则命中了目标。

### 评测前后

candidate 是独立随机的 17 个对手 × 10 局 full-trace；control 是 iter-15 immutable
best，iter-24 仅作相邻候选参考。

| 指标 | iter-15 control | iter-24 candidate | iter-25 candidate |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 107 / 61 / 2 | 104 / 61 / 5 |
| 胜率 | 68.2% | 62.9% | 61.2% |
| Meta 加权胜率 | 70.4% | 63.7% | 63.0% |
| 第二回合 Powerful Hand | 27.1% | 21.8% | 23.5% |
| post-KO 无 ready attacker | 68.5% | 65.4% | 65.3% |
| 打手断档对局率 | 36.5% | 38.2% | 36.5% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

### 决策

**observe，不晋升。** 二回合指标相对 iter-24 上升，但相对 best 仍低 3.5pp；胜率
低 7.0pp，Meta 低 7.4pp。局部动作已符合设计，整体结果不足以覆盖当前 best，
`BEST_STRATEGY.json` 继续保持 `iter-15-patch-priority`。

## AutoIter V17：无可见进化路线时的 Basic Psychic 接力 gate（iter-26）

### 改动与 advisor 结论

本轮针对真实 replay 中“Active 已经有 Psychic，却把唯一 Telepath Energy 贴回 Active”
的错误，复核并固化了 Bench 接力的资源语义。此前工作区已经覆盖 Telepath→Bench Abra、
Lana's Aid→Abra + Basic Psychic 和新下 Abra 的接力；本轮新增一条更窄的边界：Basic
Psychic 直接贴给 Bench Abra，只有在手牌已看到 Kadabra，或已看到 Alakazam + Rare Candy
且没有 Budew `Itchy Pollen` Item Lock 时，才会被当作可验证接力。Telepath 仍可在没有
可见进化牌时建立 Bench，因为它的检索效果本身会把 Basic Psychic Pokémon 放到 Bench。
`deck.csv` 未修改。

### 评测前后

candidate 是 17 个对手 × 10 局 full-trace 独立随机批次；control 是当前 `BEST_STRATEGY.json`
指向的 iter-15 immutable best。完整 trace 保存在隔壁评测仓库，仓库内只保留 summary-level
指标。

| 指标 | iter-15 control | iter-26 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 111 / 59 / 0 | 下降 |
| 胜率 | 68.2% | 65.3% | -2.9pp |
| Meta 加权胜率 | 70.4% | 64.8% | -5.6pp |
| 第二回合 Powerful Hand | 27.1% | 19.4% | -7.7pp |
| post-KO 无 ready attacker | 68.5% | 69.8% | +1.3pp |
| 对局级打手断档 | 36.5% | 38.2% | +1.7pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

### 具体验收与决策

- 新增 fixture：无可见进化路线的 Basic Psychic→Bench Abra 不再压过攻击，测试通过。
- 完整 trace 发现 67 次该类选项，候选实际选择 0 次；Telepath→Bench Abra 的 406 次
  机会中选择 95 次，未误伤 Telepath 的检索型 Bench anchor。
- 之前三个真实“Telepath 贴 Active”case 在当前策略中均改为选择 Bench attachment。

**observe，不晋升。** 本轮 action error、空 Bench Run Away Draw 和 focused case 均保持
正确，但整体胜率、Meta、第二回合指标与对局级接力断档均未超过 best。因此不更新
`BEST_STRATEGY.json`，08:05 继续提交 `iter-15-patch-priority`。下一轮保持单变量，
验证第二回合 Powerful Hand 是否错误地越过确定 Boss 高奖赏 KO 路线。

## AutoIter V18：确定 Boss KO 压过第二回合 Powerful Hand（iter-27）

### 改动与 advisor 结论

iter-25 的第二回合攻击 gate 可能把“更快攻击”排在本回合已经可确认的 Boss KO 之前。
本轮增加 `second_turn_boss_prize_route`：当 Boss's Orders 能把对手 Bench 的确定 KO
目标拉到 Active 时，Boss 路线先于第二回合 Powerful Hand；终局闭环、接力准备、保护
能量和 Item Lock 均保持原边界。`deck.csv` 未修改。

### 评测前后

candidate 是 17 个对手 × 10 局 full-trace 独立随机批次；control 是 iter-15 immutable
best，iter-26 作为相邻 candidate 参考。

| 指标 | iter-15 control | iter-26 candidate | iter-27 candidate |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 111 / 59 / 0 | 109 / 59 / 2 |
| 胜率 | 68.2% | 65.3% | 64.1% |
| Meta 加权胜率 | 70.4% | 64.8% | 65.0% |
| 第二回合 Powerful Hand | 27.1% | 19.4% | 24.7% |
| post-KO 无 ready attacker | 68.5% | 69.8% | 71.5% |
| 对局级打手断档 | 36.5% | 38.2% | 41.2% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

评测中 `yakitori_raging_bolt` 有 2 局对手侧 `IndexError`，没有计入我方 action error。

### 具体验收与决策

- 新增 Boss/Powerful Hand fixture：改动前选 Powerful Hand，改动后选 Boss，测试通过。
- 第二回合 Powerful Hand 相对 iter-26 上升 5.3pp，但 post-KO 无 ready 与对局级断档
  分别上升 1.7pp 和 2.9pp。

**observe，不晋升。** 该规则修正符合宝可梦卡牌奖赏优先级，但本批整体胜率仍低于
best，不能更新 `BEST_STRATEGY.json`；08:05 继续提交 `iter-15-patch-priority`。

## AutoIter V19：未充能 Stage 1 不等于 ready 接班人（iter-28）

### 改动与 advisor 结论

iter-27 的 post-KO 无 ready attacker 达到 71.5%，主要可疑点是
`_bench_insurance_due()` 看到任意 Bench Kadabra/Alakazam 就提前返回 False。规则 advisor
确认：没有 Psychic Energy 的 Stage 1/2 不能在攻击宣告后立即接班；本回合若有实际的
Poffin、Telepath、Wondrous Patch、Lana's Aid 或直接附能路线，应先完成接力，否则不能
为了理论上的未来资源跳过攻击。

本轮删除该过宽例外，`deck.csv`、终局闭环、Item Lock、牌库保护和 Rock/Mist Energy
伤害保护逻辑均未修改。新增两个 fixture，覆盖“有 Poffin 先建 anchor”和“无可见 anchor
仍攻击”；V7 策略测试 45/45 通过。

### 评测前后

candidate 是 17 个对手 × 10 局 full-trace 独立随机批次；control 是 iter-15 immutable
best，iter-27 作为相邻候选参考。

| 指标 | iter-15 control | iter-27 candidate | iter-28 candidate |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 109 / 59 / 2 | 108 / 61 / 1 |
| 胜率 | 68.2% | 64.1% | 63.5% |
| Meta 加权胜率 | 70.4% | 65.0% | 63.6% |
| 第二回合 Powerful Hand | 27.1% | 24.7% | 22.9% |
| post-KO 无 ready attacker | 68.5% | 71.5% | 65.3% |
| 对局级打手断档 | 36.5% | 41.2% | 35.3% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

### 决策

**observe，不晋升。** 本轮接力连续性明显改善，但胜率、Meta 和第二回合 Powerful Hand
均低于当前 best，不能让局部指标的改善覆盖主结果 guardrail。`BEST_STRATEGY.json`
继续指向 `iter-15-patch-priority`，08:05 继续提交该 immutable archive。下一轮在保留
这次 ready 判定修复的前提下，单独分析第二回合节奏下降的来源。

## AutoIter V20：二回合已有攻击线时保留 Powerful Hand（iter-29）

### 改动与 advisor 结论

iter-28 将“未充能 Stage 1/2”从 ready attacker 中排除后，二回合已有 Abra 线但仍有 Poffin
的局面也被 Bench insurance 延迟。规则 advisor 建议：二回合若 Bench 已经有 Abra、Kadabra
或 Alakazam，且没有直接 Psychic/Telepath/Patch/回收接力路线，应保留本回合 Powerful Hand；
空 Bench 或只有 Dunsparce 仍必须先建立 Bench，真实附能路线也继续优先。

本轮新增二回合 fixture；iter-28 的“第三回合有 Poffin 先建 anchor”和“无 anchor 仍攻击”
fixture 继续通过，`deck.csv` 未修改。

### 评测前后

candidate 与 control 是独立随机 full-trace 批次；iter-28 作为相邻候选参考。

| 指标 | iter-15 control | iter-28 candidate | iter-29 candidate |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 108 / 61 / 1 | 109 / 61 / 0 |
| 胜率 | 68.2% | 63.5% | 64.1% |
| Meta 加权胜率 | 70.4% | 63.6% | 63.2% |
| 第二回合 Powerful Hand | 27.1% | 22.9% | 27.1% |
| post-KO 无 ready attacker | 68.5% | 65.3% | 69.8% |
| 对局级打手断档 | 36.5% | 35.3% | 32.9% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

### 决策

**observe，不晋升。** 二回合指标恢复且对局级断档下降，但 post-KO event rate、胜率和
Meta 仍没有超过当前 best。`BEST_STRATEGY.json` 继续指向 `iter-15-patch-priority`，
08:05 继续提交 immutable iter-15。下一轮优先分析 iter-29 的具体 post-KO case，不再扩大
二回合例外。

## AutoIter V21：归一化 option 的 `indexInArea` 接力解析修复（iter-30）

### 改动与 advisor 任务

本轮针对历史 replay `iter-27-boss-before-second-turn-attack-20260720/`
`kiyotah_dragapult/game_002.json` 的 turn 7。评测报告中的 normalized option 同时包含
局部 option 下标 `index` 与原始手牌位置 `indexInArea`；若只读取前者，Basic Psychic 或
Telepath 可能被解析成另一张手牌，从而漏掉 Bench Kadabra/Alakazam 的附能接力路线。

本轮让 `_option_card_id()` 在 `type=7` 与 `area=2` 的手牌 option 中优先使用
`indexInArea`，没有该字段时回退到原始 evaluator observation 的 `index`。`deck.csv`、
终局、牌库、保护能量和 Poffin/接力策略 gate 均未改变。

按 AutoIter 约定启动独立 strategy advisor，要求其核对 deck 卡牌效果、规则合法性、历史
case 和本轮 focused trace；advisor 不直接修改生产代码，结论写入
`docs/reports/kaggle/alakazam-v7-auto-iter/iter-30/advisor.md`。

### 验收与测试

- 新增 normalized Energy option fixture：`index=1`、`indexInArea=0` 时仍解析为手牌
  Telepath，并选择 Bench handoff；
- 新增孤立手牌资源 fixture：只有手牌中的回收/进化/能量，不应被当作已暴露的 Bench route；
- 全套 `167` 个 unittest、`py_compile` 和 `python3 scripts/check_assets.py` 通过；
- 固定 `deck.csv` 未修改。

### focused 评测

本轮只运行 `kiyotah_dragapult`、`sue_alakazam` 各 10 局 full-trace，默认独立随机：

| 指标 | iter-30 focused |
|---|---:|
| 胜 / 负 / 平 | 7 / 13 / 0 |
| 胜率 | 35.0% |
| Meta 加权胜率 | 34.5% |
| 第二回合 Powerful Hand | 2/20 = 10.0% |
| post-KO 无 ready attacker | 38/54 = 70.4% |
| 对局级打手断档 | 14/20 = 70.0% |
| 空 Bench Run Away Draw | 0 |
| 我方 action error | 0 |

该 focused 批次没有复现旧 replay 的完全相同状态，因此不能从 20 局结果推断 parser 修复
的完整矩阵因果效果。原始 focused trace 保存在 `/tmp/alakazam-v7-auto-iter-30-focus/`，
repo 只保存摘要。

### 决策

**observe，不晋升。** 本轮回归、合法性和历史 fixture 通过，但 focused 样本小且独立
随机，主指标不能证明超过 iter-15 best；`BEST_STRATEGY.json` 继续指向
`iter-15-patch-priority`。下一轮应从 post-KO 断档的具体历史 trace 继续寻找最小的可验证
策略改动，而不是把 parser 兼容修复误当成整体胜率提升。

## AutoIter V22：Active Telepath 附能同时建立 Bench 锚点（iter-31）

### 改动与证据

历史 `iter-27-boss-before-second-turn-attack-20260720/kiyotah_dragapult/game_002.json` 的
shared turn 7 中，Active Kadabra 未充能，Bench 只有未充能 Alakazam 且仍有空位，手牌同时有
Basic Psychic 与 Telepath Psychic。原策略在两个 Active 附能 option 同分时选 Basic；V31 在
Active Abra-line 未充能、Bench 有空位且没有已充能接班人时优先 Telepath，利用其附能时最多搜索
两只 Basic Psychic Pokémon 的效果建立 Bench 锚点。`deck.csv` 未修改。

独立 strategy advisor 同时指出：iter-30 的 `indexInArea` 修复合理但 `area=3` 弃牌区仍需
单独核验；本轮不混入该变量。

### 评测前后

| 指标 | iter-15 full best | iter-30 focused | iter-31 focused |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 7 / 13 / 0 | 10 / 10 / 0 |
| 胜率 | 68.2% | 35.0% | 50.0% |
| Meta 加权胜率 | 70.4% | 34.5% | 48.2% |
| 第二回合 Powerful Hand | 27.1% | 10.0% | 20.0% |
| post-KO 无 ready attacker | 68.5% | 70.4% | 66.7% |
| 对局级打手断档 | 36.5% | 70.0% | 60.0% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

### 决策

**observe，不晋升。** 新 fixture 与历史 raw observation 均验证了目标动作，75 个相关测试
通过；但 focused 样本仅 20 局且与 iter-30 独立随机，不能证明完整矩阵提升。`BEST_STRATEGY.json`
继续指向 `iter-15-patch-priority`。下一轮单独研究 Night Stretcher 弃牌索引语义。

## AutoIter V23：弃牌区 `indexInArea` 解析修复（iter-32）

### 改动与 advisor 结论

本轮固定 `deck.csv`，只修改 `_option_card_id()`：对 `area=3` 的弃牌区 option 优先读取
`indexInArea`，缺失时回退到 `index`。这样 Night Stretcher 或 Lana's Aid 的选项不会把
当前 select 列表的局部序号误当成弃牌区位置，从而错判 Abra、Basic Psychic 或进化宝可梦。

本轮增加独立规则/卡牌 advisor 和 replay advisor。advisor 确认该修改符合 evaluator 的
option 语义，但也确认不能因为手牌里有孤立资源就禁止攻击；没有实际 Bench target 或
合法 recovery route 时，仍必须遵循原有攻击优先级。最新 170 局 trace 没有实际出现
`area=3` 选择，因此真实 Night Stretcher 动作链仍标记为待观察。

### 验收

- 新增 fixture 在生产修复前失败，修复后通过：`index` 与 `indexInArea` 不同时能解析到
  弃牌区的真实卡牌。
- V7 AutoIter 专项 50 个 unittest、全量 169 个 unittest 均通过。
- `deck.csv` 未修改；空 Bench Run Away Draw 与我方 action error 均保持为 0。

### 完整评测

本轮为 17 个对手 × 10 局 = 170 局，evaluator 默认独立随机，不是逐局 A/B：

| 指标 | iter-27 control | iter-32 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 109 / 59 / 2 | 118 / 51 / 1 | 胜 +9 |
| 胜率 | 64.1% | 69.4% | +5.3pp |
| Meta 加权胜率 | 65.0% | 70.1% | +5.1pp |
| 第二回合 Powerful Hand | 24.7% | 18.2% | -6.5pp |
| post-KO 无 ready attacker | 71.5% | 66.5% | -5.0pp |
| 对局级打手断档 | 41.2% | 33.5% | -7.7pp |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

### 决策

**observe，不晋升。** Candidate 的胜率、Meta 和接力事件率方向较好，但第二回合
Powerful Hand 明确回退，且本轮没有触发真实弃牌区选择；因此暂不更新
`BEST_STRATEGY.json`，保留窄解析修复，下一轮继续寻找能同时守住二回合铺场与接力的真实
动作 case。

## AutoIter V24：Dudunsparce 进化堆叠牌库估算修复（iter-33）

### 改动与 advisor 结论

本轮固定 `deck.csv`，修复 Dudunsparce 的 Run Away Draw 牌库净变化计算：能力会把选中的
Dudunsparce、其 `preEvolution` 中的 Dunsparce 以及附着卡一起洗回牌库；旧实现遗漏了
`preEvolution`。同时按实际 option 定位被选择的 Dudunsparce 实例，避免多只 Dudunsparce
时使用错误的附着数量。

规则/卡牌 advisor 确认这是牌库状态模型的正确性修复，而不是新的抽牌优先级：空 Bench
时仍不能为了抽牌使用 Run Away Draw；孤立的 Stage 1/Stage 2 仍不能被假设为当前回合的
接力路线。

### 评测前后

两批 candidate 都是 17 个对手 × 10 局的独立随机 full-trace，不能进行逐局 A/B：

| 指标 | iter-32 参考 | iter-33 首批 | iter-33 repeat |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 118 / 51 / 1 | 107 / 62 / 1 | 104 / 65 / 1 |
| 胜率 | 69.4% | 62.9% | 61.2% |
| Meta 加权胜率 | 70.1% | 65.0% | 61.3% |
| 第二回合 Powerful Hand | 18.2% | 28.2% | 25.9% |
| post-KO 无 ready attacker | 66.5% | 68.0% | 68.9% |
| 对局级打手断档 | 33.5% | 32.9% | 34.1% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

### 决策

**observe，不晋升。** 两批二回合指标均高于 iter-32，但胜率、Meta 和 post-KO 接力
没有同步稳定改善；因此不更新 `BEST_STRATEGY.json`，保留该规则修复和回归 fixture。下一
轮继续从具体 post-KO 断档的击倒前状态寻找单变量、可验证的 Bench 接力改动。

## AutoIter V25：Poffin 的 routed Abra gate（iter-34）

### 改动与 advisor 结论

本轮固定 `deck.csv`，修复 Poffin 将“带 Psychic 但没有可见进化路线的 Bench Abra”
误判为 ready handoff 的问题。只有存在显式 `Dunsparce + Enriching Energy` 抽牌路线，
或已有可见 Abra-line handoff 路线时，Poffin 才允许优先选择 Dunsparce；否则继续优先
从牌库建立 Abra。没有全局修改 ready attacker 定义，也没有放宽 Rare Candy 时序。

独立 strategy advisor 复核了 Poffin 的 HP≤70 Basic 限制和 iter-33 的真实击倒前 trace，
确认不能把孤立 Stage 1/Stage 2、Dunsparce 或未规划进化的 Abra 当作 ready。advisor 同时
定位了下一轮应单独验证的三个合法 Bench Kadabra/Alakazam Psychic 或 Telepath 附能目标
未被选择 case；该问题不与本轮 Poffin gate 混合。

### 评测与验收

本轮 candidate 只运行 5 个对手 × 10 局 focused full-trace，结果为 31/19/0，胜率
62.0%，Meta 加权胜率 66.3%，第二回合 Powerful Hand 14.0%，post-KO 无 ready attacker
75.3%，对局级接力断档 60.0%；action error=0、空 Bench Run Away Draw=0。该样本与
iter-33 repeat 的 170 局不是同一范围，不能作严格 A/B。

50 局中实际解析到 67 次我方 Poffin effect：43 次选择包含 Dunsparce，24 次包含 Abra。
这证明新分支在 evaluator option 中可执行，但没有旧代码同局面对照，不能据此声称胜率
改善。fixture、171 个 unittest、`py_compile`、资产检查通过，`deck.csv` diff 为空。

### 决策

**observe，不晋升。** 保留这项语义修复，`BEST_STRATEGY.json` 不变；完整 trace 只保留
在外部评测目录的最新批次，本 repo 仅保留本轮文档和精炼指标。下一轮以真实的“Bench
Psychic/Telepath 目标存在但 Active 目标被选择”case 为单变量，必须保留“唯一 Psychic
应优先保证 Active 能攻击”的反例。

## AutoIter V26：未充能 Active 的 Bench Psychic handoff（iter-35）

### 改动与 advisor 结论

本轮固定 `deck.csv`，针对 iter-33 repeat forensic 核验的三个真实 case 做一个单变量
修正：当 Active 未充能，且当前 legal options 明确存在未充能 Bench Kadabra/Alakazam
（或有可见进化路线的 Abra）的 Basic Psychic/Telepath 附能目标时，优先把能量贴给 Bench，
让它成为下一回合接班者。没有真实 Bench successor 时仍把唯一 Psychic 留给 Active；已
充能 Active 的攻击优先级、Poffin gate 和终局规则均不变。

advisor 同时确认：`appearThisTurn` 会限制进化但不限制 Kadabra 接收能量或下一回合攻击；
Telepath 的附能加 Basic Psychic 检索是具体可执行路线。另一个 Fezandipiti/低奖赏 Bench
insurance case 本轮只记录，不混入候选。

### 评测前后

iter-35 与 iter-34 都是 5 个对手 × 10 局的独立 focused full-trace，不能逐局比较：

| 指标 | iter-34 focused | iter-35 focused |
|---|---:|---:|
| 胜 / 负 / 平 | 31 / 19 / 0 | 25 / 25 / 0 |
| 胜率 | 62.0% | 50.0% |
| Meta 加权胜率 | 66.3% | 51.0% |
| 第二回合 Powerful Hand | 14.0% | 10.0% |
| post-KO 无 ready attacker | 75.3% | 59.2% |
| 对局级接力断档 | 60.0% | 58.0% |
| 空 Bench Run Away Draw | 0 | 0 |
| 我方 action error | 0 | 0 |

本轮生产 helper 在 50 局中触发 32 次，32 次都实际选择了 Bench 能量目标；这证明了
策略分支真实执行。胜率和二回合指标没有提升，故不能晋升；post-KO 事件率有所改善，
但仍需更大批次和具体 case 判断是否值得保留。

### 决策

**observe，不晋升。** 保留本轮窄 gate 与 Active 反例 fixture，`BEST_STRATEGY.json`
不变。下一轮先检查二回合指标下降是否来自“过早把能量给 Bench”，并分析 Fezandipiti
与低奖赏 Dunsparce/Abra insurance 的先后顺序；不再扩大 Bench 优先条件。

## AutoIter V27：当前回合 Alakazam 攻击优先（iter-36）

### 改动与 advisor 结论

本轮固定 `deck.csv`，修复 iter-35 handoff gate 的边界：当 Active 已是 Alakazam，或
Active 在本回合附 Psychic 后能完成可见的 Alakazam 路线时，不能因为 Bench 存在未充能
Kadabra/Alakazam 就把唯一附能让给 Bench。Alakazam 即使是本回合刚由 Kadabra 进化，也
仍然可以在附能后攻击；`appearThisTurn` 只限制再次进化。

advisor 确认本轮不应混入另一个候选：Telepath 附到没有 Kadabra/Alakazam 或 Rare Candy
可见路线的 Bench Abra，并不能构成下一回合 ready attacker。该候选留待下一轮独立验证。

### 验收与评测

- 历史 forensic case：`crustle_wall/game_007` shared turn 3；旧策略会把 Psychic 让给
  Bench Kadabra，正确动作应是给本回合刚进化的 Active Alakazam 附能。
- 新增 fixture：`test_second_turn_alakazam_keeps_psychic_for_current_attack`；修复前选择
  Bench，修复后选择 Active。
- focused full-trace：`crustle_wall`、`crustle_v1`、`romanrozen_v9`、
  `kiyotah_dragapult`、`yanxiaohan` 各 10 局，共 50 局，结果为 31/19/0，胜率 62.0%，
  第二回合 22.0%。该批次只用于分支执行诊断。
- 完整 full-trace：17 个对手 × 10 局，共 170 局，结果为 105/64/1，另有 1 局对手侧
  `IndexError` 未完成；原始胜率 61.8%，Meta 加权胜率 63.0%，第二回合 Powerful Hand
  16.5%，post-KO 无 ready attacker 59.6%，对局级接力断档 39.4%。

| 指标 | iter-15 best full | iter-36 full |
|---|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 105 / 64 / 1 |
| 胜率 | 68.2% | 61.8% |
| Meta 加权胜率 | 70.4% | 63.0% |
| 第二回合 Powerful Hand | 27.1% | 16.5% |
| post-KO 无 ready attacker | 68.5% | 59.6% |
| 对局级接力断档 | 未记录 | 39.4% |
| 空 Bench Run Away Draw | 0 | 0 |
| 我方 action error | 0 | 0 |

两批是独立随机样本，不能作严格 A/B。完整矩阵的二回合指标仍低于 iter-15 best，说明
本轮只修复了“不要牺牲当前攻击”的局部边界，没有完成整体接力优化。

### 决策

**observe，不晋升。** 保留本轮窄 guard 和回归 fixture；`BEST_STRATEGY.json` 继续指向
iter-15 `patch-priority`。完整 trace 只保留在隔壁评测仓库的最新 iter-36 目录，本 repo
保留本轮 decision、advisor、analysis 和 metrics。下一轮单独验证 Telepath → unrouted
Abra gate，并继续关注 Bench 接力，不同时改动多个资源优先级。

## AutoIter V28：Telepath 的 routed Abra gate（iter-37）

### 改动与 advisor 结论

本轮固定 `deck.csv`，收窄 Telepath Energy 对 Bench Abra 的“确定接力”判定。Telepath
只能寻找 Basic Psychic Pokémon，不能直接找到 Kadabra/Alakazam；因此只有手牌有 Kadabra，
或有 Alakazam + Rare Candy 且没有 Budew 的 `Itchy Pollen` Item Lock 时，Telepath→Bench
Abra 才能进入 concrete handoff gate。Telepath→Bench Kadabra/Alakazam、Telepath→Active
和独立 Bench insurance 路径保持不变。

advisor 核对了 Telepath 的检索范围、每回合手动附能限制、`appearThisTurn` 的进化边界和
Itchy Pollen 对 Rare Candy 的限制，确认本轮不应修改全局 ready attacker 定义。

### replay case 验收

历史 iter-36 full trace 中的 4 个 case 已用当前策略重新执行，旧动作分别为
`[2]`/`[2]`/`[6]`/`[4]`，当前动作分别为 `[0]`/`[1]`/`[4]`/`[3]`，均不再把 Telepath
贴给没有可见进化路线的 Bench Abra；详细状态写入 `docs/reports/kaggle/alakazam-v7-auto-iter/iter-37/cases.jsonl`。

### 评测结果

本轮 full 为 17 个对手 × 10 局，共 170 局；与 iter-36、iter-15 均为独立随机批次：

| 指标 | iter-36 full | iter-37 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 105 / 64 / 1 | 117 / 50 / 3 | +12 胜 |
| 胜率 | 61.8% | 68.8% | +7.0pp |
| Meta 加权胜率 | 63.0% | 69.5% | +6.5pp |
| 第二回合 Powerful Hand | 16.5% | 27.6% | +11.1pp |
| post-KO 无 ready attacker | 59.6% | 59.0% | -0.6pp |
| 对局级接力断档 | 39.4% | 30.0% | -9.4pp |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

新 full trace 中有 99 次“Telepath→无可见路线 Bench Abra”选项暴露，其中 28 次由本轮明确
保留的独立 Bench insurance 路径选择该目标；concrete handoff gate 选择次数为 0。3 个未
完成样本均为 `yakitori_raging_bolt` 对手侧 `IndexError`。focused 40 局为 27/13/0，胜率
67.5%，第二回合 Powerful Hand 15.0%，action error=0。

### 决策

**接受为下一轮 control，暂不覆盖 immutable best-known。** 这轮修复了具体 case，且相对
iter-36 的全局指标方向均改善；但仍需在下一轮复核 Meta 加权胜率和接力断档，再决定是否
更新 `BEST_STRATEGY.json`。完整逐局 trace 只保留在隔壁评测仓库最新 iter-37 目录，本 repo
保留本轮 `decision.md`、`advisor.md`、`analysis.md`、`metrics.json` 和 `cases.jsonl`。

## AutoIter V29：broad Poffin Bench insurance gate（iter-38）

### 改动与 advisor 结论

本轮固定 `deck.csv`，尝试扩大 Poffin 的 Bench insurance 优先级：只要当前攻击前的
Abra-line 接力不够完整，就倾向先用 Poffin 建立更多 Basic。advisor 核验了 Poffin 只能
放下 Basic、不能附能或直接制造 ready attacker，并指出这个 gate 可能把“当前回合已经
能攻击”误判成必须先铺场。

### 评测结果

与 iter-37 使用同一套 17 个对手 × 10 局的 full-trace 口径，但为独立随机批次：

| 指标 | iter-37 control | iter-38 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 117 / 50 / 3 | 101 / 66 / 3 | -16 胜 |
| 胜率 | 68.8% | 59.4% | -9.4pp |
| Meta 加权胜率 | 69.5% | 61.2% | -8.3pp |
| 第二回合 Powerful Hand | 27.6% | 22.9% | -4.7pp |
| post-KO 无 ready attacker | 59.0% | 60.1% | +1.1pp |
| 对局级接力断档 | 30.0% | 38.2% | +8.2pp |
| 空 Bench Run Away Draw / 我方 action error | 0 / 0 | 0 / 0 | 不变 |

### 决策

**observe，不晋升。** broad gate 过宽，损害了攻击节奏和整体结果；`BEST_STRATEGY.json`
不变。完整 trace 的原始 label 曾因分析命令缺少日期后缀而被误识别，导致一份临时 0% 的
第二回合结果；该结果已废弃，正式结果由完整 role label 重跑得到。详细记录见
`docs/reports/kaggle/alakazam-v7-auto-iter/iter-38/`。

## AutoIter V30：narrow fresh-Bench second-turn gate（iter-39）

### 改动与 advisor 结论

本轮回退 broad gate，仅在 Bench 上所有 Abra-line 都是本回合新进场且未附 Psychic 时，才
允许 Poffin 在攻击前先建立更多 Basic；已有带能量的 Kadabra/Alakazam 或可确认的接力线
路继续让攻击优先。advisor 认为这个条件同时尊重用户要求的 Bench 连贯性和“攻击宣告后
回合立即结束”的规则，但提醒它只改善铺场观测量，不自动保证被 KO 后下一只打手 ready。

### 评测结果

| 指标 | iter-37 control | iter-39 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 117 / 50 / 3 | 114 / 55 / 1 | -3 胜 |
| 胜率 | 68.8% | 67.1% | -1.8pp |
| Meta 加权胜率 | 69.5% | 68.5% | -1.0pp |
| 第二回合 Powerful Hand | 27.6% | 30.0% | +2.4pp |
| post-KO 无 ready attacker | 59.0% | 64.6% | +5.6pp |
| 对局级接力断档 | 30.0% | 33.5% | +3.5pp |
| 空 Bench Run Away Draw / 我方 action error | 0 / 0 | 0 / 0 | 不变 |

### 决策

**observe，作为当前工作区候选但不更新 best-known。** 相比 iter-38，二回合 Powerful Hand
明显回升，证明收窄 gate 是正确方向；但相对 iter-37 的 post-KO 接力和 Meta 加权胜率
退化，因此不把它描述为整体提升，`BEST_STRATEGY.json` 仍指向 iter-15。下一轮优先从
具体击倒前的 Poffin/Basic/进化路线失败 action 中寻找最小 handoff 修正。

## AutoIter V31：Night Stretcher / Lana’s Aid recovery gate（iter-40）

### 改动

固定 `deck.csv`，只修正弃牌区恢复的资源优先级：Lana’s Aid 只有在能独立恢复当前可验证
的宝可梦和 Psychic Energy 路线时，才抑制 Night Stretcher。若弃牌区只有 Abra、手牌已有
Psychic Energy，则允许 Night Stretcher 先取回 Abra，建立下一只 Bench 打手。对应的
Nursrijan Lucario replay case 已加入策略回归 fixture；Abra 不攻击、Trading Places 禁用和
攻击结束回合等边界保持不变。

### 评测前后

iter-39 与 iter-40 都是 17 个对手 × 10 局的独立 full-trace 批次：

| 指标 | iter-39 control | iter-40 candidate |
|---|---:|---:|
| 胜 / 负 / 平 | 114 / 55 / 1 | 102 / 65 / 3 |
| 胜率 | 67.1% | 60.0% |
| Meta 加权胜率 | 68.5% | 61.0% |
| 第二回合 Powerful Hand | 30.0% | 23.5% |
| post-KO 无 ready attacker | 64.6% | 62.4% |
| 对局级接力断档 | 33.5% | 38.2% |
| 空 Bench Run Away Draw / 我方 action error | 0 / 0 | 0 / 0 |

第二回合数字来自完整 trace 的真实 role 自动推断；一次错误固定 label 产生的临时 0% 已
废弃。完整批次中 action error 为 0，Night/Lana fixture 通过，但整体胜率、Meta 和二回合
指标均下降，不能把独立随机批次当成严格因果证明。

### 采纳状态

**observe，不晋升。** 保留本轮局部修复和 advisor/decision 记录，但不更新
`BEST_STRATEGY.json`。下一轮先检查 Dragapult、Iono 和 Nursrijan 的具体 trace，确认恢复
动作是否在其它状态延迟了当前攻击或消耗了接力资源；只有同时改善 case、接力和主要胜率
信号，才考虑晋升。

## AutoIter V32：终局攻击优先于直接 Bench 附能（iter-41）

### 改动

固定 `deck.csv`，新增终局边界：当 Active Alakazam/Kadabra 已有 Psychic，当前攻击能
拿完最后奖赏时，Energy 附加不再抢占攻击前的最后行动窗口。非终局时的 Bench Kadabra/
Alakazam handoff 仍保持优先；Abra 不攻击、Trading Places 禁用和攻击结束回合等规则不变。
新增 fixture `test_final_prize_attack_precedes_direct_bench_energy_handoff`，证明旧逻辑会
先附能，新逻辑会直接攻击。

本轮同时修复了 AutoIter analyzer 的标签容错：如果命令传入的 `--agent-label` 不存在于
trace 的完整 role 中，自动回退到实际 role，避免生成虚假的 0% 第二回合指标；该工具修复
有独立回归测试，不改变策略动作。

### 评测前后

iter-40 与 iter-41 都是 17 个对手 × 10 局的独立 full-trace 批次：

| 指标 | iter-40 control | iter-41 candidate |
|---|---:|---:|
| 胜 / 负 / 平 | 102 / 65 / 3 | 102 / 66 / 2 |
| 胜率 | 60.0% | 60.0% |
| Meta 加权胜率 | 61.0% | 60.1% |
| 第二回合 Powerful Hand | 23.5% | 21.8% |
| post-KO 无 ready attacker | 62.4% (128/205) | 62.0% (119/192) |
| 对局级接力断档 | 38.2% | 31.8% |
| 空 Bench Run Away Draw / 我方 action error | 0 / 0 | 0 / 0 |

接力指标略有改善，但胜率不变，Meta 与第二回合指标下降；由于样本独立，不能将变化
全部归因于本轮改动，也不能宣称达到整体提升。

### 采纳状态

**observe，不晋升。** 保留规则 fixture 和 advisor 记录，不更新 `BEST_STRATEGY.json`。
下一轮审计真实 terminal closure case，确认奖赏数和伤害没有被误判，并定位二回合指标
下降是否来自本轮 Energy 低优先级；在此之前不放宽其它 gate。

## AutoIter V33：复核 handoff 误报并固定 trace retention（iter-42）

### 改动

固定 `deck.csv`，本轮没有修改 `main.py`。对 iter-41 的 12 个
`handoff_preparation_missed` 逐个回到 raw trace 核对实际 action、伤害和目标 Prize value：
它们都是终局攻击，或在 Dudunsparce 抽牌后形成即时 KO；没有确认的非终局 handoff 错误。

新增两个回归边界：抽牌能形成最后 KO 时保留抽牌优先级；抽牌不能形成即时 KO 时，具体的
Telepath Energy → Bench Alakazam handoff 才优先。与此同时，`alakazam_auto_iter.py run` 默认
改为 summary-only，只有最新一轮需要复盘时才显式传 `--save-traces`。

### 评测前后

iter-42 使用 iter-41 的同一批 17 个对手 × 10 局 full trace 重新生成轻量报告；策略代码未变，
因此指标严格相同：

| 指标 | iter-41 control | iter-42 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 102 / 66 / 2 | 102 / 66 / 2 | 0 |
| 胜率 | 60.0% | 60.0% | 0.0pp |
| Meta 加权胜率 | 60.1% | 60.1% | 0.0pp |
| 第二回合 Powerful Hand | 21.8% | 21.8% | 0.0pp |
| post-KO 无 ready attacker | 62.0% | 62.0% | 0.0pp |
| 对局级接力断档 | 31.8% | 31.8% | 0.0pp |
| 空 Bench Run Away Draw / 我方 action error | 0 / 0 | 0 / 0 | 不变 |

### 采纳状态

**observe，不晋升。** 接受 trace retention 工具约定，但不把错误 case 当作策略改进，
也不更新 `BEST_STRATEGY.json`。下一轮继续从真实的 post-KO 无 ready attacker 回溯到击倒
 前一回合，只有确认存在可执行的 Poffin、Telepath、Lana、Night Stretcher 或 Psychic 附能
 路线，才创建下一个策略 candidate。

## AutoIter V34：field option 解析修复后的 control 验证（iter-43）

### 本轮改动

固定 `deck.csv`，没有改变策略优先级。评测现场的 field option 会出现
`inPlayIndex: null`，实际目标位置在 `indexInArea`；`main.py` 统一通过
`_field_option_area()` / `_field_option_index()` 回退解析，修复合法 Active/Bench 目标
无法识别的问题。这是运行时合法性修复，不应伪装成新的策略收益。

### 结果

iter-43 为 17 个对手 × 10 局，共 170 个完整 trace：106/62/2，胜率 62.35%，Meta 加权
胜率 61.29%，第二回合 Powerful Hand 43/170 = 25.29%，post-KO 无 ready attacker
110/180 = 61.11%，对局级接力断档 58/170 = 34.12%，空 Bench Run Away Draw 0，我方
action error 0。

相对 iter-42 的独立批次为：胜率 +2.35pp，Meta +1.19pp，第二回合 +3.53pp，post-KO
事件率 -0.89pp，但对局级接力断档 +2.35pp。独立样本不能作严格因果 A/B。

### Advisor / case 结论

最新代表性复核没有确认新的策略 action miss：

- `kiyotah_dragapult/game_005` 的 post-KO 断档是击倒前资源不可得；
- `nursrijan_lucario/game_001` 受同回合连续进化规则限制；
- `kacchan_anti_wall/game_003` 已在攻击前执行 Poffin，属于 analyzer 边界误报。

详细轻量证据见 `docs/reports/kaggle/alakazam-v7-auto-iter/iter-43/` 的
`analysis.md`、`decision.md`、`advisor.md`、`metrics.json` 和 `cases.jsonl`。

### 采纳状态

**observe，不更新 `BEST_STRATEGY.json`。** 保留解析修复并继续审计真正的击倒前可执行
handoff case；旧完整 trace 只在关键 case 已落档后清理，最新完整 trace 作为下一轮 discovery
样本。

## AutoIter V35：终局 Fezandipiti 让位于 Powerful Hand（iter-44）

### 本轮改动

固定 `deck.csv`，当 Active Alakazam 的 `Powerful Hand` 已经能够击倒当前目标并拿完最后
奖赏时，直接提交攻击；不再让可选的 Fezandipiti ex `Flip the Script` 抢占攻击前的动作。
该规则保留了攻击宣告立即结束回合的语义：终局已经闭合时，后场准备不再产生收益。

其它边界不变：Abra 不攻击、Trading Places 禁用、Supporter/手动附能次数限制、Rare Candy
的 Budew `Itchy Pollen` 限制，以及 Mist/Rock Fighting Energy 的保护判断。

### 评测前后

iter-43 和 iter-44 均为 17 个对手 × 10 局的完整 trace，但属于独立随机批次，不能当作
严格逐局 A/B：

| 指标 | iter-43 control | iter-44 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 106 / 62 / 2 | 127 / 43 / 0 | +21 胜 |
| 胜率 | 62.35% | 74.71% | +12.35pp |
| Meta 加权胜率 | 61.29% | 74.65% | +13.36pp |
| 第二回合 Powerful Hand | 25.29% | 23.53% | -1.76pp |
| post-KO 无 ready attacker | 61.11% | 64.50% | +3.39pp |
| 对局级接力断档 | 34.12% | 34.71% | +0.59pp |
| 空 Bench Run Away Draw / 我方 action error | 0 / 0 | 0 / 0 | 不变 |

胜率信号向好，但第二回合和 post-KO 指标没有同步改善；不能把独立样本的胜率差异直接
归因于本轮改动。

### 具体 case 与采纳状态

`kacchan_anti_wall/game_010` raw[151] 证明旧策略在剩余 1 Prize、手牌 27 张、对手
340 HP Mega Lucario ex 时先用了 Fezandipiti，正确动作应是直接用 540 伤害的 Powerful
Hand 结束比赛。该 case 已修复。

**observe：作为下一轮探索 control，但暂不更新 immutable `BEST_STRATEGY.json`。**
下一轮处理 `zoli_dragapult/game_004` raw[75]：手里没有直接 Kadabra，但有 Poké Pad，
资源账本仍显示未知 Kadabra 来源时，应把 Telepath Energy 给 Bench Abra 的接力价值识别
出来。完整 trace 只保留在外部评测目录；repo 内保留轻量报告和 case 说明。

## AutoIter V36：Poké Pad 接力路线（iter-45）

### 改动

固定 `deck.csv`，让 Telepath Energy → Bench Abra 的路线在 Poké Pad 资源账本明确显示
Kadabra 来源时成立。Abra 仍不作为主动攻击者；没有可见进化来源时，Basic Psychic、
Telepath Energy 和 Wondrous Patch 不会被当作稳定接力。

### 评测结果

| 指标 | iter-44 control | iter-45 candidate |
|---|---:|---:|
| 胜 / 负 / 平 | 127 / 43 / 0 | 104 / 64 / 2 |
| 胜率 | 74.7% | 61.2% |
| Meta 加权胜率 | 74.7% | 61.6% |
| 第二回合 Powerful Hand | 23.5% | 22.9% |
| post-KO 无 ready attacker | 64.5% | 64.4% |
| 对局级接力断档 | 34.7% | 37.1% |

### 采纳状态

**observe，不晋升。** 目标 case 已通过回归验证，但独立批次的主要指标没有同步提升；
保留该局部修复，`BEST_STRATEGY.json` 继续保持 iter-15。

## AutoIter V37：诊断口径收窄（iter-46）

### 改动

固定 `deck.csv`，本轮只修改 analyzer，不改变生产策略：

- 没有合法 `Powerful Hand` option 的第二回合 case 记为 `unavailable`；
- Wondrous Patch 只有在目标为 Kadabra/Alakazam，或 Abra 有可见下一阶段来源时才记为
  Bench anchor；
- 用真实 raw trace 回溯 post-KO case，拒绝把 KO 后状态直接当成攻击前策略错误。

### 评测结果

| 指标 | iter-46 discovery |
|---|---:|
| 胜 / 负 / 平 | 109 / 59 / 2 |
| 胜率 | 64.1% |
| Meta 加权胜率 | 64.9% |
| 第二回合 Powerful Hand | 24.7%（42/170） |
| post-KO 无 ready attacker | 58.8%（110/187） |
| 对局级接力断档 | 37.6%（64/170） |
| 空 Bench Run Away Draw / 我方 action error | 0 / 0 |

诊断上，257 个 Bench insurance case 全部为合法进展，128 个第二回合 case 全部为
`unavailable`；本轮没有确认的新策略 action miss。

### 采纳状态

**observe，不晋升。** 这是评测口径修复，不是策略收益，不更新 `BEST_STRATEGY.json`。
下一轮从新的完整 trace 中寻找具有合法替代动作和后续收益链的最小策略改动。

## AutoIter V38：提前准备 Dunsparce 的 Enriching Energy（iter-47）

### 改动

固定 `deck.csv`。当场上有 Dunsparce、手牌有 Dudunsparce 与 Enriching Energy 时，即使
Dunsparce 是本回合刚放下、暂时不能进化，也允许先附 Enriching Energy，为下一回合自然
进化后的 Dudunsparce 过牌做准备。该规则不改变 Enriching Energy 不能给 Abra 线提供
Psychic 攻击能量的边界，也不允许 Abra 攻击。

对应真实 case：`crustle_v1/game_007 raw[9]`。

### 评测结果

| 批次 | 胜 / 负 / 平 | 胜率 | action error | trace 指标 |
|---|---:|---:|---:|---|
| iter-46 discovery baseline | 109 / 59 / 2 | 64.1% | 0 | 24.7% 二回合，58.8% post-KO 无 ready |
| iter-47 full batch 1 | 96 / 71 / 3 | 56.5% | 3（已知对手侧） | 不可用 |
| iter-47 full batch 2 | 104 / 66 / 2 | 61.2% | 2（已知对手侧） | 不可用 |

focused `crustle_v1` 10 局为 10W/0L/0D，但不足以证明全局收益。summary-only 评测不
保存 trace，第二回合和 post-KO 指标必须记为不可用，不能填 0。

### 采纳状态

**observe / 不晋升。** 真实局面和 fixture 支持这条规则在局部是正确的，但两批全矩阵
胜率均低于 iter-46；保留候选代码，下一轮先用有 trace 的 focused batch 验证进化铺场
和接力指标，再决定接受或回退。

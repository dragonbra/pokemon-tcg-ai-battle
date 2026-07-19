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
- outcome guardrail：磁盘受限时可以先运行 summary-only 批次，但只能比较胜负、平局、
  异常数量；没有完整 trace 时，第二回合和接力指标必须记为“不可用”，不能填 0
  或推断为失败。
- 每轮必须记录随机性、样本数、trace 覆盖数和是否使用同一批随机样本。独立批次的
  结果只能作为噪声下的 guardrail，不能把单次胜率变化直接解释为策略因果。

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
- `reports/kaggle/alakazam-v7-auto-iter/iter-00-baseline/`：当前实际可读取的 74
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

## AutoIter V2：待开始

### 迭代假设

待 V1 focused trace 验收后提出。一次只处理一个主要 case，不预先把打手接力、能量
分配和第二回合攻击同时改动。

### 结果

| 指标 | control | candidate | 变化 | 状态 |
|---|---:|---:|---:|---|
| 胜率 | — | — | — | 未开始 |
| 第二回合 Powerful Hand | — | — | — | 未开始 |
| post-KO 无 ready attacker | — | — | — | 未开始 |
| 空 Bench Run Away Draw | — | — | — | 未开始 |

## AutoIter V3：待开始

### 迭代假设

待 V2 的 case 和独立 outcome guardrail 完成后提出。若 V2 未能解决目标 case，先回退
或重写假设，不叠加新的策略变量。

### 结果

| 指标 | control | candidate | 变化 | 状态 |
|---|---:|---:|---:|---|
| 胜率 | — | — | — | 未开始 |
| 第二回合 Powerful Hand | — | — | — | 未开始 |
| post-KO 无 ready attacker | — | — | — | 未开始 |
| 空 Bench Run Away Draw | — | — | — | 未开始 |

## 后续追加规则

每完成一轮，在对应小节中追加：

1. 具体改动文件和策略假设；
2. control/candidate 的完整指标与样本口径；
3. 至少一个具体 case 的动作前后对照；
4. correctness、第二回合、接力和胜率 guardrail；
5. `accept`、`observe` 或 `reject` 决策及理由。

如果某轮只完成了 summary-only 评测，也必须明确写出“哪些指标不可用”，待有空间
后补跑完整 trace，不能用缺失值制造看似连续的提升曲线。

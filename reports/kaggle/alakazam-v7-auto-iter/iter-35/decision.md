# AutoIter iter-35 决策：未充能 Active 的 Bench Psychic handoff

## 1. 本轮尝试的改动

- 基线：iter-34 `routed-abra-pivot`。
- 改动：当 Active 是未充能的 Abra/Kadabra/Alakazam，且 legal options 中存在未充能
  Bench Kadabra/Alakazam（或有可见进化路线的 Abra）的 Basic Psychic/Telepath Energy
  目标时，优先把能量贴到 Bench，准备下一回合接班。
- 反例：如果不存在真实的 Bench 攻击线，唯一 Psychic 仍贴给 Active；已充能 Active
  的攻击优先级、终局闭环和 Poffin gate 不变。
- `deck.csv`、攻击终止规则、进化时序和 Supporter/手动附能次数均未修改。

## 2. 改动前后对比

本轮和 iter-34 是不同的 5×10 独立 focused 批次；以下数字用于观察，不是逐局 A/B。

| 指标 | iter-34 focused | iter-35 focused |
|---|---:|---:|
| 胜 / 负 / 平 | 31 / 19 / 0 | 25 / 25 / 0 |
| 胜率 | 62.0% | 50.0% |
| Meta 加权胜率 | 66.3% | 51.0% |
| 第二回合 Powerful Hand | 14.0% | 10.0% |
| post-KO 无 ready attacker | 75.3% | 59.2% |
| 对局级接力断档 | 60.0% | 58.0% |
| 我方 action error | 0 | 0 |
| 空 Bench Run Away Draw | 0 | 0 |

iter-35 的 50 局中，策略条件在 32 个真实 observation 触发，32 次均选择 Bench 能量
目标；没有触发时仍保留 Active 能量反例。触发计数来自 full trace 的实际 option，
不是 analyzer 的估算 case 数。

## 3. 验收与状态

- Basic Psychic、Telepath、Active 反例 fixture 均通过。
- 全量 unittest、语法和资产检查通过；`deck.csv` diff 为空。
- 状态：**observe，不晋升**。post-KO 无 ready event rate 和对局级断档略有改善，但
  胜率、Meta 和第二回合指标未改善，且 focused 样本很小。保留候选用于下一批观察，
  不更新 `BEST_STRATEGY.json`。

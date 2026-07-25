# AutoIter 21：收紧 Wondrous Patch 的 Abra gate

## 改动

- 相对 iter-20，只有当 Bench Abra 未附 Psychic 且手牌可见 Kadabra，或可见
  `Rare Candy + Alakazam` 路线时，Wondrous Patch 才能抢在攻击前执行。
- Bench Kadabra/Alakazam 的 Patch 路线保持不变。
- `deck.csv` 未修改；进化时机、Supporter/手动填能量次数、低牌库保护和终局攻击
  规则未修改。

具体 replay case：`penguin_915/game_010.json`，iter-20 的相关动作位于 trace 79--82。
原策略把 Patch 的 Psychic 贴给已经带 Psychic 的 Abra；iter-21 的 gate 不再把这类
“合法但没有新增接力价值”的目标排在攻击前。

## Case 验收

| 观察项 | iter-20 | iter-21 |
|---|---|---|
| 无可见下一阶段的 Bench Abra 被 Patch 抢先处理 | 允许 | 禁止，改走攻击/其它合法准备 |
| `bench_insurance_missed` 非 pass case | 需复核 | 0；244 条均为 analyzer `pass` |
| 我方 action error | 0 | 0 |

## 完整评测

两批都是 17 个对手 × 10 局的 full-trace，但不是逐局相同随机样本；下表只能作为
guardrail，不把胜率差直接归因于一个 gate。control 是当前已接受的 iter-15 immutable
artifact；candidate 是 iter-21 工作区。

| 指标 | iter-15 control | iter-21 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 105 / 63 / 2 | 下降 |
| 胜率 | 68.2% | 61.8% | -6.5pp |
| Meta 加权胜率 | 70.4% | 62.7% | -7.7pp |
| 第二回合 Powerful Hand | 46/170 (27.1%) | 36/170 (21.2%) | -5.9pp |
| post-KO 无 ready attacker | 124/181 (68.5%) | 133/191 (69.6%) | +1.1pp |
| 打手断档对局率 | 62/170 (36.5%) | 66/170 (38.8%) | +2.4pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

候选的原始评测数据保留在 `/tmp/alakazam-v7-auto-iter-21-full`；仓库只保存摘要，
避免把 170 局 JSON trace 纳入版本树。

## 决策：observe，不晋升

本轮目标 case 的局部行为得到修复，但整体胜率、Meta、第二回合攻击和接力断档率均
未达到晋升 guardrail。继续保留当前 `BEST_STRATEGY.json` 的 iter-15 immutable
artifact，Patch gate 作为工作区策略保留，下一轮不再重复这个变量。

下一轮候选是严格限定的 Poffin 目标实验：当 Dunsparce + Enriching Energy + 明确的
Active Alakazam 路线同时可见时验证是否能提升第二回合攻击；如果没有完整路线，必须
继续优先放 Abra，不能牺牲 Bench 接力。

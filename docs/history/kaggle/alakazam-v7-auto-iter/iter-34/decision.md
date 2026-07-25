# AutoIter iter-34 决策：Poffin 的 routed Abra gate

## 1. 本轮尝试的改动

- 基线：`iter-33-repeat-dudunsparce-stack-20260720`。
- 改动：Poffin 只有在存在显式 `Dunsparce + Enriching Energy` 抽牌路线，或已经存在
  可见的 Abra-line handoff 路线时，才允许把牌库槽位优先给 Dunsparce。
- 修复目标：两个 Bench Abra 虽然各自带 Psychic，但手上没有 Kadabra、Alakazam 或
  Rare Candy 进化路线时，不能把它们当成 ready attacker，从而错误地用 Poffin 找
  Dunsparce。
- 不变：`deck.csv`、Rare Candy 时序、Item Lock、Supporter/手动附能次数、攻击终止
  规则、Mist/Rock Fighting Energy 伤害保护。

## 2. 改动前后对比

本轮没有运行同一批 17×10 的完整 A/B；candidate 是 5 个对手 × 10 局的 focused
批次，因此不能把胜率变化解释为因果提升。

| 指标 | iter-33 repeat 基线 | iter-34 focused candidate |
|---|---:|---:|
| 样本 | 17 对手 × 10 局 | 5 对手 × 10 局 |
| 胜 / 负 / 平 | 104 / 65 / 1 | 31 / 19 / 0 |
| 胜率 | 61.2% | 62.0% |
| Meta 加权胜率 | 61.3% | 66.3% |
| 第二回合 Powerful Hand | 25.9% | 14.0% |
| post-KO 无 ready attacker | 122/177 = 68.9% | 61/81 = 75.3% |
| 对局级接力断档 | 58/170 = 34.1% | 30/50 = 60.0% |
| 我方 action error | 0 | 0 |
| 空 Bench Run Away Draw | 0 | 0 |

focused 批次中共解析到 67 次我方 Poffin effect：43 次选择中包含 Dunsparce，24 次
包含 Abra。该统计证明新分支在真实 evaluator option 中可执行，但没有旧版本同局面的
对照，不能单独证明选择分布就是改动造成的。

## 3. 验收与状态

- 新增的“带 Psychic 但无可见进化路线的 Bench Abra 仍优先找 Abra” fixture 通过。
- 相关策略测试、全量 unittest、语法和资产检查均通过；`deck.csv` diff 为空。
- 本轮状态：**observe，不晋升**。`BEST_STRATEGY.json` 保持原稳定 best，不把 focused
  样本当成新的胜率基线。

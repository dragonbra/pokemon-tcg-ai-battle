# AutoIter iter-33 决策：Dudunsparce 进化堆叠牌库估算修复

## 1. 本轮尝试的改动

- 基线：`iter-32-discard-index`。
- 改动：`_dudunsparce_deck_delta()` 统计被 Run Away Draw 洗回的
  `Dudunsparce + preEvolution + attached cards`，并优先按实际 option 定位目标实例。
- 目的：让牌库安全线判断符合真实卡牌效果，避免错误阻止合法抽牌或错误估算牌库风险。
- 不变：`deck.csv`、Bench 连贯性 gate、Rare Candy 规则、攻击优先级和保护能量判断。

## 2. 改动前后对比

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

## 3. 验收与状态

- `preEvolution` 回归 fixture：通过。
- 两批 full-trace：均为 170 局，action error=0，空 Bench Run Away Draw=0。
- 两批独立样本都显示二回合指标改善，但主胜率和 post-KO 接力没有稳定改善。

**决策：observe，不晋升。** 这是必须保留的规则正确性修复，但暂不更新稳定 best；下一
轮继续做单变量、case 驱动的接力分析。

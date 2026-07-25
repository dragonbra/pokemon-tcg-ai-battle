# AutoIter iter-43 决策：修复 field option 解析，保持策略观察

## 1. 本轮尝试

- Control：iter-42 / 当前 V7 AutoIter 策略。
- 本轮：不改变 `deck.csv`，不改变策略优先级；修复 field option 在 `inPlayIndex` 为
  `null` 时使用 `indexInArea` 的运行时解析回退。
- 目标：消除合法 Active/Bench 目标无法被识别造成的策略假象，并重新获得完整 trace 的
  可复核指标。

## 2. 前后对比

本轮和 iter-42 是独立随机批次，不能视为严格 paired A/B：

| 指标 | iter-42 | iter-43 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 102 / 66 / 2 | 106 / 62 / 2 | +4 胜 |
| 胜率 | 60.00% | 62.35% | +2.35pp |
| Meta 加权胜率 | 60.10% | 61.29% | +1.19pp |
| 第二回合 Powerful Hand | 21.76% | 25.29% | +3.53pp |
| post-KO 无 ready attacker | 62.00% | 61.11% | -0.89pp |
| 对局级接力断档 | 31.76% | 34.12% | +2.35pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

## 3. 采纳状态

**observe，不晋升为新的策略 best。** 解析修复应保留，它改善了运行时合法性；但本轮没有
经过同随机样本的策略 A/B，也没有确认新的可执行策略遗漏。`BEST_STRATEGY.json` 不更新。

## 4. 下一轮验收边界

只接受满足以下条件的策略 candidate：

1. raw trace 能证明候选动作在击倒前回合合法且可执行；
2. 当前攻击不是终局 closure，也不是抽牌后立即形成 KO；
3. candidate 的实际动作与预期动作不同，并能解释后续接力收益；
4. 不回退空 Bench `Run Away Draw = 0`、action error = 0 和第二回合指标。

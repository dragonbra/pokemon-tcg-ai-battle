# AutoIter V1 决策：空 Bench Run Away Draw

## 本轮改动

相对 V7 AutoIter 基线（iter-00），禁止唯一 Active Dudunsparce 在 Bench 为空时
使用 `Run Away Draw`；同时修正 field option 的 Active 目标解析。固定 `deck.csv`，
不改变卡组数量。

## 评测前后

口径：17 个对手 × 10 局，完整 trace；control 与 candidate 为独立随机批次。

| 指标 | iter-00 control | V1 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 101 / 67 / 2 | 105 / 65 / 0 | 胜 +4 |
| 胜率 | 59.4% | 61.8% | +2.4pp |
| Meta 加权胜率 | 59.1% | 61.3% | +2.2pp |
| 第二回合 Powerful Hand | 23.5% | 20.0% | -3.5pp |
| post-KO 无 ready attacker | 66.1% (119/180) | 68.8% (117/170) | +2.7pp |
| 打手断档对局率 | 34.7% | 37.1% | +2.4pp |
| 空 Bench Run Away Draw | 4 | 0 | -4 |
| evaluator error | 2 | 0 | -2 |

## 决策

- 状态：**observe**
- 目标 case：`True`；硬错误已消失。
- 不晋级原因：第二回合和接力指标出现回退，且本批为独立随机样本；需要后续
  focused trace 复核，不能只按胜率提升保留。

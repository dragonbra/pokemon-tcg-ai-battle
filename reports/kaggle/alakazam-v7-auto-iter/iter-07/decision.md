# AutoIter iter-07 决策：Rock Fighting Energy 属性边界

## 改动

相对 `iter-06 candidate`，固定 `deck.csv`，把 Rock Fighting Energy 的保护范围从
“所有带 ID 20 的目标”收紧为“Fighting 属性目标”；Mist Energy 行为不变。属性优先
读取提交包内同版本引擎卡表，未知类型保守视为可能受保护。

## evaluate 前后

两批都是 17 个对手 × 10 局的独立随机 trace，不能逐局 A/B。

| 指标 | iter-06 candidate | iter-07 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 104 / 64 / 2 | 97 / 71 / 2 | -7 胜 |
| 胜率 | 61.2% | 57.1% | -4.1pp |
| Meta 加权胜率 | 62.5% | 57.2% | -5.3pp |
| 第二回合 Powerful Hand | 25.3% | 20.0% | -5.3pp |
| post-KO 无 ready attacker | 61.2% | 78.9% | +17.7pp |
| 打手断档对局率 | 33.5% | 42.4% | +8.9pp |
| 我方 action error | 0 | 0 | 持平 |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |

## 决定

**observe**，不晋级为稳定 control。属性边界是规则正确性修复，不能因独立随机批次
的整体回退就撤销；继续用后续 trace 验证，并优先处理接力断档。

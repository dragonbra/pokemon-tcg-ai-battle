# AutoIter V3 决策：恢复路线不把孤立 Stage 1/Stage 2 当作 Basic 来源

## 本轮改动

相对 iter-02 control，`_recovery_needs()` 只把手牌中的 Abra 视为恢复路线的 Basic
来源；孤立的 Kadabra/Alakazam 不能直接替代 Basic Abra。固定 `deck.csv`，不改变
Enriching Energy 规则。

## 评测前后

口径：17 个对手 × 10 局，完整 trace；control 与 candidate 为独立随机批次。

| 指标 | iter-02 control | V3 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 91 / 78 / 1 | 102 / 68 / 0 | 胜 +11 |
| 胜率 | 53.5% | 60.0% | +6.5pp |
| Meta 加权胜率 | 51.7% | 59.4% | +7.7pp |
| 第二回合 Powerful Hand | 25.3% | 25.3% | 持平 |
| post-KO 无 ready attacker | 72.9% (129/177) | 70.6% (125/177) | -2.3pp |
| 打手断档对局率 | 37.1% | 37.6% | +0.6pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| evaluator error | 1 | 0 | -1 |

## 决策

- 状态：**observe**，V3 作为下一轮策略分析的 control。
- 目标 case：原比较文件为 `False`，原因是当时分析器尚未识别该恢复 case；fixture
  已验证新规则，不能把这个机械标记当成策略失败。
- 保留理由：本批胜率、Meta 和 post-KO 事件率均改善；接力断档对局率基本持平，
  仍需继续从具体 trace 找到下一项改动。

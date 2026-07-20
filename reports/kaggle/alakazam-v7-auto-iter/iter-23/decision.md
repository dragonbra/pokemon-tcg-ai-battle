# AutoIter 23：Lana's Aid 回收弃牌区接力资源

## 改动

- 增加严格的弃牌区接力条件：Active 已充能、Bench 无 Abra-line、Bench 有空位、弃牌
  区有 Abra+Basic Psychic，并且当前不是终局。
- 满足条件时，Lana's Aid 排在非终局攻击前；选择阶段明确取回 Abra 与 Basic Psychic。
- 增加终局攻击回归测试；`deck.csv` 未修改。

## Case 验收

新增 fixture 覆盖完整的第一步和 Lana 选择结果：主动作选择 Lana's Aid，取回集合为
`{Abra, Basic Psychic}`。另一个 fixture 确认最后奖赏闭环仍选择 Powerful Hand。

## 完整评测

candidate 为独立随机 17×10 full-trace；control 为 iter-15 immutable best。

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

## 决策：observe，不晋升

接力指标方向改善，但胜率、Meta 和第二回合指标仍明显低于当前 best；不能仅凭局部
post-KO 改善晋升。保留该策略作为工作区候选，继续分析第二回合与 analyzer 误报，
`BEST_STRATEGY.json` 仍指向 iter-15 immutable artifact。

# AutoIter iter-08 决策：Active Kadabra 接力能量

## 改动

相对 `iter-07 candidate`，固定 `deck.csv`，将 Psychic 接力准备 gate 扩展到已带 Psychic
的 Active Kadabra，并修复 Telepath 在 Active 与 Bench 均可选时优先贴回 Active 的策略
选择。新增失败 fixture 后先确认旧策略失败，再实现最小修改。

## evaluate 前后

两批都是 17 个对手 × 10 局的独立随机 trace；不能逐局归因。

| 指标 | iter-07 candidate | iter-08 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 97 / 71 / 2 | 101 / 68 / 1 | +4 胜 |
| 胜率 | 57.1% | 59.4% | +2.3pp |
| Meta 加权胜率 | 57.2% | 59.6% | +2.4pp |
| 第二回合 Powerful Hand | 20.0% | 17.1% | -2.9pp |
| post-KO 无 ready attacker | 78.9% | 70.9% | -8.0pp |
| 打手断档对局率 | 42.4% | 38.2% | -4.2pp |
| 我方 action error | 0 | 0 | 持平 |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |

## 决定

**observe**，不晋级为稳定 control。目标 fixture 已改善，post-KO 指标方向也改善，
但第二回合指标回退，整体仍低于 iter-06 独立 control；下一轮继续围绕具体断档 case 做
单变量分析，不直接宣称胜率提升。

# Iter-20 决策：observe

## 本轮改动

- Bench Abra 即使暂时看不到 Kadabra，也允许在非终局攻击前使用 Wondrous Patch、Basic
  Psychic 或 Lana's Aid 进行能量准备。
- 保留最终奖赏闭环、攻击终止和牌库保护规则；不修改 `deck.csv`。

## Case 验收

`penguin_915/game_010.json`、trace[98]：

- control action `[5]`：选择普通手牌动作，未使用可见的 Wondrous Patch；
- candidate action `[4]`：选择 Wondrous Patch，把弃牌区 Basic Psychic 准备给 Bench Abra；
- 该局面不是最后奖赏闭环，且本回合不要求 Abra 立即进化。

## 完整评测

control 为 iter-15 的 17×10 full-trace 批次；candidate 为独立随机的 iter-20 17×10
full-trace 批次。

| 指标 | iter-15 control | iter-20 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 103 / 64 / 3 | 胜率下降 |
| 胜率 | 68.2% | 60.6% | -7.6pp |
| Meta 加权胜率 | 70.4% | 60.2% | -10.2pp |
| 第二回合 Powerful Hand | 27.1% | 28.2% | +1.2pp |
| post-KO 无 ready attacker | 68.5% | 69.0% | +0.5pp |
| 打手断档对局率 | 36.5% | 40.0% | +3.5pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

本批原始 summary 有 3 个 `yakitori_raging_bolt` evaluator error；均为对手 role，
分析器未计入我方 action error。

## 决策

`observe`，不更新 best。目标 case 与 analyzer 中的本轮 Bench insurance fail 已修复，
但总体胜率、Meta 和接力断档率恶化，说明放宽条件的收益不能直接泛化到所有 Abra。当前
定时任务仍提交 iter-15 immutable artifact；下一轮应优先按具体资源来源拆分 gate，而不是
继续扩大触发范围。

# AutoIter 27：确定 Boss KO 压过第二回合 Powerful Hand

## 1. 本轮改动

- 新增 `second_turn_boss_prize_route` 判断。
- 当 Boss's Orders 能把对手 Bench 的可确定 KO 目标拉到 Active 时，第二回合
  Powerful Hand 不再越过这条 Supporter + 攻击路线。
- 终局和已有 Bench 接力边界保持不变；`deck.csv` 未修改。

## 2. Case 验收

新增 fixture 验证：Active Alakazam 已充能、Bench 有 Kadabra、第二回合同时存在 Boss
和 Powerful Hand 合法选项，且 Boss 可 KO 对方 Bench 时，首选 Boss。原实现选
Powerful Hand，改动后选 Boss；终局 Boss/Poffin 边界测试继续通过。

## 3. 评测前后

候选、iter-26 与 iter-15 都是独立随机批次，不能当作严格 A/B。当前定时提交 best 仍为
`iter-15-patch-priority`。

| 指标 | iter-15 best | iter-26 candidate | iter-27 candidate |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 111 / 59 / 0 | 109 / 59 / 2 |
| 胜率 | 68.2% | 65.3% | 64.1% |
| Meta 加权胜率 | 70.4% | 64.8% | 65.0% |
| 第二回合 Powerful Hand | 27.1% | 19.4% | 24.7% |
| post-KO 无 ready attacker | 68.5% | 69.8% | 71.5% |
| 对局级打手断档 | 36.5% | 38.2% | 41.2% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

`yakitori_raging_bolt` 有 2 局对手侧 `IndexError`，记录为 opponent error，不计入
我方 action error；这两局导致 draw，需在后续评测中单独观察。

## 4. 决策

**observe，不晋升。** 本轮相对 iter-26 提高了第二回合 Powerful Hand 和 Meta，但胜率
下降，post-KO 与对局级接力断档变差；相对 iter-15 best 的主要指标仍未达标。因此
`BEST_STRATEGY.json` 不变，08:05 仍提交 iter-15 immutable archive。下一轮继续关注
“被击倒后 ready attacker 接不上”的真实 case，而不是继续追逐单一第二回合比例。

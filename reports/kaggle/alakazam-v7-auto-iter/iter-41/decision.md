# AutoIter iter-41 决策：终局攻击优先于直接 Bench 附能

## 1. 本轮改动

- Control：iter-40 的 Night Stretcher / Lana’s Aid gate。
- Candidate：当 `_v7_terminal_prize_closure` 已确认 Active 当前攻击可以拿完最后奖赏时，把所有可选 Energy 附加放到攻击之后的优先级；不再为了给 Bench Kadabra/Alakazam 准备接班而延迟已经能结束对局的攻击。
- 其它策略不变：非终局时仍优先处理可验证的 Bench handoff；`deck.csv`、Abra 不攻击、Trading Places 禁用、Supporter/手动附能次数和保护能量规则均未改变。

## 2. 具体验收 case

新增 fixture：`test_final_prize_attack_precedes_direct_bench_energy_handoff`。

局面是 Active Alakazam 已带 Psychic、Bench 有未充能 Kadabra、手牌有 Basic Psychic，且当前 Powerful Hand 已足以击倒对手并拿最后奖赏。旧逻辑会先选择给 Bench 附能；V7 新逻辑选择攻击。这个边界符合“宣告攻击立即结束回合”和“终局不再需要下一只打手”的规则。

独立 advisor 复核了 `_v7_terminal_prize_closure` 的 Active、伤害和奖赏数判断，认为规则层面可接受，但要求继续用真实 trace 审计是否存在误判终局。

## 3. 完整评测前后

iter-40 与 iter-41 均为 17 个对手 × 10 局的独立 full-trace 批次，不能作逐局 A/B 因果结论。两轮均 0 个我方 action error。

| 指标 | iter-40 control | iter-41 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 102 / 65 / 3 | 102 / 66 / 2 | 胜数不变 |
| 胜率 | 60.0% | 60.0% | 0.0pp |
| Meta 加权胜率 | 61.0% | 60.1% | -0.9pp |
| 第二回合 Powerful Hand | 23.5% | 21.8% | -1.8pp |
| post-KO 无 ready attacker | 62.4% (128/205) | 62.0% (119/192) | -0.4pp |
| 对局级接力断档 | 38.2% | 31.8% | -6.5pp |
| 空 Bench Run Away Draw | 0 | 0 | 不变 |
| 我方 action error | 0 | 0 | 不变 |

接力事件方向略好，但胜率没有提升，Meta 和第二回合指标下降；独立随机样本不能证明这些变化全部由本轮改动造成。因此不能宣称整体提升。

## 4. 决策

**observe，不晋升。**

保留终局 fixture 和实现作为候选证据，不更新 `BEST_STRATEGY.json`。下一轮重点审计 closure 为真且 Bench 有未充能 Kadabra/Alakazam 的真实 trace，确认没有把非终局攻击、伤害不足或奖赏数误判成终局；同时定位二回合 Powerful Hand 从 23.5% 降到 21.8% 的具体动作来源。

完整逐局 trace 只保留在隔壁评测仓库的最新 iter-41；当前 repo 保留本报告和轻量摘要。

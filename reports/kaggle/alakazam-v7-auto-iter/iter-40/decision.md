# AutoIter iter-40 决策：Night Stretcher / Lana’s Aid gate 观察

## 1. 本轮改动

- Control：iter-39 的 narrow fresh-Bench gate。
- Candidate：修正恢复资源优先级。只有当 Lana’s Aid 能从弃牌区同时补齐当前可验证的宝可梦与 Psychic Energy 路线时，才让 Lana 抑制 Night Stretcher；如果弃牌区没有 Basic Psychic，或 Lana 不能独立完成恢复链，则允许 Night Stretcher 先取回 Abra，再使用手牌中的 Psychic Energy 建立 Bench 接力。
- 固定不变：`deck.csv`、Abra 不攻击、Trading Places 禁用、攻击宣告立即结束回合、Supporter/手动附能次数、Rare Candy 的 Budew `Itchy Pollen` 限制，以及 Mist/Rock Fighting Energy 的伤害保护语义。

## 2. 具体验收 case

来源：iter-39 的 `nursrijan_lucario/game_008`。当时 Active Alakazam 已带 Psychic，Bench 只有 Dudunsparce，弃牌区有 Abra，手牌有 Night Stretcher、Telepath Energy、Lana’s Aid，但弃牌区没有 Basic Psychic。

旧逻辑只要看到 Lana’s Aid 就排除 Night Stretcher，无法建立明确的下一只 Bench 打手。新逻辑要求 Lana 必须能独立完成恢复链；在这个状态下应保留并使用 Night Stretcher。对应回归 fixture 已加入 `tests/test_alakazam_v7_auto_iter_strategy.py`，并通过 60 个策略测试。

## 3. 完整评测前后

本轮与 iter-39 都是 17 个对手 × 10 局的独立 full-trace 批次，不能作逐局 A/B 因果结论。指标来自 170 个带完整 trace 的 `game_*.json`；第二回合统计使用 trace 中自动推断的真实 agent role，未使用错误的固定 label。

| 指标 | iter-39 control | iter-40 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 114 / 55 / 1 | 102 / 65 / 3 | -12 胜 |
| 胜率 | 67.1% | 60.0% | -7.1pp |
| Meta 加权胜率 | 68.5% | 61.0% | -7.5pp |
| 第二回合 Powerful Hand | 30.0% | 23.5% | -6.5pp |
| post-KO 无 ready attacker | 64.6% | 62.4% | -2.2pp |
| 对局级接力断档 | 33.5% | 38.2% | +4.7pp |
| 空 Bench Run Away Draw | 0 | 0 | 不变 |
| 我方 action error | 0 | 0 | 不变 |

这是独立随机批次；post-KO 事件率略降不足以抵消胜率、Meta 和二回合指标的同步下降。当前没有证据证明这次 gate 的整体收益，不能把它晋升为最佳策略。

## 4. 决策

**observe，不晋升。**

本轮保留具体 Night/Lana 回归修复和 fixture，但不更新 `BEST_STRATEGY.json`。下一轮应先复核恢复动作是否在其它状态过度延迟了当前攻击或消耗了接力资源，重点检查 `kiyotah_dragapult`、`kiyotah_iono` 和 Nursrijan 的具体失败 trace；不要仅凭总胜率对恢复优先级继续扩大 gate。

完整逐局 trace 只保留在隔壁评测仓库的最新 iter-40；当前 repo 只保留本报告、指标、case 摘要和比较文件。

# Alakazam AutoIter iter-36 分析摘要

## 本轮改动

iter-35 的 Bench Psychic handoff 是有价值的接力规则，但它把“Active 当前无法攻击”
和“Active 只差一次附能即可攻击”混在了一起。iter-36 新增
`_active_can_attack_after_psychic_attachment()`：只有 Active 没有本回合可完成的
Alakazam 攻击路线时，才允许未充能 Active 把 Psychic 让给合法 Bench 接班者。

## 真实验收 case

在旧批次 `crustle_wall/game_007` 的 shared turn 3，我方第二回合状态为：

- Active：本回合刚由 Kadabra 进化的 Alakazam，尚未附能；
- Bench：三只未附能 Kadabra；
- 手牌：Psychic Energy；
- legal options：Psychic 可附 Active 或 Bench；
- 正确顺序：Psychic → Active，随后 Powerful Hand。

旧的 iter-35 gate 会把 Bench 目标排在 Active 前面。这个状态被固化为 unittest，保证
以后优化 Bench 连续性时不会再次牺牲当前回合的确定攻击。

## focused 诊断结果

本轮使用 evaluator 默认独立随机、swap 开启的以下 5 个对手，各 10 局：
`crustle_wall`、`crustle_v1`、`romanrozen_v9`、`kiyotah_dragapult`、`yanxiaohan`。

| 指标 | 结果 |
|---|---:|
| 对局 | 50 |
| 胜 / 负 / 平 | 31 / 19 / 0 |
| 胜率 | 62.0% |
| Meta 加权胜率 | 59.7% |
| 第二回合 Powerful Hand | 11/50 = 22.0% |
| 被击倒事件 | 54 |
| 击倒后无 ready attacker | 42/54 = 77.8% |
| 出现过接力断档的对局 | 19/50 = 38.0% |
| 我方 action error | 0 |
| 空 Bench Run Away Draw | 0 |

与 iter-35 的 50 局 focused 批次相比，二回合指标从 10.0% 上升到 22.0%，对局级接力
断档从 58.0% 降到 38.0%；但两批样本独立、对手集合不同，不能视为因果 A/B。post-KO
无 ready attacker 事件率本轮较高，说明当前修复只保护了“当前回合攻击”边界，没有解决
整体的打手资源接力问题。

## 下一轮候选

advisor 发现另一个需要单独验证的边界：Telepath Energy 给没有 Kadabra、Alakazam 或
Rare Candy 可见路线的 Bench Abra，并不能构成下一回合 ready attacker。下一轮可将其
作为独立 gate，且必须保留本轮的 Active 即时攻击反例，不与其他资源策略混合。

完整 trace 位于隔壁评测仓库的 iter-36 目录；主 repo 不复制逐局 JSON。

## 完整矩阵结果

随后使用同一候选、17 个对手 × 10 局重新运行完整矩阵，共 170 局。该批次独立随机，
并非与 iter-15 best 的逐局 A/B：

| 指标 | iter-36 full |
|---|---:|
| 胜 / 负 / 平 | 105 / 64 / 1 |
| 胜率 | 61.8% |
| Meta 加权胜率 | 63.0% |
| 第二回合 Powerful Hand | 28/170 = 16.5% |
| 被击倒事件 | 193 |
| 击倒后无 ready attacker | 115/193 = 59.6% |
| 出现过接力断档的对局 | 67/170 = 39.4% |
| 我方 action error | 0 |
| 空 Bench Run Away Draw | 0 |

`yakitori_raging_bolt/game_009` 有 1 个未完成样本，trace 最后一步的 `role` 是
`opponent`，属于对手侧 `IndexError`，不计为我方 action error。

完整 trace 位于：
`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-36-full-active-attack-guard-20260720/`。

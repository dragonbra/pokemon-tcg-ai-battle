# Iter-19 决策：observe

## 本轮改动

- 在非终局确定 KO 前，识别已带 Psychic 的 Bench Kadabra → Alakazam 合法进化路线，
  并把它放在 Active Powerful Hand 之前。
- 终局奖赏闭环、牌库保护、每回合手填能量和攻击终止语义不变。
- `deck.csv` 不变；iter-19 叠加在 iter-18 工作区候选之上，当前 best 仍是独立的
  iter-15 immutable artifact。

## Case 验收

`romanrozen_v9/game_003.json`、trace[50]：

| 版本 | 首步动作 | 解释 |
|---|---|---|
| control | `[10]` Powerful Hand | 直接 KO，未先进化已充能 Bench Kadabra |
| candidate | `[4]` | Bench Kadabra → Alakazam；避免 Active 被击倒后接力断档 |

该局面没有被泛化成“所有进化都先做”：candidate 只在合法、已充 Psychic、非终局
确定 KO 的完整条件同时满足时触发。

## 完整评测

control 为 iter-15 的 17×10 full-trace 批次；candidate 为独立随机的 17 个对手 × 10
局 full-trace 批次，不能视作逐局 A/B。

| 指标 | iter-15 control | iter-19 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 106 / 63 / 1 | 胜率下降 |
| 胜率 | 68.2% | 62.4% | -5.9pp |
| Meta 加权胜率 | 70.4% | 60.7% | -9.7pp |
| 第二回合 Powerful Hand | 27.1% | 28.2% | +1.2pp |
| post-KO 无 ready attacker | 68.5% | 62.8% | -5.7pp |
| 打手断档对局率 | 36.5% | 38.8% | +2.4pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

原始 candidate summary 有 1 个 `yakitori_raging_bolt/game_002` 的对手侧
`IndexError`；分析器未计入我方 action error。

## 决策与后续

`observe`，不覆盖 best。目标 case 的动作排序正确，且 post-KO 事件率下降，但本批
胜率、Meta 加权胜率和打手断档对局率没有同时改善。下一轮重点是从 62.4% 批次中抽取
具体败局，确认是新进化 gate 的副作用、资源消耗，还是独立随机噪声；在此之前不能把
iter-19 包交给 08:05 定时提交。

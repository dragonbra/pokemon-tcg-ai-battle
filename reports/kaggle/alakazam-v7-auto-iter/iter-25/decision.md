# AutoIter 25：已有 Stage 1/2 Bench 时的二回合攻击优先级

## 1. 本轮改动

- 新增 `POWERFUL_HAND_ATTACK = 1072`，避免把 Powerful Hand 的攻击 ID 写成未定义名。
- 当 Active Alakazam 已附 Psychic、当前属于自己的第二回合、合法选项包含 Powerful
  Hand，且 Bench 已有 Kadabra/Alakazam 时，不再因泛化 Poffin/Hilda setup 阻塞攻击。
- 直接 Psychic 附能、Active Kadabra 自然进化、空 Bench/只有 Dunsparce 的 Bench
  insurance、Item Lock 和终局攻击优先级均保持不变。
- `deck.csv` 未修改。

## 2. 评测前后

candidate 与 control 都是 17 个对手 × 10 局 full-trace 的独立样本。control 是
`iter-15-patch-priority` immutable best；iter-24 仅作为近邻候选参考。

| 指标 | iter-15 control | iter-24 candidate | iter-25 candidate |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 107 / 61 / 2 | 104 / 61 / 5 |
| 胜率 | 68.2% | 62.9% | 61.2% |
| Meta 加权胜率 | 70.4% | 63.7% | 63.0% |
| 第二回合 Powerful Hand | 27.1% | 21.8% | 23.5% |
| post-KO 无 ready attacker | 68.5% | 65.4% | 65.3% |
| 打手断档对局率 | 36.5% | 38.2% | 36.5% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

本轮在 raw trace 中观察到 38 次符合条件的“已有 Stage 1/2 Bench 且 Powerful Hand
可选时直接攻击”动作。这个局部验收通过，但整体胜率和 Meta 仍明显低于 best。

## 3. 决策

**observe，不晋升。** 二回合指标较 iter-24 上升 1.8pp，对局级断档率回到 36.5%，
但与 iter-15 best 比仍分别低 3.5pp 和 0pp，胜率低 7.0pp，Meta 低 7.4pp。当前
`BEST_STRATEGY.json` 继续提交 iter-15 immutable archive。

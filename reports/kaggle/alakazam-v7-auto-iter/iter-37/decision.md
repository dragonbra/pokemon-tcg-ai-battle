# AutoIter iter-37 决策：收窄 Telepath→Bench Abra 接力判定

## 1. 本轮尝试的改动

- 基线：iter-36 `active-attack-guard`。
- 改动：Telepath Energy 只有在 Bench 目标为 Kadabra/Alakazam，或目标为 Abra 且手牌
  可见 Kadabra，或可合法组成 `Alakazam + Rare Candy` 时，才把该选项视为确定的下一回合
  打手接力。
- 原因：Telepath 只能附能并检索 Basic Psychic Pokémon，不能凭空把一个没有进化路线的
  Bench Abra 变成下一回合可攻击的 Kadabra/Alakazam。
- 保持不变：Telepath→Active、Telepath→Bench Kadabra/Alakazam、独立的空 Bench 保险路径、
  Poffin、Rare Candy 时序、Supporter/手动附能次数、攻击终止规则、Mist/Rock Fighting
  Energy 保护规则和固定 `deck.csv`。

## 2. replay case 验收

历史 iter-36 trace 中，以下局面旧策略把 Telepath 贴给没有可见进化路线的 Bench Abra；
用当前策略重新喂入同一个 observation 后，均不再选择该 Bench Abra 选项：

| case | 旧动作 | 当前策略动作 | 结果 |
|---|---:|---:|---|
| `kiyotah_dragapult/game_006.json` trace 18 | `[2]` | `[0]` | Telepath 从 Bench Abra 改为 Active Abra |
| `kokinn_search/game_001.json` trace 28 | `[2]` | `[1]` | Telepath 从 Bench Abra 改为 Active Abra |
| `nursrijan_lucario/game_001.json` trace 24 | `[6]` | `[4]` | Telepath 从 Bench Abra 改为 Active Abra |
| `sue_alakazam/game_007.json` trace 46 | `[4]` | `[3]` | Telepath 从 Bench Abra 改为 Active Alakazam |

新的 170 局 trace 中共 99 次暴露了“Telepath→无可见路线 Bench Abra”选项；其中 28 次
最终动作仍选择了该目标，但它们来自本轮明确保留的独立 Bench insurance 路径，不是
concrete handoff gate。4 个 targeted replay case 均已不再走原来的错误 handoff 选择，不能
把所有 Bench Abra 资源动作都当成同一种错误。

## 3. 评测前后对比

| 指标 | iter-36 full | iter-37 full | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 105 / 64 / 1 | 117 / 50 / 3 | +12 胜 |
| 胜率 | 61.8% | 68.8% | +7.0pp |
| Meta 加权胜率 | 63.0% | 69.5% | +6.5pp |
| 第二回合 Powerful Hand | 16.5% | 27.6% | +11.1pp |
| post-KO 无 ready attacker | 59.6% | 59.0% | -0.6pp |
| 对局级接力断档 | 39.4% | 30.0% | -9.4pp |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

两批 full trace 使用 evaluator 默认独立随机性，不能解释为逐局因果 A/B；3 个未完成样本
均来自 `yakitori_raging_bolt` 对手侧 `IndexError`，不计为我方 action error。

## 4. 决策

**接受为下一轮 control，暂不覆盖 immutable best-known。** 本轮同时修复了明确 replay
case，且胜率、第二回合攻击率和对局级接力断档率均优于 iter-36 独立批次；但 Meta 加权
胜率仍略低于 iter-15 best 的 70.4%，需要下一轮继续复核后再更新
`BEST_STRATEGY.json`。`deck.csv` diff 为空。

完整 trace 只保留在隔壁评测仓库的最新 iter-37 目录；本 repo 只保存本摘要和精炼 case。

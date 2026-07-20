# AutoIter 31：Active Telepath 附能同时建立 Bench 锚点

## 1. 本轮尝试的改动

- 固定 `deck.csv`，只修改策略，不改卡组。
- 历史 replay `iter-27-boss-before-second-turn-attack-20260720/kiyotah_dragapult/game_002.json`
  的 shared turn 7 暴露：Active Kadabra 未充能，Bench 只有未充能 Alakazam，Bench 尚有空位，
  手牌同时有 Basic Psychic 与 Telepath Psychic。
- 原策略在两个 Active 附能 option 同分时选择 Basic Psychic；V31 让 Telepath 在该窄条件下
  优先：它既提供 Psychic Energy，又能触发“最多搜索两只 Basic Psychic Pokémon 放到 Bench”。
- 条件限定为 Active Abra-line 未充能、Bench 有空位且没有已充能的 Abra-line 接班人；ready
  handoff、满 Bench、普通 Basic 附能、Item Lock、终局和攻击排序保持不变。

## 2. 改动前后对比

### (a) 基线

- 行为基线：iter-30 的当前策略；稳定 best 仍为 `iter-15-patch-priority`。
- V31 candidate：`iter-31-telepath-active-anchor`。
- 两次 focused evaluator 都是独立随机批次，不是同一局面 A/B；下表只作信号，不作严格
  因果证明。

### (b) 指标

| 指标 | iter-15 full best | iter-30 focused | iter-31 focused |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 7 / 13 / 0 | 10 / 10 / 0 |
| 胜率 | 68.2% | 35.0% | 50.0% |
| Meta 加权胜率 | 70.4% | 34.5% | 48.2% |
| 第二回合 Powerful Hand | 27.1% | 10.0% | 20.0% |
| post-KO 无 ready attacker | 68.5% | 70.4% | 66.7% |
| 对局级打手断档 | 36.5% | 70.0% | 60.0% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

iter-31 的样本为 `kiyotah_dragapult` 与 `sue_alakazam` 各 10 局；完整 trace 仅保存于
`/tmp/alakazam-v7-auto-iter-31-focus.J3Nmxh/`，本 repo 只保存本目录的摘要。

## 3. 验收与结论

- 新增 replay-derived fixture 在生产改动前失败，改动后通过。
- 当前代码重新读取历史 turn 7 raw observation 后选择 Telepath Psychic（card 19），而不是
  Basic Psychic（card 5）。
- 相关 75 个 unittest、`check_assets.py`、`compileall` 和 `git diff --check` 通过，固定
  `deck.csv` 仍为 60 张。
- focused 指标方向较 iter-30 好，但独立随机且只有 20 局，不能宣称完整矩阵提升。

**决策：observe，不晋升。** 保留 V31 的最小策略改动用于后续验证，`BEST_STRATEGY.json`
继续指向 `iter-15-patch-priority`。下一轮优先独立分析 advisor 指出的 Night Stretcher
弃牌区 `indexInArea` 解析问题，不与本轮 Telepath 变量混合。

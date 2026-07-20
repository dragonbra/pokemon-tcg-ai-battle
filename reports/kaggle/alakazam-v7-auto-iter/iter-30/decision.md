# AutoIter 30：归一化 option 的 `indexInArea` 接力解析修复

## 1. 本轮尝试的改动

- 固定 `deck.csv`，没有改动卡组。
- 在 `submission/alakazam_v7_auto_iter/main.py` 的 `_option_card_id()` 中，
  `type=7` 手牌 option 和 `area=2` 手牌/附能 option 优先读取 `indexInArea`，兼容
  evaluator 报告归一化后的 option；没有该字段时仍回退到原始 `index`。
- 修正并补强回归测试：
  - normalized Energy option 的局部 option 下标与原始手牌下标分离时，能解析到正确的
    Telepath/Basic Psychic；
  - 手里虽然有回收、进化和能量，但观察中没有实际 Bench 目标时，不因理论资源跳过攻击。
- 没有修改 `Mist Energy`/`Rock Fighting Energy` 伤害保护、终局、牌库保护或 Bench gate
  的策略优先级。

## 2. 改动前后对比

### (a) 基线

- 前一版本：`iter-29-second-turn-existing-line`，稳定 best 仍为
  `iter-15-patch-priority`。
- 本轮 candidate：`iter-30-index-in-area-handoff`。
- 评测随机性：evaluator 默认独立随机；不是 control/candidate 同批 A/B。

### (b) 验证结果

本轮只运行了 `kiyotah_dragapult` 与 `sue_alakazam` 各 10 局的 focused full-trace，
因此不能把下表当作完整矩阵结果，也不能用它推断策略因果变化。

| 指标 | iter-30 focused（20 局） |
|---|---:|
| 胜 / 负 / 平 | 7 / 13 / 0 |
| 胜率 | 35.0% |
| Meta 加权胜率 | 34.5% |
| 第二回合 Powerful Hand | 2/20 = 10.0% |
| post-KO 无 ready attacker | 38/54 = 70.4% |
| 出现对局级打手断档 | 14/20 = 70.0% |
| 空 Bench Run Away Draw | 0 |
| 我方 action error | 0 |

完整 focused trace 仅写入 `/tmp/alakazam-v7-auto-iter-30-focus/`；本 repo 只保留
`metrics.json`、本决策和 advisor 摘要。

## 3. 具体 case 验收

- 历史目标 case：外部 `iter-27-boss-before-second-turn-attack-20260720/`
  的 `kiyotah_dragapult/game_002.json`，turn 7。该状态中 Active Alakazam 已有
  Psychic，Bench 有未充能 Alakazam；normalized report 同时暴露局部 `index` 和原始
  手牌 `indexInArea`。本地 fixture 验证当前 parser 可解析 Basic Psychic，并选择 Bench
  attachment，而不会把它误认为其他手牌卡。
- 本次 20 局 focused 批次没有复现完全相同的旧 replay 状态；因此该结果只能证明回归
  fixture 和合法性，不证明完整矩阵中的触发率已经提升。

## 4. 决策

**observe，不晋升。**

- fixture、全套 167 个测试、语法检查和 `check_assets.py` 均通过；
- focused 结果覆盖太小且独立随机，主指标明显低于当前 iter-15 best，不能宣称提升；
- `BEST_STRATEGY.json` 保持指向 immutable `iter-15-patch-priority`；
- 下一轮应继续从完整 trace 中找具体的 Bench 接力断档，而不是扩大本轮 parser 兼容逻辑。

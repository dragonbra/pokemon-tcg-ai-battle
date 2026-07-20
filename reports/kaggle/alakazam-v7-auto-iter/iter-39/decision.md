# AutoIter iter-39 决策：narrow fresh-Bench gate 保留观察

## 1. 本轮改动

- Control：iter-37 `Telepath→Abra` 收窄后的策略；iter-38 的 broad Poffin gate 不采纳。
- Candidate：只有当当前 Bench 上的 Abra-line 都是本回合刚进场、且都没有 Psychic 时，才允许 Poffin 在本回合攻击前建立更多 Basic；已有带能量的接力线路仍保持攻击优先。
- 固定：`deck.csv`、Abra 不攻击、Trading Places 禁用、Supporter/附能次数、攻击终止规则和保护能量规则。

## 2. 同口径结果

| 指标 | iter-37 control | iter-39 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 117 / 50 / 3 | 114 / 55 / 1 | -3 胜 |
| 胜率 | 68.8% | 67.1% | -1.8pp |
| Meta 加权胜率 | 69.5% | 68.5% | -1.0pp |
| 第二回合 Powerful Hand | 27.6% | 30.0% | +2.4pp |
| post-KO 无 ready attacker | 59.0% | 64.6% | +5.6pp |
| 对局级接力断档 | 30.0% | 33.5% | +3.5pp |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

两批均为 17 个对手 × 10 局独立 full-trace，不是逐局 A/B；本轮实际读取 170 个 `game_*.json`，使用 trace 中完整 role label `iter-39-fresh-bench-second-turn-20260720`。iter-39 相对 iter-37 的二回合指标改善明确，但后续接力事件率和 Meta 加权胜率没有同步改善。

## 3. 决策

**observe，作为当前工作区候选但不更新 best-known。** 这轮证明窄 gate 比 broad gate 更接近预期：第二回合 Powerful Hand 从 22.9% 回升到 30.0%，胜率也从 iter-38 的 59.4% 回升到 67.1%。但与 iter-37 control 相比，post-KO 接力退化，不能宣称整体提升，也不更新 `BEST_STRATEGY.json`（仍为 iter-15）。下一轮应从具体 post-KO 断档的击倒前 action 入手，寻找能够同时保留二回合攻击和后续 ready attacker 的最小改动。

完整逐局 trace 仍位于隔壁评测仓库；当前 repo 只保留本报告和精炼 case。

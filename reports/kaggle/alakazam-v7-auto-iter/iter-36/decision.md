# AutoIter iter-36 决策：当前回合 Alakazam 攻击优先

## 1. 本轮尝试的改动

- 基线：iter-35 `energy-handoff-target`。
- 改动：当我方第二回合及以后，Active 已是 Alakazam，或在本回合附上 Psychic 后能完成可见的 Alakazam 路线时，禁止 Bench handoff gate 抢走这次附能。
- 目标：修复“Active 刚进化成 Alakazam、Bench 有 Kadabra 时，把唯一 Psychic 给 Bench，错过本回合 Powerful Hand”的规则错误。
- 不变：`deck.csv`、Bench 接力规则、Poffin gate、Rare Candy 时序、Supporter/手动附能次数、攻击终止规则以及 Mist/Rock Fighting Energy 保护规则。

## 2. 改动前后对比

主结果为完整的 17 对手 × 10 局矩阵；它与 iter-15 best 的证据批次也是独立随机，不能作逐局 A/B。

| 指标 | iter-35 focused | iter-36 focused |
|---|---:|---:|
| 样本 | iter-15 best full | iter-36 full |
| 胜 / 负 / 平 | 116 / 52 / 2 | 105 / 64 / 1 |
| 胜率 | 68.2% | 61.8% |
| Meta 加权胜率 | 70.4% | 63.0% |
| 第二回合 Powerful Hand | 27.1% | 16.5% |
| post-KO 无 ready attacker | 68.5% | 59.6% |
| 对局级接力断档 | 未记录 | 39.4% |
| 我方 action error | 0 | 0 |
| 空 Bench Run Away Draw | 0 | 0 |

iter-36 另有一批 5 对手 × 10 局 focused 诊断：31/19/0、胜率 62.0%、第二回合 22.0%；
它同样是独立随机，仅用于确认新分支真实执行。完整矩阵的二回合指标仍低于 iter-15
best，不能声称本轮带来了全局提升。

## 3. 验收与状态

- 真实 forensic case：历史 `crustle_wall/game_007` shared turn 3；Active 是本回合刚进化的 Alakazam，手牌有 Psychic，Bench 有三只 Kadabra。预期先给 Active 附能，再使用 Powerful Hand。
- 新增回归 fixture：`test_second_turn_alakazam_keeps_psychic_for_current_attack`，修复前实际选 Bench，修复后选 Active。
- 175 个 unittest、`py_compile`、资产检查通过；`deck.csv` diff 为空。
- 状态：**observe，不晋升**。`BEST_STRATEGY.json` 仍指向 iter-15 `patch-priority`；本轮完整 trace 只保留在隔壁评测仓库的最新 iter-36 full 目录，本 repo 只保留本摘要。

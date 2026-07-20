# AutoIter iter-32 决策：弃牌区 `indexInArea` 解析修复

## 1. 本轮尝试的改动

- 基线：`iter-27-boss-before-second-turn-attack`。
- 改动：固定 `deck.csv`，仅在 `_option_card_id()` 中让 `area=3` 的弃牌区选项优先
  读取 `indexInArea`，无该字段时回退到原始 `index`。
- 目的：避免 Night Stretcher/Lana's Aid 选择阶段把局部 option 序号误当成弃牌区
  位置，错判回收资源，导致后续 Abra/Psychic 接力路线断裂。
- 不变：Poffin、Telepath、Rare Candy、Dudunsparce、Item Lock、终局攻击和牌库保护
  gate；不修改卡组。

## 2. 改动前后对比

| 指标 | iter-27 control | iter-32 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 109 / 59 / 2 | 118 / 51 / 1 | 胜 +9 |
| 胜率 | 64.1% | 69.4% | +5.3pp |
| Meta 加权胜率 | 65.0% | 70.1% | +5.1pp |
| 第二回合 Powerful Hand | 42/170 (24.7%) | 31/170 (18.2%) | -6.5pp |
| post-KO 无 ready attacker | 138/193 (71.5%) | 117/176 (66.5%) | -5.0pp |
| 对局级打手断档 | 70/170 (41.2%) | 57/170 (33.5%) | -7.7pp |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

两批均为 evaluator 默认独立随机，不能把胜率上升解释为单一修复的严格因果结果。
另一方面，第二回合指标明确回退，因此不能把本轮 candidate 直接标为新的稳定 best。

## 3. 验收与状态

- 新增 fixture 在修复前失败，修复后通过；专项 50 个 unittest、全量 169 个 unittest
  均通过。
- 完整 candidate：170 个 trace、我方 action error=0、空 Bench Run Away Draw=0。
- 最新批次没有实际触发 `area=3` 选择，所以 parser 的真实 recovery 动作链仍需后续
  trace 复核。

**决策：observe，不晋升。** 保留该窄解析修复进入下一轮实验，但暂不更新
`BEST_STRATEGY.json`；下一轮优先在不损害第二回合铺场的前提下，寻找可复现的接力断档
或 recovery 误选 case。

# AutoIter 22：Poffin 的 Enriching 目标 gate

## 改动

- 新增 `_poffin_dunsparce_draw_route()`。
- 只有“至少三只 Abra-line、无 Dunsparce、手中有 Enriching Energy、Active 有合法
  直接 Alakazam 路线”全部可见时，Poffin 搜索优先 Dunsparce。
- 其它情形仍优先 Abra；`deck.csv` 不变。

## Case 验收

新增 fixture：当 Active Abra 带 Psychic、Bench 还有 Abra/Kadabra、手中有
Alakazam+Rare Candy+Enriching 时，Poffin 选择 Dunsparce；缺少明确路线时既有 Abra
优先测试保持通过。

## 完整评测

candidate 为独立随机的 17×10 full-trace；control 为 iter-15 immutable best，不能
逐局归因。

| 指标 | iter-15 control | iter-22 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 102 / 67 / 1 | 下降 |
| 胜率 | 68.2% | 60.0% | -8.2pp |
| Meta 加权胜率 | 70.4% | 58.9% | -11.5pp |
| 第二回合 Powerful Hand | 27.1% | 22.4% | -4.7pp |
| post-KO 无 ready attacker | 68.5% | 71.5% | +3.0pp |
| 打手断档对局率 | 36.5% | 36.5% | 持平 |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

## 决策：observe，不晋升

局部 Poffin 选择符合设计，但没有带来整体指标提升，且胜率、Meta、第二回合和
post-KO 事件均劣于 best。`BEST_STRATEGY.json` 继续保持 iter-15；该 gate 只作为
下一轮工作区基础，不作为定时 Kaggle 提交版本。

# AutoIter 24：Night Stretcher 窄接力路径

## 1. 本轮改动

- 在 `submission/alakazam_v7_auto_iter/main.py` 增加窄条件
  `_discarded_abra_successor_stretcher_due()`。
- 只有 Active 已充能、Bench 没有 Abra-line、弃牌区有 Abra、手牌已有 Psychic、没有
  更优 Poffin/手牌 Abra/Lana 路径、非终局且非 Item Lock 时，Night Stretcher 才排在
  非终局攻击前。
- Night Stretcher 的选择阶段只取 Abra；不虚构它能同时恢复 Energy。
- `deck.csv` 未修改。
- 同时修正 analyzer 对 Lana's Aid 恢复接力的误报识别；这是测量修复，不计入策略收益。

## 2. 评测前后

candidate 与 control 均按 17 个对手 × 10 局观察；两批是独立随机样本，因此只作
guardrail，不作逐局因果结论。control 是 `iter-15-patch-priority` immutable best。

| 指标 | iter-15 control | iter-24 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 107 / 61 / 2 | 胜率下降 |
| 胜率 | 68.2% | 62.9% | -5.3pp |
| Meta 加权胜率 | 70.4% | 63.7% | -6.7pp |
| 第二回合 Powerful Hand | 27.1% | 21.8% | -5.3pp |
| post-KO 无 ready attacker | 68.5% | 65.4% | -3.1pp |
| 打手断档对局率 | 36.5% | 38.2% | +1.8pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

## 3. Case 验收与决策

- 单元 case：无更优 anchor 时先 Night Stretcher，随后只取回 Abra；最后奖赏仍先攻击；
  Budew Item Lock 下不使用 Night Stretcher；Lana's Aid 仍优先于 Night Stretcher。
- full-trace：窄路线实际触发 23 次，均选择正确，action error 为 0。
- 接力事件率局部改善，但对局级断档率反而上升，且胜率、Meta 和第二回合指标均低于
  当前 best。

**决策：observe，不晋升。** `BEST_STRATEGY.json` 继续指向 `iter-15-patch-priority`；
Night Stretcher 候选保留用于后续 case 分析，不覆盖定时提交版本。

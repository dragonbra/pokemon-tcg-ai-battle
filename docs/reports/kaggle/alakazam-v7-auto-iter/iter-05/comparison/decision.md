# AutoIter V5 决策：Active Alakazam 的 Bench 保险

## 本轮改动

前一版本是 `iter-04` candidate。本轮固定 `deck.csv`，只在
`Active = Alakazam`、`Bench = 空`、Poffin 可合法使用且手牌有 Abra/Dunsparce 时，
把 Poffin 提到普通攻击之前；若当前攻击能够完成最后奖赏闭环，则仍保留攻击优先。
同时把 `bench_insurance_missed` 纳入 trace analyzer，并修正 evaluator error 的归属：
对手最后行动造成的异常不计为我方 action error。

## 评测前后

17 个对手 × 10 局，完整 trace；control 与 candidate 为独立随机样本，不能逐局配对。

| 指标 | iter-04 control | V5 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 106 / 63 / 1 | 106 / 62 / 2 | 胜持平 |
| 胜率 | 62.4% | 62.4% | 持平 |
| Meta 加权胜率 | 62.3% | 63.7% | +1.4pp |
| 第二回合 Powerful Hand | 47/170 (27.6%) | 41/170 (24.1%) | -3.5pp |
| post-KO 无 ready attacker | 108/161 (67.1%) | 116/180 (64.4%) | -2.7pp |
| 打手断档对局率 | 60/170 (35.3%) | 56/170 (32.9%) | -2.4pp |
| Bench 保险 miss | 8/10 | 0/5 | 解决；触发数受独立样本影响 |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

原始 evaluator error 为 1 → 2，但对应 trace 的最后行动方均为对手，不能作为我方
策略退化证据。

## 独立 repeat

| 指标 | iter-04 repeat control | V5 repeat candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 106 / 63 / 1 | 97 / 72 / 1 | 胜 -9 |
| 胜率 | 62.4% | 57.1% | -5.3pp |
| Meta 加权胜率 | 61.8% | 57.8% | -3.9pp |
| 第二回合 Powerful Hand | 36/170 (21.2%) | 49/170 (28.8%) | +7.6pp |
| post-KO 无 ready attacker | 117/174 (67.2%) | 137/183 (74.9%) | +7.7pp |
| 打手断档对局率 | 68/170 (40.0%) | 63/170 (37.1%) | -2.9pp |
| Bench 保险 miss | 12/13 | 0/6 | 解决；触发数受独立样本影响 |
| 我方 action error | 0 | 0 | 持平 |

## 决策

- 状态：**observe**，不晋级为稳定 control。
- 目标 case：`True`；两批中所有触发局面均未再漏掉 Poffin。
- 保留原因：该规则直接对应宝可梦卡牌的 Bench 生存条件，且没有引入我方 action
  error。
- 暂不晋级原因：repeat 胜率、Meta 和 post-KO ready 指标回退；第一批第二回合指标
  也下降。下一轮先由 advisor 复核 Poffin 与当回合进化/能量/终局闭环的冲突，再决定
  是收窄触发条件还是调整更上游的铺场优先级。

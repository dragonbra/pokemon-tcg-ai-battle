# AutoIter V38 决策：提前准备 Dunsparce 的 Enriching Energy

## 本轮改动

固定 `deck.csv`。当场上有 Dunsparce、手牌有 Dudunsparce + Enriching Energy 时，即使
Dunsparce 是本回合刚放下、暂时不能进化，也允许先附 Enriching Energy，为下一回合进化
后的 Dudunsparce 过牌做准备。

## 评测前后

| 批次 | 胜 / 负 / 平 | 胜率 | action error | trace 指标 |
|---|---:|---:|---:|---|
| iter-46 discovery baseline | 109 / 59 / 2 | 64.1% | 0 | 24.7% 二回合，58.8% post-KO 无 ready |
| iter-47 full batch 1 | 96 / 71 / 3 | 56.5% | 3（已知对手侧） | 不可用 |
| iter-47 full batch 2 | 104 / 66 / 2 | 61.2% | 2（已知对手侧） | 不可用 |

`crustle_v1` focused 10 局为 10W/0L/0D，但不能代表全局因果收益。两批全矩阵均低于
iter-46，且没有保存完整 trace，因此不能判断第二回合或接力指标是否改善。

## 决策

**observe / 不晋升。** 保留真实 case、策略 fixture 和当前候选代码，但不更新
`BEST_STRATEGY.json`，不进行 Kaggle 提交。下一轮优先用有 trace 的 focused batch 验证
这条规则是否真正改善进化铺场，而不是继续扩大范围。

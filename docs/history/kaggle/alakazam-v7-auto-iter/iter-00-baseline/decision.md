# AutoIter iter-00 基线

## 本轮改动

无策略改动。`iter-00` 只固定 V7 AutoIter 的观察基线，未改变 `deck.csv`。

## 评测前后

这是起始版本，没有更早的 AutoIter iteration，因此“改动前”与“改动后”不适用。
后续每轮均以本目录或上一轮 candidate 作为 control，并记录 evaluate 的完整统计。

| 指标 | iter-00 baseline |
|---|---:|
| 样本 | 17 个对手 × 10 局 = 170 局 |
| 胜 / 负 / 平 | 106 / 64 / 0 |
| 胜率 | 62.4% |
| 第二回合 Powerful Hand | 46/170 = 27.1% |

本目录另有 74 个可读取 trace，只用于验证分析器口径，不能替代上面的完整批次，
也不能与后续独立随机批次逐局配对。

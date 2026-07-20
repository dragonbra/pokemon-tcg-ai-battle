# AutoIter V2 决策：Active Dunsparce 优先 Enriching Energy

## 本轮改动

相对 iter-01 control，在 Active Dunsparce 可合法进化、或 Active Dudunsparce 已在场
时，允许优先把 Enriching Energy 给 Dunsparce 抽牌引擎；没有该条件时保持 Abra 线
Psychic Energy 的优先级。沿用 V1 的空 Bench Run Away Draw 保护。

## 评测前后

口径：17 个对手 × 10 局，完整 trace；control 与 candidate 为独立随机批次。

| 指标 | iter-01 control | V2 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 115 / 53 / 2 | 103 / 67 / 0 | 胜 -12 |
| 胜率 | 67.6% | 60.6% | -7.1pp |
| Meta 加权胜率 | 70.5% | 58.8% | -11.7pp |
| 第二回合 Powerful Hand | 25.9% | 30.0% | +4.1pp |
| post-KO 无 ready attacker | 66.0% (103/156) | 73.5% (122/166) | +7.5pp |
| 打手断档对局率 | 33.5% | 37.6% | +4.1pp |
| 空 Bench Run Away Draw | 2 | 0 | -2 |
| evaluator error | 2 | 0 | -2 |

独立 repeat：胜率 55.3% → 57.1%，Meta 54.4% → 56.1%，第二回合 22.4% →
27.1%；但接力断档对局率 35.9% → 39.4%。

## 决策

- 状态：**observe**
- 目标 case：`True`；空 Bench 硬错误继续为 0。
- 不晋级原因：第一批胜率、Meta 和接力指标明显回退，repeat 虽方向较好仍不足以
  证明 Enriching 路线对整体接力稳定有益；iter-01 control 保留为稳妥参照。

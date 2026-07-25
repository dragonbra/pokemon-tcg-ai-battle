# AutoIter V4 决策：Active Alakazam 的可见接力线优先 Psychic 资源

## 本轮改动

相对 V3 control，若 Active Alakazam 已有 Psychic Energy 且 Bench 存在无能量的
Kadabra/Alakazam，同时当前选项暴露 Psychic 附能或 Lana's Aid 回收 Basic Psychic
的路径，则先完成接力准备；不先把唯一手动附能机会给 Dudunsparce 的 Enriching
Energy，也不在 Lana's Aid 路径可见时直接攻击。固定 `deck.csv`。

## 评测前后

两批均为 17 个对手 × 10 局完整 trace，control/candidate 为独立随机样本。

### 第一批

| 指标 | V3 control | V4 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 109 / 61 / 0 | 106 / 63 / 1 | 胜 -3 |
| 胜率 | 64.1% | 62.4% | -1.8pp |
| Meta 加权胜率 | 63.7% | 62.3% | -1.4pp |
| 第二回合 Powerful Hand | 21.8% | 27.6% | +5.9pp |
| post-KO 无 ready attacker | 72.5% (129/178) | 67.1% (108/161) | -5.4pp |
| 打手断档对局率 | 38.2% | 35.3% | -2.9pp |
| 接力准备遗漏 case | 12 | 0 | -12 |
| evaluator error | 0 | 1（对手侧） | 不计为我方 action error |

### 独立 repeat

| 指标 | V3 control | V4 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 96 / 72 / 2 | 106 / 63 / 1 | 胜 +10 |
| 胜率 | 56.5% | 62.4% | +5.9pp |
| Meta 加权胜率 | 55.9% | 61.8% | +5.8pp |
| 第二回合 Powerful Hand | 20.0% | 21.2% | +1.2pp |
| post-KO 无 ready attacker | 72.8% (139/191) | 67.2% (117/174) | -5.6pp |
| 打手断档对局率 | 40.6% | 40.0% | -0.6pp |
| 接力准备遗漏 case | 6 | 0 | -6 |
| evaluator error | 2 | 1 | 两批均为对手侧，不计为我方 action error |

## 决策

- 状态：**observe，不替换 V3 control**。
- 目标 case：`True`；两批均从 12/6 降到 0，且第二回合与 post-KO 指标没有回退。
- 不直接晋级原因：两批胜率方向不一致；第一批略降，repeat 明显上升。需要在
  下一轮保持该接力修复，并验证新的 Bench 保险规则，不能把单批随机差异解释成
  已达到稳定提升。

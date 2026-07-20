# AutoIter V36 决策：Poké Pad 接力路线

## 本轮改动

固定 `deck.csv`，在 Telepath Energy → Bench Abra 的判断中使用 Poké Pad 资源账本确认
Kadabra 来源。该改动只扩大一个有证据的接力路线，不允许孤立 Abra 作为攻击者，也不改变
终局、Poffin、Dudunsparce 或 Item Lock 优先级。

## 评测前后

control/candidate 各为 17 个对手 × 10 局的独立批次，不能作严格逐局 A/B：

| 指标 | iter-44 control | iter-45 candidate |
|---|---:|---:|
| 胜 / 负 / 平 | 127 / 43 / 0 | 104 / 64 / 2 |
| 胜率 | 74.7% | 61.2% |
| Meta 加权胜率 | 74.7% | 61.6% |
| 第二回合 Powerful Hand | 23.5% | 22.9% |
| post-KO 无 ready attacker | 64.5% | 64.4% |
| 对局级接力断档 | 34.7% | 37.1% |

目标 case 已通过回归验证，但整体指标没有形成可接受的同步提升。

## 决策

**observe，不晋升。** 保留代码和 case 文档作为后续分析依据，不更新
`BEST_STRATEGY.json`。

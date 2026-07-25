# AutoIter V6 决策：Bench 接力连续性与 Rock Fighting Energy

## 本轮改动

相对 iter-05 candidate，固定 `deck.csv`，本轮完成两类策略/分析修正：

- Bench 连续性不再只判断“是否有 Pokémon”：Dunsparce 只是缓冲/抽牌引擎，Abra 线
  需要能量和可见进化/接力路线。没有 ready handoff 时，优先 Poffin、Telepath 或
  直接放下手牌 Basic；已存在直接附能/进化路径时不再误报保险遗漏。
- Rock Fighting Energy（ID `20`）与 Mist Energy（ID `11`）统一视为会阻挡
  Alakazam `Powerful Hand` 的保护能量；Enhanced Hammer、攻击伤害、Boss KO 和终局
  判断都复用该模型。`deck.csv` 未修改。

## evaluate 前后

17 个对手 × 10 局，完整 trace；iter-05 candidate 与 iter-06 是独立随机批次，不能
逐局归因。当前批次原始 evaluator 有 2 个异常，但 trace 归属为对手侧，因此我方
`errors=0`。

| 指标 | iter-05 candidate | iter-06 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 106 / 62 / 2 | 104 / 64 / 2 | 胜 -2 |
| 胜率 | 62.4% | 61.2% | -1.2pp |
| Meta 加权胜率 | 63.7% | 62.5% | -1.2pp |
| 第二回合 Powerful Hand | 24.1% | 25.3% | +1.2pp |
| post-KO 无 ready attacker | 64.4% | 61.2% | -3.2pp |
| 打手断档对局率 | 32.9% | 33.5% | +0.6pp |
| Bench insurance miss | 0 个 fail | 0 个 fail | 保持解决 |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

## Case 验收与决定

- 单元测试覆盖 Rock Fighting Energy 和 Mist Energy：两种能量都使策略优先
  Enhanced Hammer，而不是把 `Powerful Hand` 当作有效 KO。
- 30 局 focused trace 重新分析后，之前的 `bench_insurance_missed` 误报降为 0；
  这证明 analyzer 不再把“已有直接附能/进化选项”的状态错误记为保险遗漏。
- 完整批次中记录到 252 个 Bench insurance 触发点，全部为 `pass`，没有 `fail`；
  其余 post-KO 断档多发生在击倒前没有可见的 ready attacker，不能在 KO 后临时
  补救，应作为下一轮的上游铺场 case。

状态：**observe**。本轮 post-KO 无 ready attacker 和第二回合指标方向改善，且没有
我方合法性错误；但胜率、Meta 加权胜率和打手断档对局率没有同时改善，独立随机批次
也不足以证明因果，因此暂不把 iter-06 晋级为稳定 control。

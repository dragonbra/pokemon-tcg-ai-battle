# Alakazam AutoIter iter-33 分析摘要

## 本轮改动

固定 `deck.csv`，只修正 Dudunsparce 的 Run Away Draw 牌库消耗估算：能力会把被选中的
Dudunsparce、其进化前的 Dunsparce、以及附着卡一起洗回牌库。原实现只计算 Dudunsparce
和能量，遗漏 `preEvolution`，可能把“抽 3 张、放回 3 张”错误估算成牌库净减少 1 张。

本轮修复同时保留了根据实际 option 定位被选中的 Dudunsparce；没有修改卡组、攻击目标、
Bench gate 或其他策略优先级。

## 评测结果

两批都是 17 个对手 × 10 局的独立随机 full-trace，不能进行逐局 A/B 因果比较。

| 指标 | iter-32 参考 | iter-33 首批 | iter-33 repeat |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 118 / 51 / 1 | 107 / 62 / 1 | 104 / 65 / 1 |
| 胜率 | 69.4% | 62.9% | 61.2% |
| Meta 加权胜率 | 70.1% | 65.0% | 61.3% |
| 第二回合 Powerful Hand | 18.2% | 28.2% | 25.9% |
| post-KO 无 ready attacker | 66.5% | 68.0% | 68.9% |
| 对局级打手断档 | 33.5% | 32.9% | 34.1% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

二回合指标在两批都高于 iter-32，但胜率、Meta 和 post-KO 接力事件率没有同步改善。
这说明本轮不能被晋升为新的稳定 best；同时它是对游戏语义的正确性修复，不能因为独立
样本的胜率回退就撤销。

## Case 解释

最新 repeat 的 analyzer 记录了 122 个 post-KO 无 ready attacker 事件和 126 个二回合
未使用 Powerful Hand 的诊断项。它们包含“当时没有合法动作路线”的诊断样本，不能直接
等同于策略错误。`bench_insurance_missed` 中也同时存在 pass 与 fail，后续要从真实 trace
中区分：

1. 是否在击倒发生前已经有可执行的 Poffin、Telepath、Basic 放置或 recovery 路线；
2. 是否只是手里有孤立 Stage 1/Stage 2，却不能在当前回合合法建立接班人；
3. 是否存在可以通过下一轮最小 gate 修复的实际动作，而不是把“希望有资源”硬编码成路线。

## 决策

状态为 **observe，不晋升**。保留 Dudunsparce `preEvolution` 计数修复和对应 fixture，
但 `BEST_STRATEGY.json` 仍指向当前稳定 best `iter-15-patch-priority`。下一轮应继续从
具体 post-KO 断档中寻找最小可验证的 Bench 接力改动，并同时守住胜率和二回合节奏。

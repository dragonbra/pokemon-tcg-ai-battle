# AutoIter V37 决策：诊断口径收窄，策略保持

## 本轮改动

固定 `deck.csv`，只修改 `scripts/alakazam_auto_iter.py`：

- 第二回合没有合法 `Powerful Hand` option 时记录为 `unavailable`，不再误报为策略漏攻；
- Wondrous Patch 只有在能验证 Kadabra/Alakazam 路线时才算 Bench anchor；
- 没有修改 `submission/alakazam_v7_auto_iter/main.py` 的策略行为。

## 最新评测

最新完整样本为 17 个对手 × 10 局，共 170 局：

| 指标 | iter-46 |
|---|---:|
| 胜 / 负 / 平 | 109 / 59 / 2 |
| 胜率 | 64.1% |
| Meta 加权胜率 | 64.9% |
| 第二回合 Powerful Hand | 24.7%（42/170） |
| post-KO 无 ready attacker | 58.8%（110/187） |
| 对局级接力断档 | 37.6%（64/170） |
| 空 Bench Run Away Draw | 0 |
| 我方 action error | 0 |

诊断结果：257 个 Bench insurance case 全部为合法进展，128 个第二回合 case 全部
`unavailable`；当前没有确认的新策略 action miss。

## 决策

**observe，不晋升。** 这轮是分析器修复，不是策略收益，不更新 `BEST_STRATEGY.json`，
也不把 64.1% 与其它独立批次直接作因果比较。下一轮只接受有 raw trace、合法 option
和后续动作链支持的最小策略改动。

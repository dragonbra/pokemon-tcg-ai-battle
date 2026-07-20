# AutoIter 29：二回合已有攻击线时保留 Powerful Hand

## 1. 本轮改动

- 在 `_bench_insurance_due()` 中增加二回合窄例外：如果 Bench 已存在 Abra/Kadabra/Alakazam，
  且没有本回合可完成的直接接力资源，则不因 Poffin 等 speculative insurance 延迟二回合
  Powerful Hand。
- 空 Bench、只有 Dunsparce、直接 Psychic/Telepath/Patch/Lana's Aid 路线和终局判断保持
  iter-28 规则。
- `deck.csv` 未修改。

## 2. 评测前后

candidate 与 control 是独立随机 full-trace 批次；iter-28 作为相邻候选参考。

| 指标 | iter-15 control | iter-28 candidate | iter-29 candidate |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 108 / 61 / 1 | 109 / 61 / 0 |
| 胜率 | 68.2% | 63.5% | 64.1% |
| Meta 加权胜率 | 70.4% | 63.6% | 63.2% |
| 第二回合 Powerful Hand | 27.1% | 22.9% | 27.1% |
| post-KO 无 ready attacker | 68.5% | 65.3% | 69.8% |
| 对局级打手断档 | 36.5% | 35.3% | 32.9% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

## 3. 决策

**observe，不晋升。** 二回合指标恢复、对局级断档下降，但 post-KO event rate、胜率和
Meta 仍不足以超过当前 best。`BEST_STRATEGY.json` 继续保持 `iter-15-patch-priority`，
08:05 继续提交 immutable iter-15。下一轮应优先分析 iter-29 中仍然发生的 post-KO case，
不要再扩大二回合例外的范围。

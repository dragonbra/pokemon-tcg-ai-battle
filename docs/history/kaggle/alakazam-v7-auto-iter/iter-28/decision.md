# AutoIter 28：未充能 Stage 1 不再被误判为 ready 接班人

## 1. 本轮改动

- 移除 `_bench_insurance_due()` 中“只要 Bench 有 Kadabra/Alakazam 就跳过 Bench insurance”的
  例外。
- 现在只有已经附 Psychic Energy 的 Abra 路线，或本回合观察到的实际 Poffin、Telepath、
  Wondrous Patch、Lana's Aid、直接附能路线，才会被视为接力已保障。
- 若没有可见接力动作，仍保持原攻击选择；终局 KO、Item Lock、牌库保护和 `deck.csv` 不变。

## 2. 验收结果

- 新增两个 fixture：
  - 未充能 Bench Kadabra + Poffin + 非终局 KO → 先用 Poffin；
  - 未充能 Bench Kadabra、没有 anchor → 仍攻击。
- V7 策略测试：45/45 通过。
- 完整评测：17 个对手 × 10 局，共 170 局；我方 action error 为 0。

## 3. 评测前后

本轮 candidate 与 control 是独立随机 full-trace 批次，不能视为逐局 A/B；iter-27 仅作
相邻候选参考。

| 指标 | iter-15 control | iter-27 candidate | iter-28 candidate |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 109 / 59 / 2 | 108 / 61 / 1 |
| 胜率 | 68.2% | 64.1% | 63.5% |
| Meta 加权胜率 | 70.4% | 65.0% | 63.6% |
| 第二回合 Powerful Hand | 27.1% | 24.7% | 22.9% |
| post-KO 无 ready attacker | 68.5% | 71.5% | 65.3% |
| 对局级打手断档 | 36.5% | 41.2% | 35.3% |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

## 4. 决策

**observe，不晋升。** 本轮明确改善了接力连续性，目标 case 的方向正确，但胜率、Meta
和第二回合攻击率均低于当前 best，不能让局部指标的改善覆盖主要结果 guardrail。
`BEST_STRATEGY.json` 继续指向 `iter-15-patch-priority`；08:05 定时任务继续提交该
immutable archive。下一轮需要在保住这次接力修复的前提下，单独分析二回合节奏的损失来源。

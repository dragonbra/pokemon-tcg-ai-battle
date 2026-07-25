# AutoIter iter-38 决策：broad Poffin gate 不晋升

## 1. 本轮改动

- Control：iter-37 `Telepath→Abra` 收窄后的策略。
- Candidate：扩大 Bench insurance 的 Poffin 优先级，只要当前攻击前的 Bench 接力不够完整，就倾向先用 Poffin 建立更多 Basic。
- 固定：`deck.csv`、Abra 不攻击、Dunsparce 不使用 Trading Places、攻击立即结束回合、Supporter/手动附能次数限制和保护能量规则。

## 2. 同口径结果

| 指标 | iter-37 control | iter-38 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 117 / 50 / 3 | 101 / 66 / 3 | -16 胜 |
| 胜率 | 68.8% | 59.4% | -9.4pp |
| Meta 加权胜率 | 69.5% | 61.2% | -8.3pp |
| 第二回合 Powerful Hand | 27.6% | 22.9% | -4.7pp |
| post-KO 无 ready attacker | 59.0% | 60.1% | +1.1pp |
| 对局级接力断档 | 30.0% | 38.2% | +8.2pp |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

两批均为 17 个对手 × 10 局的独立 full-trace，不是逐局 A/B；数据来自同一 analyzer，实际读取 170 个 `game_*.json`。分析器第一次生成时曾传入缺少日期后缀的 role label，得到的 0% 第二回合数据已废弃，以上结果使用 trace 中的完整 label `iter-38-second-turn-bench-insurance-20260720` 重跑。

## 3. 决策

**observe，不晋升。** broad gate 虽然减少了部分 Bench insurance case，但同时损害当前回合的攻击节奏、第二回合 Powerful Hand 和整体胜率。`BEST_STRATEGY.json` 不变；iter-38 不作为后续 control。下一轮只保留“全部现有 Abra-line 都是本回合新进场且未附 Psychic”这一窄条件进行验证。

完整逐局 trace 仍位于隔壁评测仓库；当前 repo 只保留本报告和精炼 case。

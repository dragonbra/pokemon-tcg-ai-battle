# Alakazam AutoIter iter-32 分析摘要

本轮固定 `deck.csv`，只修复 `_option_card_id()` 对弃牌区 option 的索引解析：当
`area=3` 时优先使用 `indexInArea`，缺失时回退到 `index`。

最新完整矩阵为 17 个对手 × 10 局、共 170 局；使用 evaluator 默认独立随机性，
不是与 iter-27 的逐局 A/B。完整 trace 仅保留在隔壁评测仓库的 iter-32 目录。

| 指标 | iter-27 control | iter-32 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 109 / 59 / 2 | 118 / 51 / 1 | 胜 +9 |
| 胜率 | 64.1% | 69.4% | +5.3pp |
| Meta 加权胜率 | 65.0% | 70.1% | +5.1pp |
| 第二回合 Powerful Hand | 24.7% | 18.2% | -6.5pp |
| post-KO 无 ready attacker | 71.5% | 66.5% | -5.0pp |
| 对局级打手断档 | 41.2% | 33.5% | -7.7pp |
| 空 Bench Run Away Draw | 0 | 0 | 0 |
| 我方 action error | 0 | 0 | 0 |

`bench_insurance_missed` 的 analyzer fail 数量从 142 降到 65，但该分类仍包含大量
“已有接力、终局或没有明确可执行路线”的诊断项，不能直接当作真实错误数。

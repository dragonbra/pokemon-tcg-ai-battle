# Alakazam AutoIter iter-34 分析摘要

## 本轮改动

V7 之前把“带 Psychic 的 Abra 数量”作为 Poffin 是否可以转向 Dunsparce 的重要信号。
这会把“场上看起来有能量”误当成“下一回合已有可执行的进化接力”。本轮只在 Poffin
effect 选择处补上 `_has_visible_bench_handoff_route()` gate，保留显式
`Dunsparce + Enriching Energy` 的例外；没有修改生产层全局 ready 定义。

## 真实 focused 评测

评测范围为 `kiyotah_dragapult`、`yanxiaohan`、`zoli_dragapult`、`romanrozen_v9`、
`sue_alakazam` 各 10 局，共 50 局，默认独立随机、swap 开启、保存完整 trace。原始
trace 位于评测仓库/临时目录，不纳入主 repo。

| 指标 | 结果 |
|---|---:|
| 胜 / 负 / 平 | 31 / 19 / 0 |
| 胜率 | 62.0% |
| Meta 加权胜率 | 66.3% |
| 第二回合 Powerful Hand | 7/50 = 14.0% |
| 被击倒事件 | 81 |
| 击倒后无 ready attacker | 61/81 = 75.3% |
| 出现过接力断档的对局 | 30/50 = 60.0% |
| 我方 action error | 0 |
| 空 Bench Run Away Draw | 0 |

分析器记录了 87 条 `bench_insurance_missed`、61 条
`post_ko_no_ready_attacker` 和 43 条 `second_turn_powerful_hand_missing`。其中
`post_ko_no_ready_attacker` 主要是击倒后的状态诊断，不能直接推断击倒前有免费可执行
动作；`bench_insurance_missed` 也包含 pass/fail 两类，不能用总数当作策略错误数。

## Poffin 分支是否实际触发

从 50 个 full-trace 中按 effect id `1086` 解析到 67 次我方 Poffin 选择。选择结果中：

- 43 次包含 Dunsparce（其中部分同时选了 Abra）；
- 24 次包含 Abra（其中部分同时选了 Dunsparce）。

因此新代码确实进入了真实 Poffin option 路径；但 focused 批次没有固定随机种子的旧
版本对照，不能声称这 67 次相对于 V6/V7 baseline 的分布变化就是本轮改动带来的。

## 结论

本轮修复了一个语义明确、fixture 可复现的策略错误，并保持 0 action error、0 空 Bench
Run Away Draw；但 focused 样本的第二回合指标和接力指标低于 iter-33 repeat，且样本范围
不同，不能晋升或回退该修复。

下一轮的主问题应从真实 trace 中选取“合法 Bench Kadabra/Alakazam Psychic 或 Telepath
目标存在，但策略把能量贴给 Active”的单变量 case。仍需保留 Active 攻击优先的反例，
避免把资源保护 gate 重新放宽成无条件 Bench 优先。

# Alakazam V7 AutoIter iter-43 分析

## 样本与结果

- 评测：17 个对手 × 10 局，共 170 局；完整 trace 覆盖 170/170。
- 胜 / 负 / 平：106 / 62 / 2；胜率 62.35%。
- Meta 加权胜率：61.29%。
- 第二回合实际使用 `Powerful Hand`：43/170 = 25.29%。
- post-KO 无 ready attacker：110/180 = 61.11%。
- 出现过接力断档的对局：58/170 = 34.12%。
- 空 Bench `Run Away Draw`：0；我方 action error：0。

与 iter-42 的同分析器结果相比，本批次为独立随机样本，不能作严格因果 A/B：

| 指标 | iter-42 | iter-43 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 102 / 66 / 2 | 106 / 62 / 2 | +4 胜 |
| 胜率 | 60.00% | 62.35% | +2.35pp |
| Meta 加权胜率 | 60.10% | 61.29% | +1.19pp |
| 第二回合 Powerful Hand | 21.76% | 25.29% | +3.53pp |
| post-KO 无 ready attacker | 62.0% | 61.11% | -0.89pp |
| 对局级接力断档 | 31.76% | 34.12% | +2.35pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |

## 本轮真正的改动

评测现场的 field option 会出现 `inPlayIndex: null`，实际目标位置在
`indexInArea`。`main.py` 现在统一通过 `_field_option_area()` 和
`_field_option_index()` 回退解析，避免把合法的 Active/Bench 目标误判为不可见。
这修复了运行时 option 解析和动作选择的合法性边界，但没有新增卡组策略优先级，故不能
把本轮指标提升归因于某个新策略 gate。

## Case 复核结论

1. `kiyotah_dragapult/game_005` turn 6：击倒前已经完成可见的 Bench Kadabra/Alakazam
   进化和附能尝试；当时没有可用 Basic Psychic、Lana's Aid、Night Stretcher、Poffin
   或有效的 Telepath handoff。post-KO 断档属于资源不可得，不是漏动作。
2. `nursrijan_lucario/game_001` 的第二回合：Active Abra 到 Kadabra 的进化发生在该回合
   后段，同回合不能连续进化到 Alakazam；trace 中没有合法 `Powerful Hand` option。
3. `kacchan_anti_wall/game_003` turn 11：Poffin 已在攻击前执行，之后才提交攻击；分析器的
   `bench_insurance_missed` 标签不能直接作为策略失败。

因此本轮没有足够证据把攻击优先级改成更激进的 Bench handoff。下一轮只应采纳能在 raw
options 中同时证明“当时可执行、非终局、且当前策略选择了其它动作”的 case。

## 结论

本轮 `observe`，不更新 `BEST_STRATEGY.json`。保留最新完整 trace 用于下一轮审计；旧的完整
trace 在关键 case 已落档后清理，repo 内仅保留本轮轻量证据。

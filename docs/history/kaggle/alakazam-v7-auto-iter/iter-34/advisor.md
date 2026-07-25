# AutoIter iter-34 Strategy Advisor

## 复核结论

本轮 advisor 复核了固定卡组的卡牌效果、进化时序、Poffin 的合法范围以及 iter-33
repeat 的击倒前 trace。结论是：`_ready_attack_line_count()` 可以用于描述场上已经
带 Psychic 的资源，但它不能独自证明下一次 Poffin 选择 Dunsparce 是正确的 handoff。
Poffin 的分支应继续使用更严格的“可见进化/接力路线”判定，而不是全局重定义 ready。

Poffin 的两个边界必须保持：

1. Poffin 只能从牌库放置 HP 不超过 70 的 Basic Pokémon，不能直接制造 ready attacker；
2. Dudunsparce 是抽牌引擎，不是 Psychic 打手。只有明确的 Enriching Energy 抽牌路线，
   或 Abra-line 的可见 handoff 路线，才足以让 Dunsparce 暂时占用 Poffin 槽位。

## 对击倒前接力的建议

iter-33 repeat 的 forensic review 还确认了三个真实的“合法 Bench 附能目标未被选中”
case：

- `kacchan_anti_wall/game_005`，turn 5：唯一 Basic Psychic 被贴给 Active Kadabra，
  Bench Kadabra 也有合法附能目标；
- `kiyotah_dragapult/game_003`，turn 7：Telepath Psychic 贴给 Active Kadabra，
  没有贴给 Bench Kadabra；
- `romanrozen_v9/game_010`，turn 4：Telepath Psychic 贴给 Active Abra，Bench
  Kadabra 已经可以在下一回合直接攻击。

这些不是“只要有 Bench 就强制接力”的理由：若唯一能量是当前攻击所必需，必须保留
当前攻击；但当 Bench Kadabra/Alakazam 的 Psychic 或 Telepath 目标明确存在，且 Active
仍有可替代攻击路径时，应在下一轮单独验证“优先选择 Bench 目标”。不能把这项改动与
Poffin gate 或 Lana's Aid 回收策略混在同一轮。

## Advisor 状态

本 advisor 不修改生产代码。iter-34 保留 Poffin 的窄 gate；下一轮候选建议以一个真实
的 Bench Psychic/Telepath 目标选择 fixture 为主变量，并保留唯一能量仍供 Active 攻击的
反例。

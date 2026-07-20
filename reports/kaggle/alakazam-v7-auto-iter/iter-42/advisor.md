# iter-41 独立规则 / 卡牌 Advisor

## 结论

本轮 `handoff_preparation_missed` 不是可以整体修复的真实 failure 集合。逐一用 raw trace 的实际行动索引、攻击伤害和目标 Prize value 复核后，这 12 个候选都属于终局攻击，或在攻击前已有更高价值的立即 KO 路径；没有留下可确认的非终局 handoff 策略错误。建议 `observe`，不修改终局规则，也不为了通过误设的测试强行提高 handoff 优先级。

## 复核中排除的候选：`maktha_1084-game-9-turn-13`

- `case_id`: `maktha_1084-game-9-turn-13-handoff_preparation_missed`
- raw trace：`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-41-final-prize-energy-gate-20260720/maktha_1084/game_009.json`
- `trace[139]` 的确是行动 `[13]` 的 Dudunsparce Run Away Draw，但它不能单独作为 handoff 失败：手牌从 14 张增加 3 张后，Powerful Hand 从 280 提高到 340，正好可以击倒 340 HP 的 Active `678`。
- 该 draw 是可继续进行的主行动，不是攻击后的动作；随后发动攻击仍符合回合规则。因此这里应把“抽牌后获得 KO”视为高价值的直接闭环，而不是把 handoff 固定压在它前面。
- analyzer 标记的 case 状态实际对应 `trace[147]`：手牌 20 张、目标 `678`、剩余 Prize 2。此时 Powerful Hand 造成 400，`678` 属于 3 Prize Pokémon，击倒后剩余 Prize 不超过目标 Prize value，攻击完成最后奖赏，直接攻击正确。
- 结论：原 advisor 报告把 trace 索引和行动状态混淆了；这是 analyzer/复盘口径问题，不是需要修改 `main.py` 的策略 case。

## Analyzer 误报 / 正确的终局攻击

- `kiyotah_lucario-game-7-turn-13`：`trace[152]` Active Alakazam 手牌 21 张，对手 `674` 仅 80 HP，Powerful Hand 造成 420，Prize=1；攻击拿完最后奖赏，直接攻击正确。
- `kokinn_search-game-8-turn-12`：`trace[141]` 手牌 25 张，攻击造成 500，对手 `678` 为 340 HP，Prize=2；`678` 属 3 Prize 目标，攻击闭环成立。Bench Kadabra 附能不应阻止终局攻击。
- 同类误报包括 `maktha_1084-game-6-turn-8`、`maktha_1084-game-7-turn-11`、`maktha_1084-game-10-turn-8`、`nursrijan_lucario-game-2/game-8`、`romanrozen_v9-game-10`、`yakitori_raging_bolt-game-2`、`zoli_dragapult-game-9`：逐一按手牌伤害、目标 HP、Prize 数和目标 Prize value 检查后，攻击均可完成终局，不能作为 handoff failure。
- 因此 `post_ko_no_ready_attacker` 以及这些 attack 型 `handoff_preparation_missed` 不能直接推出需要提前附能；对局在最后奖赏结算后已经结束，Bench 打手没有后续价值。

## 规则与实现复核

- 官方规则要求先抽牌，再按任意顺序完成主行动；攻击是终止提交。附能、进化、Ability、Item 和 Supporter 都不是“攻击前必须全部执行”的固定步骤。
- `main.py:391–470` 的 `_v7_terminal_prize_closure` 基于实际 Active、攻击伤害、目标 HP 和 Prize value 判断终局；`main.py:2889–2896` 在 closure 为真时降低 Energy 附加优先级，规则语义正确。
- `main.py:1555–1680` 的 `_bench_handoff_preparation_due` 已有直接 Basic/Telepath 附能、Lana 和 Wondrous Patch 路径，但终局 closure 会先返回 False；这对 Case 1 不应触发，因为攻击不能 KO。
- 官方卡面：Buddy-Buddy Poffin（`1086`）找最多 2 张 70 HP 以下 Basic；Wondrous Patch（`1146`）把弃牌 Basic Psychic 直接附到 Bench Psychic；Lana’s Aid（`1184`）从弃牌最多拿 3 张无 Rule Box Pokémon/Basic Energy。当前 Case 1 实际可见的是 Telepath Energy，不应把它误归为不可得资源。

## 迭代结论

1. 不修改终局攻击优先规则；终局 attack cases 应保留为反例。
2. 保留 `maktha_1084/game_009 trace[139]` 作为“抽牌后形成 KO，不能被 handoff 抢先”的回归 fixture。
3. 保留 `maktha_1084/game_009 trace[147]`、`kokinn_search/game_8 trace[141]` 与 `kiyotah_lucario/game_7 trace[152]` 作为“3 Prize 目标完成最后奖赏，终局攻击必须优先”的反例。

**总体 confidence：高。** 本轮没有足够证据支持生产策略改动；下一轮应转向真正的 post-KO 无 ready attacker 或空 Bench 连续性 case。

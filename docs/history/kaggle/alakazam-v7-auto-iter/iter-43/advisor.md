# V7 AutoIter iter-43 独立规则 / 卡牌 Advisor

## 结论

**observe：no actionable case。** 在限定审计范围内没有确认真实的策略错误，因此不建议修改 `main.py` 或 `deck.csv`。

本轮只核查了优先对手中的三个代表性 full-trace case；没有继续遍历全部 trace。分析器的 failure label 不能单独证明“应当先做 handoff”：必须同时存在可执行的 Poffin、Telepath、Lana's Aid、Night Stretcher、Psychic 附能或合法进化路径，并且当前攻击不属于终局 closure。

## 代表性复核

### 1. `post_ko_no_ready_attacker`：资源不可得，不是策略错误

- case：`kiyotah_dragapult-game-5-turn-6-post_ko_no_ready_attacker`
- raw：`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-42-control-20260720/kiyotah_dragapult/game_005.json`
- 击倒前我方回合：turn 5，`trace[53]` 到 `trace[62]`
- 实际动作链：`trace[53]` 选择 option type `9`，把手牌 `Alakazam (743)` 合法进化到 Bench Kadabra；随后完成 Poffin 链；`trace[61]` 选择 option type `8`，把 `Telepath Energy (19)` 附到 Active Abra；`trace[62]` 的实际 action `[3]` 是 option type `14`（结束回合）。
- 关键状态：此时两只 Bench Alakazam 都是 `appearThisTurn=true` 且没有 Energy。手牌没有 Basic Psychic；Basic Psychic 与 Telepath Energy 已在 discard。Lana's Aid、Night Stretcher、Wondrous Patch 也没有可执行 option。Telepath Energy 不能附到已经进化的 Alakazam，因此不能把它当作 Bench handoff。
- 结论：没有可执行的 Poffin/Telepath/Lana/Night Stretcher/Psychic attachment/evolution 路径；结束回合不是可确认的策略错误。对手随后击倒 Active 属资源断档，不是漏掉了可见 handoff。

### 2. `second_turn_powerful_hand_missing`：进化时机和资源都不满足

- case：`nursrijan_lucario-game-1-turn-3-second_turn_powerful_hand_missing`
- raw：`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-42-control-20260720/nursrijan_lucario/game_001.json`
- trace：第二回合主行动链为 `trace[21]` 到 `trace[38]`；最终 `trace[38]` 的实际 action `[0]` 选择 option type `13`、`attackId=1071`，而不是 `attackId=1072`。
- 关键状态：`trace[36]` 才把 Active Abra 进化为 Kadabra，`trace[37]` 后 Active Kadabra 已在本回合进化；`trace[38]` 不能再连续进化为 Alakazam。虽然手牌有 Alakazam `743`，但没有 Basic/Telepath Psychic 可用于当前攻击路线，合法选项也没有 `1072`。
- 卡牌/规则依据：同一只 Pokémon 同回合不能连续进化；Powerful Hand 需要已经合法存在并具备 Psychic Energy 的 Alakazam。这里是进化时机与资源不可得，不是应把攻击改成 Powerful Hand。

### 3. `bench_insurance_missed`：实际已经先做 Poffin

- case：`kacchan_anti_wall-game-3-turn-11-bench_insurance_missed`
- raw：`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-42-control-20260720/kacchan_anti_wall/game_003.json`
- trace：`trace[83]`、`trace[84]`、`trace[85]`、`trace[87]`
- 实际动作：`trace[83]` 的 action `[2]` 对应 option type `7`、手牌 `Poffin (1086)`；`trace[84]` 是 Poffin 的 deck selection；之后才在 `trace[87]` 选择 option type `13`、`attackId=1071`。
- 结论：该候选不是“攻击覆盖了可执行 Poffin”；Poffin 已在攻击前执行。当前手牌没有 Psychic attachment，Lana's Aid 虽在 discard 但不在当回合合法主选项中，不能倒推为漏用。后续接力不足不构成已确认 action error。

## 规则边界

- 只要没有可执行的 handoff 路径，或者当前攻击能形成最后 Prize closure，就保留攻击/结束回合，不为通过 analyzer label 强行提高准备优先级。
- `Poffin` 只能检索最多两张符合条件的 Basic；`Telepath Energy` 只能按卡面允许的 Basic Psychic 路径使用，不能直接给 Alakazam 建立接力。
- 进化受“本回合开始时已在场”和“同回合不能连续进化”约束；攻击一旦提交即结束本回合。

## 建议

本轮保持 **observe**。上述 case 可作为后续 analyzer 的负例：只有在击倒前回合能从 raw option 证明存在具体的 Poffin/Telepath/Lana/Night Stretcher/Psychic attachment/evolution 路径，且不是终局或对手先手造成的不可控结果时，才升级为 actionable strategy case。

# AutoIter iter-35 Strategy Advisor

## 复核结论

本轮 advisor 复核了 iter-33 repeat 的三个击倒前真实 case：

- `kacchan_anti_wall/game_005` turn 5：Active Kadabra 与 Bench Kadabra 都可接收唯一
  Basic Psychic；
- `kiyotah_dragapult/game_003` turn 7：Telepath Psychic 同时暴露 Active/Bench Kadabra
  目标；
- `romanrozen_v9/game_010` turn 4：Active Abra 与已进化 Bench Kadabra 同时暴露 Telepath
  目标。

三者的共同点是：目标是实际 legal option，Bench Kadabra/Alakazam 下一回合不需要再
进化；因此这不是“手里可能有资源”的推测。反过来，只有孤立 Bench Abra、没有 Kadabra
或 Rare Candy 路线时，仍不能把 Basic Psychic 的附能当成确定 handoff。

卡牌/规则边界：

- 手动 Energy 每回合一次，所以选择 Active 还是 Bench 是同一回合的互斥资源决策；
- Telepath Energy 的附能目标必须是 Psychic Pokémon，同时它能搜索 Basic Psychic，
  因而贴给已有 Bench Kadabra 是一个可验证的接力路线；
- Kadabra 已经在场时，`appearThisTurn` 只限制再次进化，不限制接收 Energy 或下一回合攻击；
- 没有真实 Bench successor 时，必须保留 Active Energy 路线，不能为了抽象的“Bench 保险”
  放弃当前攻击准备。

## 额外观察

另一份 advisor review 发现 `kiyotah_iono/game_002` 中 Fezandipiti ex 曾先于低奖赏
Dunsparce/Abra insurance 被使用，但该局最终获胜，不能作为本轮直接修复目标。它将作为
后续候选单独验证，不能与 Energy target 优先级混合。

本 advisor 不修改生产代码；本轮只验证 Bench Energy target 的窄 gate。

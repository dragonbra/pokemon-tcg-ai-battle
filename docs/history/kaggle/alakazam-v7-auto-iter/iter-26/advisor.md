# AutoIter 26 Strategy Advisor

## 复盘结论

本轮重点复核的是“把 Psychic Energy 贴错目标”的接力问题。此前 trace 中有三个明确
case：Active Alakazam/Kadabra 已有 Psychic、Bench Abra 已经在场、手牌有 Telepath
Psychic Energy，但策略仍把能量贴回 Active。Active 之后被击倒时，Bench 没有足够快的
接班线。

- `kiyotah_lucario/game_010.json`，turn 4 / trace 36：原动作把 Telepath 贴给已充能
  Active Alakazam；当前策略应选两个 Bench Abra 选项之一。
- `pilkwang_v2/game_004.json`，turn 14 / trace 129：Active Alakazam 只剩 10 HP，
  原动作仍贴 Active；当前策略应贴 Bench Abra。
- `kiyotah_abomasnow/game_001.json`，turn 9 / trace 81：Active Kadabra 已有 Psychic，
  原动作把 Telepath 贴回 Active；当前策略应为 Bench Abra 准备能量。

这些 case 已由工作区现有的 Bench handoff gate 覆盖。本轮新增的边界是：Basic Psychic
贴给 Bench Abra 只有在当前手牌已经看见 Kadabra，或看见 Alakazam + Rare Candy 且没有
Item Lock 时，才算“可验证接力”。未来抽牌不是当前状态的事实。

Telepath Energy 保持例外：它除了提供 Psychic，还会检索最多两只 Basic Psychic
Pokémon，因此即使下一阶段尚未在手牌中出现，也能建立真实的 Bench anchor。Lana's Aid
也保持现有恢复路径：它可一次取回 Abra 与 Basic Psychic，不应被本轮的 Basic 手动
贴能量 gate 误伤。

## 规则边界

- 新放下的 Abra 本回合不能进化，但可以合法接收 Psychic Energy。
- 直接 Basic Psychic attachment 不得预设下一回合会抽到 Kadabra。
- Telepath 的检索效果是本回合可观察、可执行的资源，因此仍可作为 Bench anchor。
- 最后奖赏闭环、Budew `Itchy Pollen` Item Lock、每回合一次手填 Energy 不变。

本轮 advisor 不建议再扩大第二回合 Powerful Hand 的优先级；下一轮可独立验证它是否
会越过确定的 Boss 高奖赏路线，但不与本轮混合。

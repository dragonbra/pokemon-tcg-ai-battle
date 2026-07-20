# AutoIter iter-36 Strategy Advisor

## 规则复核

- 手动附能每回合只能一次，因此 Active 与 Bench 的 Psychic 目标是互斥决策。
- Alakazam 在本回合刚由 Kadabra 进化后，仍可在同一回合接受 Energy 并攻击；
  `appearThisTurn` 禁止的是再次进化，不是附能或攻击。
- Bench handoff 只有在 Active 当前没有可完成的攻击路线时才应优先。
- Kadabra/Alakazam 是具体的接班目标；孤立 Abra、Dunsparce、Dudunsparce 和 Fezandipiti
  不能自动视为 Psychic 打手。

## 本轮结论

本轮修复的 gate 边界是合理的：Active 已是 Alakazam 且只差 Psychic 时，当前回合的
确定攻击优先于下一回合的 Bench 保险。回归 fixture 与真实历史状态一致。

## 未采纳的候选

Telepath 会附能并搜索 Basic Psychic Pokémon，但它不能直接搜索 Kadabra/Alakazam。
因此 Telepath → Bench Abra 只有在手牌可见 Kadabra，或可用 Alakazam + Rare Candy 且
没有 Itchy Pollen Item Lock 时，才是明确 handoff；否则只是 Bench insurance。该候选
留到下一轮，避免与本轮即时攻击 guard 混合。

本 advisor 不修改生产代码。

# iter-38 Strategy Advisor

## 卡牌与规则核验

- Buddy-Buddy Poffin 只能从牌库放置符合条件的 Basic Pokémon 到 Bench；它不会附 Psychic，也不会把新放下的 Abra 直接变成下一回合可攻击的打手。
- Enriching Energy 给 Dudunsparce 的抽牌路线很有价值，但它不能替代当前回合的攻击或已经存在的 Abra-line 接力。只有牌面明确显示这条抽牌路线时，才应把 Dunsparce 纳入 Poffin 的特殊优先级。
- Active Alakazam 已经带 Psychic 且 Powerful Hand 合法时，宣告攻击会立即结束回合；若只是为了“再多铺一个 Basic”而广泛使用 Poffin，会直接牺牲本回合攻击。
- Bench Kadabra/Alakazam 的 Psychic handoff 与“刚放下、未附能量的 Bench Abra”不是同一类状态。后者需要额外的进化来源，不能泛化成 ready attacker。

## 本轮结论

iter-38 的 broad Poffin gate 过宽：它把“Bench 尚未完美”当成了足以压过攻击的理由。建议保留 Bench 连续性的硬约束，但把先 Poffin 的条件收窄到可证明的极窄状态；下一轮应检查实际手牌中是否存在 Poffin、以及 Bench 线路是否全部为本回合新进场且未附 Psychic。

本 advisor 不建议修改卡组，也不建议把 Abra、Dunsparce 或 Fezandipiti ex 重新定义为 ready attacker。

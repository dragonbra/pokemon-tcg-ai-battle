# iter-39 Strategy Advisor

## 卡牌与规则核验

- Poffin 适合在 Active 攻击之后的下一回合资源连续性不足时补 Basic，但不能无条件越过本回合已经可完成的 Powerful Hand。
- 如果 Bench 上已有带 Psychic 的 Kadabra/Alakazam，或存在能在下一回合自然进化并接力的路线，当前回合不应为了“再多放一只”而延迟攻击。
- 如果 Bench 上的 Abra-line 全部都是本回合刚放下且都没有 Psychic，且手牌有 Poffin，那么建立更多 Basic 是有明确防止单点 KO 失败的价值；这正是 iter-39 相比 iter-38 收窄后的可解释边界。
- Dudunsparce + Enriching Energy 是抽牌路线，不是 Abra-line ready attacker。只有在已有完整 Abra-line、Active 有明确 Alakazam 攻击路线且不会牺牲攻击节奏时，才可考虑给 Dudunsparce 资源。
- Telepath Energy 只能检索 Basic Psychic；Telepath→孤立 Bench Abra 不能被当作确定的下一回合打手，除非手牌同时能提供 Kadabra，或能合法完成 Alakazam + Rare Candy 路线且没有 Budew 的 Itchy Pollen 锁。

## 主要风险

iter-39 将第二回合 Powerful Hand 提升到 30.0%，但 post-KO 无 ready attacker 事件率升到 64.6%。这说明“新鲜 Bench 全部未充能时先 Poffin”的 gate 仍可能只解决了二回合观测量，却没有保证后续击倒后的实际接班。下一轮应优先寻找击倒前已经存在合法 Poffin/Basic/进化路线、但最终仍没有 ready attacker 的具体 trace，再做更小的 handoff 修正。

本轮没有发现可以合理放宽 Abra 攻击或 Trading Places 的证据；这两条用户确认的硬边界保持不变。

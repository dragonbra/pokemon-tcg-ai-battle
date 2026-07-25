# AutoIter iter-37 Strategy Advisor

## 卡牌与规则核验

- Telepath Energy 的附能效果只能寻找 Basic Psychic Pokémon，因此它可以找 Abra，不能
  直接找 Kadabra 或 Alakazam。
- 手动附能每回合只有一次；Telepath→Active 与 Telepath→Bench 是互斥的资源分配。
- Bench Kadabra/Alakazam 在附上 Psychic 后可以作为明确接班者；Bench Abra 还必须有可见
  的下一步进化来源。
- `appearThisTurn` 只限制本回合再次进化，不限制这回合附能，也不妨碍下一回合攻击。
- Budew 的 `Itchy Pollen` 会锁住 Rare Candy 路线；因此 `Alakazam + Rare Candy` 只有在
  Item 未被锁定时才是有效的 Abra 接力路线。

## 建议与边界

1. `Telepath → Bench Abra` 仅在手牌有 Kadabra，或有 Alakazam + Rare Candy 且没有
   Itchy Pollen lock 时提升为 concrete handoff。
2. `Telepath → Bench Kadabra/Alakazam` 保持 concrete handoff。
3. `Telepath → Active` 和“没有 concrete handoff 时建立 Bench insurance”的路径不要被
   本轮 gate 改写。
4. 不要修改全局 ready attacker 定义，也不要把 Dunsparce、Dudunsparce 或 Fezandipiti ex
   当成 Abra 线接班者。

本轮 advisor 只提供卡牌/规则核验，不直接修改生产策略。

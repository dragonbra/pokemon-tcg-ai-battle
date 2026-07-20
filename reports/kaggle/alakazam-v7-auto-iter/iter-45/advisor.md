# Iter-45 Advisor

## 本轮问题

本轮尝试修复 `zoli_dragapult/game_004 raw[75]`：Active Alakazam 已有 Psychic，
Bench Abra 可以通过 Telepath Energy 接到下一阶段 Kadabra；手牌没有直接可见 Kadabra，
但 Poké Pad 的牌库资源账本允许把这条路线视为可验证来源。

## 卡牌与规则核验

- Telepath Energy 附到 Psychic Pokémon 后可以检索 Basic Psychic Pokémon，因而同时承担
  附能和建立 Bench 锚点的作用。
- Abra 本回合不能进化；它只能作为下一回合 Kadabra/Alakazam 路线的基础，不能被当成
  当前回合的稳定攻击者。
- 当前实现的 `_pokepad_kadabra_route_available()` 只在资源账本明确显示 Kadabra 来源时
  放宽该路线，不会因为未知牌库资源而无条件假设成功。

## 结论

该 case 已有策略回归测试，当前代码会把 Telepath Energy 给 Bench Abra；不再增加第二层
“空 Bench / Abra fallback”规则。独立评测中没有足够证据证明本轮应晋升为新的稳定 best。

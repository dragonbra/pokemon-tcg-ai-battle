# Iter-21 Strategy Advisor

## 本轮问题

本轮只验证 `Wondrous Patch` 的窄化 gate：当 Bench Abra 已有 Psychic、另一只
Bench Abra 没有能量，且手牌没有 Kadabra 或 Rare Candy + Alakazam 路线时，不应因为
Patch 合法就把 Psychic 再贴到已经充能的 Abra。

## 卡牌与规则核验

- `Wondrous Patch` 可以把弃牌区的 Basic Psychic Energy 直接贴到 Bench 的 Psychic
  Pokémon；它不消耗本回合的手动填能量次数。
- 但“合法附能”不等于“形成下一只打手”。没有可见的 Kadabra，或没有可用的
  Rare Candy + Alakazam，给无能量 Abra 充能的收益不能在当前已知信息下确认。
- 进化不能在刚下场的同一回合完成；Patch 只应提前准备下一回合路线，不能改变这个
  规则。
- Active Alakazam 的攻击是回合终止动作；终局奖赏闭环仍优先于未来接力。

## 对下一轮的建议

Patch gate 应保持本轮收紧，不再放宽。更值得单独验证的是 Poffin 的目标分配：仅在
已看到至少三只 Abra-line、场上没有 Dunsparce、手中同时有 Enriching Energy 和
明确的 Active Alakazam 路线时，Poffin 才可以优先找 Dunsparce；否则继续优先 Abra
以保证 Bench 连贯性。

这个假设必须用单一 fixture 验收，并观察第二回合 Powerful Hand、post-KO 接力和
胜率 guardrail；不能把 Dunsparce 的过牌价值泛化成“永远比 Abra 更重要”。

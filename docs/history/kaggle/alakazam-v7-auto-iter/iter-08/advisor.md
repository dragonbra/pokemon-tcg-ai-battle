# AutoIter iter-08 Advisor：Active Kadabra 的接力能量

## 真实 case

- `sue_alakazam/game_003.json`，trace 43：Active Kadabra 已有 Psychic；Bench 有已在场
  的 Abra；手牌有 Telepath、Kadabra、Alakazam。原策略把 Telepath 贴回 Active，随后
  Bench Abra 只进化成无能量 Kadabra，Active 被击倒后接力断档。
- `kacchan_anti_wall/game_008.json`，trace 71：Active Kadabra 已有 Psychic；Bench
  Kadabra 无能量；手牌有 Telepath 和 Basic Psychic。原策略仍把 Telepath 贴回 Active，
  没有为 Bench Kadabra 准备能量。

这两个 case 的资源均可见，属于策略漏用，不是资源不可得。

## 本轮假设与实现

把“准备 Bench handoff”的 gate 从仅 Active Alakazam 扩展到“已带 Psychic 的 Active
Kadabra 或 Alakazam”。当 Bench 有合法、无能量的 Abra/Kadabra/Alakazam 且当前选项暴露
Psychic/Telepath 或 Lana's Aid 路线时，先完成接力准备；仍保留自然进化和最后奖赏闭环
优先级。

新增 fixture：Active Kadabra + Psychic、Bench Kadabra、Telepath 同时可贴 Active 或
Bench，断言选择 Bench 目标。

## 边界

Active Kadabra 没有 Psychic 时不触发该 gate；它应先完成自己的自然进化/攻击路线。仅有
Dunsparce 仍不算 Abra-line ready handoff。

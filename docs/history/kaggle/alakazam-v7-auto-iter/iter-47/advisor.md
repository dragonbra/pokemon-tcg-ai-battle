# Iter-47 Advisor

## 本轮 case

`crustle_v1/game_007 raw[9]`：Active Dunsparce 是本回合刚放下的宝可梦，手牌同时有
Dudunsparce 和 Enriching Energy，合法 options 包含把 Enriching Energy 附到 Active
Dunsparce。旧策略因为本回合不能进化而结束回合。

## 规则核验

- 本回合不能把刚放下的 Dunsparce 进化为 Dudunsparce；但附能本身合法。
- 附能不会消耗进化机会，能为下一回合的自然进化和 Dudunsparce 过牌提前准备。
- Enriching Energy 仍不能被当作 Psychic Energy 给 Abra/Kadabra/Alakazam 直接准备攻击。
- 本轮没有放宽 Abra 攻击或孤立 Abra 的 Basic Psychic handoff。

## 评测结论

该 case 的局部动作修复明确，`crustle_v1` focused 10 局为 10W/0L/0D、action error 0。
但两批独立全矩阵分别为 96/71/3 和 104/66/2，胜率 56.5% 和 61.2%，都低于
iter-46 的 64.1%。由于没有保存完整 trace，第二回合和 post-KO 指标在这两批中不可用；
不应把 summary-only 的 0 当成真实指标。

## 建议

保留该规则和 fixture 作为 observe candidate，但不晋升；下一轮需要有 trace 的 focused
评测确认它是否改善第二回合准备而没有牺牲其它接力路径。

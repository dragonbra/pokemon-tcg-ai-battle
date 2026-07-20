# Iter-25 Strategy Advisor

## 规则核验

条件式支持“已有 Bench Kadabra/Alakazam 时，不因泛化 Poffin/Hilda setup 阻塞
Powerful Hand”：

- Active 必须是已附 Psychic 的 Alakazam；
- 当前合法选项必须确实包含 Powerful Hand（1072）；
- Bench 已有 Stage 1/Stage 2 攻击线，因此 Bench presence 已经满足；
- 若当前仍暴露直接 Psychic 附能、Active Kadabra 自然进化、Recovery 或终局闭环，
  这些动作仍按原优先级处理。

不能把这个条件扩大到空 Bench、只有 Dunsparce、未充能 Active Kadabra，或没有真实
Powerful Hand 选项的局面。Dunsparce 仍是抽牌/缓冲位，不替代 Abra-line 接力；Rare
Candy 仍受进化时机和 Budew `Itchy Pollen` Item Lock 约束。

## Trace 建议

本轮原始 trace 中有 38 次“Active Alakazam 已充能 + Bench 有 Stage 1/2 + Powerful
Hand 可选”而实际直接攻击的样例。这个局部行为符合上述条件，但整体胜率尚未改善，
因此只作为候选观察，不应直接晋升。

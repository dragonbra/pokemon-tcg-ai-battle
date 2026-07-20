# 当前候选策略

当前候选优先完成合法的 Alakazam 进化与攻击路线。攻击是回合终止提交，因此在攻击前先处理进化、铺场、过牌、Supporter 和附能。

- 根据 observation 中实际可见的 Active、Bench、Energy、伤害、Prize 和牌库压力决策，不预设对手构筑。
- Active Kadabra 已具备 Psychic Energy 且手中有 Alakazam 时，优先自然进化并攻击，再处理合法的 Bench 进化和过牌。
- 牌库进入低位时保护终局闭环；只有能增加伤害并完成收尾时才继续消耗牌库。
- 所有动作必须来自 simulator 的合法选项，状态判断补充记录进场、进化、Supporter、附能和攻击时点。

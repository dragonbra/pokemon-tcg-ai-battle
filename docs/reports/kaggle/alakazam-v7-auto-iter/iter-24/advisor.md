# Iter-24 Strategy Advisor

## 规则与卡牌核验

本轮只测试 Night Stretcher 的窄接力路径。它可以从弃牌区取回一只宝可梦，但不能像
Lana's Aid 一样同时取回 Basic Energy。因此只有在以下信息都可见时才成立：

- Active Kadabra/Alakazam 已附 Psychic，且本回合可以正常结束攻击；
- Bench 没有 Abra/Kadabra/Alakazam，仍有空位；
- 弃牌区有 Abra，手牌已经有 Basic Psychic 或 Telepath Psychic Energy；
- 当前没有 Poffin、手牌 Abra 或 Lana's Aid 这类更直接/更完整的路径；
- 不是最后奖赏闭环，也没有 Budew 的 `Itchy Pollen` Item Lock。

正确动作链是：本回合 `Night Stretcher -> Abra`，随后把 Abra 放到 Bench 并附上手牌
Psychic；新下的 Abra 本回合不能进化，下一回合才进入 Kadabra/Alakazam 路线。这个
动作不能为了“看起来有资源”而覆盖终局攻击、Lana's Aid 或 Poffin。

## Trace 观察

本批 full-trace 中，窄条件实际触发 23 次，23 次都选择了 Night Stretcher；其中没有
action error。iter-23 的 3 个 `bench_insurance_missed=fail` 经过 advisor 复核均是
analyzer 误报，分析器已增加对 `Lana's Aid -> Abra + Psychic` 恢复链的识别，不把它们
误算成新的策略失败。

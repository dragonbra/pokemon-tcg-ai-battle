# iter-40 Strategy Advisor

## 卡牌与规则核验

- Night Stretcher 只能把弃牌区的一张宝可梦或一张基础能量取回手牌，不能单独同时补齐 Abra 与 Psychic Energy。
- Lana’s Aid 可以把弃牌区的宝可梦和能量同时恢复到手牌；只有这两类资源都在弃牌区、且恢复后能对应到当前可验证的 Bench 接力路线时，才有理由把它视为 Night Stretcher 的完整替代。
- 当弃牌区只有 Abra、手牌已有 Psychic Energy 时，Night Stretcher → Abra → Bench，再由手牌能量完成附能，是比“因为手里有 Lana 就继续等待”更具体的路线。
- 这个判断不能把 Dudunsparce 的 Enriching Energy 抽牌路线误当成 Abra-line ready attacker；Dudunsparce 是资源引擎，不能替代有 Psychic 的 Kadabra/Alakazam 接班。
- Abra 仍不应攻击；Trading Places 仍不应使用。恢复 gate 不能突破这些已确认的规则边界。

## 本轮副作用审查

iter-40 的具体回归 case 方向正确，但 full matrix 给出警示：胜率 67.1% → 60.0%，第二回合 Powerful Hand 30.0% → 23.5%，对局级接力断档 33.5% → 38.2%。post-KO 事件率 64.6% → 62.4% 的改善不能单独证明 gate 有益。

合理的解释假设是：恢复动作的局部修复减少了部分“击倒后完全无 ready attacker”事件，却可能在其它局面把本来可以立即完成的 Supporter、附能或攻击顺序推迟，或者消耗了原本用于当回合进攻的资源。这需要逐局 trace 验证，不能仅凭指标确定。

## 下一轮建议

1. 先锁定 `kiyotah_dragapult`、`kiyotah_iono` 和 `nursrijan_lucario` 中“手里有恢复牌但仍未形成下一只 ready attacker”的真实 action；区分是资源不存在，还是策略错过了 Poffin、直接放下 Abra、附能或自然进化。
2. 将 Night/Lana 优先级限制在“确实能在本回合或下一回合完成 Bench handoff”的状态，不把恢复牌本身当作价值；已有可攻击 Active 且没有明确接力收益时，不能为了恢复而延迟攻击。
3. 任何下一轮候选必须同时观察第二回合 Powerful Hand、post-KO 事件级接力和胜率；单个 case 修复但二回合或胜率持续下降时应回退或继续 observe。

本轮没有证据支持放宽 Abra 攻击、Trading Places，或改变 Mist/Rock Fighting Energy 的保护处理。

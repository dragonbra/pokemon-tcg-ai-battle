# 设计原则

- 先遵守 simulator 提供的合法动作集合，再优化策略指标；攻击是回合终止提交。
- 不预设对手构筑，只依据 observation 中实际可见的 Pokémon、Energy、伤害、手牌、Prize 和牌库压力决策。
- 进化、附能、Supporter、Retreat 和牌库消耗都记录时点与资源区域，不能用当前可见状态替代历史事实。
- 低牌库时优先保护可完成的终局闭环；继续过牌必须有明确的伤害或收尾收益。
- 评测使用统一分母比较 control 和 candidate；本地对局用于合法性与崩溃检查，官方 Kaggle 结果作为最终依据。
- 每轮 Auto Iteration 只验证一个主要假设，先通过 correctness gate，再处理 case resolution、focused evaluation 和 outcome guardrail。

详细规则证据见 [规则报告](../../docs/reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md)。

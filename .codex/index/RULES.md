# 规则索引

这里保存可执行的规则摘要；完整证据见 [规则报告](../../docs/reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md)。

- 攻击是回合终止动作；宣告后不能继续执行主行动。
- 进化、Supporter、手动附能、Retreat 和卡牌效果必须遵守 simulator 提供的时点与合法选项。
- 牌库、Prize、手牌、Active、Bench 和弃牌区是独立资源区域，资源账本必须随区域变化更新。
- 低牌库时优先保护终局闭环，不能把“抽不到足够数量”误判为立即失败。

策略实现仍以 simulator 的 observation 和合法动作集合为最终边界。

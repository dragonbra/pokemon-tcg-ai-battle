# AutoIter iter-06 Strategy Advisor

## 复核结论

本轮用户评测的主要问题不是合法性：我方 action error 为 0，空 Bench 的
`Run Away Draw` 也为 0。更核心的问题是把“Bench 有宝可梦”误当成“KO 后有可用
接力”。用户给出的 170 局数据中，击倒后无 ready attacker 为 116/165（70.3%），
60/170 局出现过接力断档。

本轮严格采用以下定义：

- Bench 占位：宝可梦已经实际进入 Bench；Dunsparce 只能算低奖赏缓冲或抽牌引擎。
- 进化链：Bench Abra 需要可见的 Kadabra，或可见的 Rare Candy + Alakazam；Bench
  Kadabra 需要可见的 Alakazam；本回合刚放下的 Basic 本回合不能进化。
- 能量：Abra 线的接力必须有 Psychic Energy，或当前有明确的合法附能选项；
  Enriching Energy 只有 Colorless，不能满足 Psychic 攻击费用。
- ready handoff：假设 Active 被击倒，Bench 宝可梦下一回合能完成必要进化、附能并
  发起攻击。

## 卡牌语义核对

依据 `engine/source/ptcgProgram 22/CardImpl.h`：

- Buddy-Buddy Poffin（1086）从牌库放置最多 2 只 HP≤70 的 Basic Pokémon，不能从
  手牌拿 Pokémon，也不提供能量；本卡组的 Abra/Dunsparce 合法。
- Telepath Psychic Energy（19）附到 Psychic Pokémon 时，提供 Psychic，并从牌库
  放置最多 2 只 Basic Psychic Pokémon 到 Bench；附到 Dunsparce 不触发检索。
- Dawn（1231）只把 Basic、Stage 1、Stage 2 各一张拿到手牌，不直接建立 Bench。
- Lana’s Aid（1184）可从弃牌区拿回非 Rule Box Pokémon 和 Basic Energy，但拿回后仍
  需要后续放置/附能动作；不能拿回 Fezandipiti ex。
- Kadabra 的 Psychic Draw 只有“从手牌打出并进化”时可用；不能把已经在场的
  Kadabra 当作自动抽牌资源。
- Budew 的 Itchy Pollen 只锁对手下一回合的 Item；它会阻断 Poffin/Rare Candy，
  但不阻断 Supporter、手动附能、Telepath 的附能触发、自然进化和 Ability。

## 本轮建议与采纳范围

advisor 建议最终使用 `B/E/P/R` 四状态模型，并先从低风险的“单个可验证接力”开始：

1. ready Bench Kadabra/Alakazam 优先；
2. Bench Abra 有 Psychic 且有可见进化路线时，保留 Psychic 资源；
3. 没有接力时优先 Poffin；若没有 Poffin，优先 Telepath 或直接把手牌 Abra/Dunsparce
   放入 Bench；
4. 终局奖赏闭环仍优先攻击；不可观察、无法当回合完成的 Dawn/Lana’s Aid 路线不应
   被假定为 ready。

本轮已采纳的最小策略改动：

- Poffin 不再要求 Abra/Dunsparce 出现在手牌，因为它搜索的是牌库；
- Dunsparce-only Bench 不再被视为 Abra 接力；
- Bench Abra 纳入 Psychic 附能保护，避免唯一附能被 Enriching Energy 消耗；
- 在 Poffin 不可用时，直接放下手牌 Abra/Dunsparce 或使用 Telepath 建立 Bench；
- 终局攻击使用显式奖赏闭环例外，不被接力 gate 拦截。

本轮暂不一次性重写 Dawn/Lana’s Aid 的完整 pipeline，也不把目标提高到“两个 ready
attacker”；这些变量留给后续 iteration，避免把第二回合 Powerful Hand 和整体胜率
同时压低。

## 证据边界

仓库当前保留的是 iter-05 的 metrics/cases 摘要，而不是用户最新 170 局的原始
`game_*.json`。因此不能凭汇总数据伪造那两个 miss 的具体 game/step；本轮用针对性
unit fixture 验证相同状态决策，并把用户提供的 170 局指标作为 control 记录。

## Rock Fighting Energy 补充核验

官方 `data/official/EN_Card_Data.csv` 和引擎 `CardImpl.h` 中，Rock Fighting Energy
的 ID 是 `20`，效果实现为 `NoEffectEnemyAttack`，与 Mist Energy（ID `11`）使用同一
类保护效果。Alakazam 的 `Powerful Hand`（attack `1072`）在引擎中通过
`DamageCounter` 放置伤害指示物；`State::isPreventDamageCounter()` 会使这类伤害为
0。因此这不是普通的“特殊能量价值较高”判断，而是会直接改变能否 KO 的规则分支。

本轮已将 ID `11` 和 `20` 统一进入策略的 `PROTECTIVE_DAMAGE_ENERGIES`：攻击伤害、
Boss 目标、终局奖赏闭环和 Enhanced Hammer 的目标优先级都调用同一模型，并分别增加
Mist/Rock Fighting 的策略 fixture。

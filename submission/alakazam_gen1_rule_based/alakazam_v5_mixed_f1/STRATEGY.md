# Alakazam V5 Mixed F1 Strategy Specification

`v5_mixed_f1` 不改变 V5 牌表。它的核心目标是：每回合优先形成一次有意义的
Alakazam 攻击，同时用可见状态保证下一只打手的最短攻击路径。所有选择都从
官方 simulator 返回的合法 option 中确定性地排序。

公开 notebook 与 v5_mixed 评测提供的是状态建模方向，而不是可直接复制的权重或卡组：本版本吸收
`ready now`、`next attack path`、未来攻击线、Dunsparce 引擎和目标驱动抽牌，拒绝
2-ply search、随机性以及公开 notebook 的卡组变更。

## 1. 进化与 Poké Pad

### 首回合

首回合不能进化，且本回合才放下的 Pokémon 也不能进化。因此 Poké Pad 不会为了
“提前拿到 Kadabra”而消耗资源；有 Dunsparce 时优先拿它，作为后续
`Dudunsparce + Run Away Draw` 的入口，否则拿 Abra 或其他 setup Basic。

### 第二回合

策略先判断 Active Abra 是否有 Psychic Energy（或手牌能量）、Rare Candy、
Alakazam/Hilda 的直通路线。若能在本回合完成 Alakazam 攻击，保留 Active Abra，
把 Kadabra 进化落在合法的 Bench Abra 上，让 Psychic Draw 增加找到缺件的机会。

没有直通线路时，Dawn 的 Stage 1 搜索和 Poké Pad 都服务于本回合可进化或下一回合
可自然进化的 Kadabra，而不是盲目搜索 Alakazam。

### 第三回合以后

Active Kadabra 已带 Psychic Energy、手牌有 Alakazam 时，优先直接自然进化并攻击。
`Rare Candy + Alakazam` 是第二回合从 Abra 直通的特殊线路，不是所有回合的固定
动作。进化目标选择保留 Active 的进攻机会：有直通可能时，普通 Kadabra 进化优先
落在 Bench。

## 2. 攻击连续性、抽牌与 Fezandipiti

- 场上三只 Abra 系列仍是铺场目标，但额外统计当前能攻击的 Active、已充能的
  Bench 打手、下一回合可见攻击路径、未充能攻击线和弃牌区可恢复攻击线。三只未
  充能 Abra 不等于三只可以连续交接的打手。
- 首只 Alakazam 建立前，优先恢复 V5 的真实场上攻击线节奏：手牌中的 Abra/Kadabra
  数量不能替代场上的攻击线，Poffin 不会因为 `future_count` 达标就过早转向 Dunsparce。
- 首只 Alakazam 建立后，`ready_attacker_count` 才成为主要接力 gate；已充能的
  Alakazam、Kadabra 或 Abra 都可作为下一只攻击手，恢复牌只有在手牌中真实存在时
  才能构成 next attack path。
- Alakazam 的 `Psychic Draw`、Kadabra 的 `Psychic Draw` 和 Dudunsparce 的
  `Run Away Draw` 受手牌 20 张、牌库 10 张以及“牌库至少保留剩余 Prize + 1”保护；
  已经能击倒时不为了额外手牌破坏牌库或手牌安全边界。
- 抽牌按四档判断：已有有效 KO 时直接攻击；一次明确抽牌能形成 KO 时允许；当前
  不能 KO 但下一只攻击线断裂时只保留必要动作；本回合和下回合都安全时才额外过牌。
- 只有上一回合己方确实有 Pokémon 被击倒，且当前没有可直接完成的 Alakazam
  攻击时，才考虑 Fezandipiti ex 的 `Flip the Script`。手牌只有 Xerosic、
  Enhanced Hammer、Fezandipiti 这类低动点时，Fezandipiti 优先于把 Active
  Kadabra 留在 30 点攻击。
- Fezandipiti 是两奖宝可梦；已有普通 Bench 且对手只剩两张 Prize 时，不为普通
  抽牌把它放下。

## 3. Energy、Telepath 与 Retreat

- Abra、Kadabra、Alakazam 每只默认只需要一张 Psychic Energy。已经有能量的进化
  线不会再接受第二张，除非 simulator 没有其他合法目标可选。
- Telepath Psychic Energy 只有贴到 Psychic Pokémon 时才触发搜索，搜索目标是
  Basic Psychic Pokémon。本卡组中实际目标是 Abra；即使已经贴在合规 Psychic
  目标上，也会在效果可选时执行合法搜索。
- Telepath 和 Basic Psychic 都优先贴给没有 Psychic Energy 的 Abra、Kadabra、
  Alakazam；Active 的未充能攻击线优先于 Bench 的未充能攻击线。如果手牌有 Abra
  且场上还没有 Psychic 目标，主行动会先放 Abra，再贴 Telepath。
- Active Fezandipiti ex 或 Shaymin、Bench 有带 Psychic Energy 的 Abra 系列时，
  一张能支付 Retreat 的能量可以换来本回合的有效攻击；否则不支付 Retreat 成本。
  首回合即使 option 存在也不执行无收益撤退。
- Active Dunsparce 只有在 `Trading Places` 能把已充能的 Alakazam 换到 Active
  并立即攻击时才选择该攻击；它不等同于支付 Retreat。

## 4. 恢复与 Dudunsparce 过牌

- Night Stretcher 和 Lana's Aid 都按攻击线恢复。若 Active/Bench 没有 Abra 而
  弃牌区有 Abra，先取回 Abra；否则按当前的攻击和进化缺口选择 Abra、Kadabra、
  Alakazam。Lana's Aid 会在最多三张的合法范围内尽量多取回，而不是只取一张。
- Enriching Energy 在本卡组中提供无色能量并抽 4 张。只有 Active Alakazam 已
  能攻击、场上没有尚未充能的 Abra 系列、Dudunsparce 的 `Run Away Draw` 仍可用，
  且牌库通过保护线时，才把这次手填转化为额外过牌；下一只打手仍有能量需求时，
  攻击线优先。

## 5. 伤害目标和资源保护

- 对手 Active 带 Mist Energy 时，按当前官方 runtime 的实际 `DamageCounter`
  行为把 Alakazam 伤害视为 0；Enhanced Hammer 先处理 Active 的 Mist/特殊能量。
- 若 Active 不能击倒，Boss's Orders 只在拉出可击倒目标或提高 Prize 价值时使用。
- Xerosic's Machinations 只有在第二回合以后、对手手牌至少四张，且当前没有更高
  优先级 KO、进化或防守动作时使用。可见的 lethal-to-non-lethal 判断仍然优先，
  但对手是 Alakazam deck 或手牌至少八张时保留 V5 的高置信 fallback，不假定知道
  对手隐藏手牌内容。

## 6. 公开策略的边界与运行时事实

- Telepath Psychic Energy 的官方引擎 effect 条件是贴到 Psychic Pokémon。场上没有
  Psychic 目标、手里有 Abra 和 Telepath 时，先放 Abra，再在下一次合法主行动贴能量；
  不能因为 review 中的自然语言描述而假设 Dunsparce 也能触发搜索。
- 当前 runtime 将带 Mist Energy 的目标视为 Alakazam `Powerful Hand` 的 0 伤害目标，
  因此先考虑能改变结果的 Enhanced Hammer；这一条来自本仓库的 engine/replay 事实，
  不被公开 notebook 的一般规则覆盖。
- 没有新的 Kaggle replay 前，`first_alakazam_turn`、`first_effective_attack_turn`
  等指标只是后续观测项，不把 notebook 自报 Elo 或胜率当作本版本成绩。

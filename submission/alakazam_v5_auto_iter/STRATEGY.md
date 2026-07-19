# Alakazam V5 Strategy Specification

V5 的核心目标是：每回合优先形成一次有意义的 Alakazam 攻击，同时为下一只
Abra 系列保留能量和进化资源。所有选择都从官方 simulator 返回的合法 option
中确定性地排序。

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

## 2. 抽牌与 Fezandipiti

- Alakazam 的 `Psychic Draw`、Kadabra 的 `Psychic Draw` 和 Dudunsparce 的
  `Run Away Draw` 受手牌 20 张、牌库 10 张保护；已经能击倒时不为了额外手牌
  破坏牌库或手牌安全边界。
- Dudunsparce 使用 `Run Away Draw` 后会把自身和所附卡洗回牌库；牌库只剩 2–3
  张时可以抽出剩余牌并安全回收，因此只在牌库至多 1 张时额外禁止该能力。
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
- Active Fezandipiti ex 或 Shaymin、Bench 有带 Psychic Energy 的 Alakazam 时，
  一张能支付 Retreat 的能量可以换来本回合的 Alakazam 攻击；否则不支付 Retreat
  成本。首回合即使 option 存在也不执行无收益撤退。
- Active Dunsparce 只有在 `Trading Places` 能把已充能的 Alakazam 换到 Active
  并立即攻击时才选择该攻击；它不等同于支付 Retreat。

## 4. 恢复与 Dudunsparce 过牌

- Night Stretcher 和 Lana's Aid 都按攻击线恢复。若 Active/Bench 没有 Abra 而
  弃牌区有 Abra，先取回 Abra；否则按当前的攻击和进化缺口选择 Abra、Kadabra、
  Alakazam。Lana's Aid 会在最多三张的合法范围内尽量多取回，而不是只取一张。
- Enriching Energy 在本卡组中提供无色能量并抽 4 张。只有 Active Alakazam 已
  能攻击、场上没有尚未充能的 Abra 系列、且有 Dudunsparce 时，才把这次手填转化
  为 `Enriching Energy -> Run Away Draw`；下一只打手仍有能量需求时，攻击线优先。

## 5. 伤害目标和资源保护

- 对手 Active 带 Mist Energy 时，按当前官方 runtime 的实际 `DamageCounter`
  行为把 Alakazam 伤害视为 0；Enhanced Hammer 先处理 Active 的 Mist/特殊能量。
- 若 Active 不能击倒，Boss's Orders 只在拉出可击倒目标或提高 Prize 价值时使用。
- Xerosic's Machinations 在第二回合以后、对手手牌至少四张且没有更重要的进化、
抽牌或持续攻击动作时使用，尤其用于压低 Alakazam 内战对手的手牌伤害。

## 6. Iono 定向 Boss（当前 v44）

当场上识别到 Iono 的 Voltorb/Tadbulb/Bellibolt/Wattrel/Kilowattrel，且 Active 无法
击倒、Bench 的一奖 Voltorb/Tadbulb/Wattrel 能被当前攻击击倒时，允许 Boss's Orders
进行奖赏交换；其他对手仍使用通用的“击倒或更高奖赏”判断。该条件在 v44 两批完整
对照中均领先同批 control，当前暂保留。

v45 曾把该条件收窄到 Bellibolt ex Active，但合并两批仅多 3 胜/960 局且方向不稳定，
已回退，避免把 Iono 局部提升误当作全局收益。

## 7. 多局状态隔离（当前 v46）

评测器会复用 agent module 运行多局；每局开始收到 `select=None` 时清空多步效果的
serial 进度，避免上一局的 Dawn/Hilda/Poffin 选择状态泄漏到下一局。这是生命周期
正确性修复，不代表独立胜率提升；当前策略胜率仍以 opponent 矩阵实测为准。

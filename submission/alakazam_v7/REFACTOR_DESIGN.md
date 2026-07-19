# Alakazam V7 Refactor Design

> 状态：设计边界已确认，可进入实现计划阶段。
>
> 基线：`submission/alakazam_v6/`。
>
> 边界：固定 V6 的 `deck.csv`，只重构策略、状态模型和动作选择；不修改卡组构成。

## 0. 这份文档的作用

V7 不是 V6 的几个分数调整，也不是为了单独追求“第二回合一定攻击”。V7 的目标是
把我们已经确认的宝可梦 TCG 规则和这套 Alakazam 卡组的实际操作逻辑，重新组织成一
套可以解释、可以测试、可以从 replay 验收的策略。

在开始改动 `main.py` 之前，需要先确认本文中的以下内容：

- 什么是本回合必须完成的攻击路线；
- 哪些准备动作会因为攻击终止回合而失去机会；
- 哪些卡牌只是“当前合法”，但不值得现在使用；
- 低阶段攻击、牌库保护和接力攻击之间的边界；
- 每一个用户反馈样例中，V6 的错误动作和 V7 的验收动作。

本文先把设计写完整，不代表代码已经实现。

## 1. V7 的目标与不变项

### 1.1 总目标

在不违反宝可梦 TCG 规则、回合预算和牌库生存预算的前提下，最大化未来一到两回合
完成有效 Prize 推进的概率。

如果本回合不能有效取奖，则保护一条现实存在的进化、能量和换位路线，而不是单纯
增加“场上看起来很高级的宝可梦”数量。

### 1.2 固定边界

- `deck.csv` 与 V6 完全相同。
- V7 只修改 `main.py` 中的策略和状态判断，以及必要的策略说明和验收测试。
- 不针对某一个 opponent ID 写整套 matchup 特例。
- 只使用 observation 中实际可见的 Active、Bench、手牌、弃牌、Prize、牌库数量、
  能量、伤害和合法 option。
- 不把本地模拟胜率当作官方 Kaggle 胜率；本地模拟用于动作行为、合法性和指标回归。
- 允许对卡牌规则中的稳定效果使用窄范围硬编码；本版本只硬编码 `Budew (235)` 的
  `Itchy Pollen (attackId=323)` Item Lock，不扩展为按对手构筑选择策略。

### 1.3 V6 中应保留的正确基础

V6 已经建立的以下内容不应被回退：

- 攻击是本回合的终止提交；宣告攻击后不能再进化、附能、使用 Supporter、使用普通
  Ability 或铺 Bench。
- 进化目标必须在本回合开始前已经在场，且同一只 Pokémon 本回合不能重复进化。
- 一回合只能使用一次 Supporter、手动附一次手牌能量和一次 Retreat。
- `TurnMemory` 记录每个 serial 的回合开始状态、放入场上的回合、进化回合和本回合
  已消耗的预算。
- 牌库 15/10 张是策略保护线，不是游戏规则；当动作能够完成最终 Prize 闭环时，
  保护线可以被终局收益覆盖。
- `Lana's Aid` 在需要从弃牌区同时恢复 Pokémon 和 Psychic Energy 时，优先于
  “Supporter 搜索加 Night Stretcher”的两步组合。
- Trading Places 不作为普通换位手段；使用它本身会消耗攻击机会，不能再让换到
  Active 的 Alakazam 在本回合攻击。

## 2. V6 反馈复盘后的核心问题

### 2.1 把卡牌类别当成攻击路线

V6 的 `_recovery_needs()`、`_needs_attack_line()` 和相关搜索逻辑，仍可能把手牌或
弃牌区中的 Kadabra/Alakazam 当成“已经拥有攻击路线”。但一个孤立的 Stage 1 或
Stage 2 并不说明场上存在合法的 Basic 来源，也不说明下一回合能完成进化。

真正需要记录的是：

```text
Basic 来源 -> 合法进化时机 -> Stage 1/Stage 2 -> Psychic Energy -> Active 方式
-> 本回合或下一回合的有效攻击
```

### 2.2 进化目标选择没有先确定“哪一只要完成攻击”

V6 的 `_choose_evolution_target()` 在有 Alakazam 和 Rare Candy 时偏向 Active Abra，
这会破坏另一只 Abra 的自然进化和 Kadabra Ability 过牌机会。

这次反馈要求 V7 先区分两类场面：

1. **Active Abra 是本回合的直接 Alakazam 目标**：保留它，不先把它自然进化成 Kadabra；
   先让另一只合法的 Bench Abra 进化成 Kadabra并使用 Ability，再用 Rare Candy 加
   Alakazam 完成 Active 的攻击路线。
2. **Active 已经是带 Psychic Energy 的 Kadabra**：先自然进化 Active 到 Alakazam，
   再进化合法的 Bench Abra 到 Kadabra并使用 Ability，最后让 Active Alakazam 攻击。

这里的判断对象不是“场上哪一只等级更低”，而是“本回合哪一只必须保留为有效攻击
目标”。

### 2.3 第一只 Alakazam 完成后仍然消耗多张 Rare Candy

V6 把多个合法 Rare Candy option 视为连续的 setup 动作，导致已经有可攻击的
Alakazam 后，又把第二只、第三只 Abra 直接变成 Alakazam。

V7 的默认设计是：

- 每个回合先定义一个 `primary_attacker`；
- Rare Candy 只有在它能完成当前 primary attacker 的有效攻击路线时，才进入高优先级；
- primary attacker 已经可攻击后，第二只 Alakazam 不是本回合的默认目标；
- 后续 Bench Abra 优先保留为自然进化 Kadabra 的资源，利用 Ability 增加手牌；
- 只有当前没有可攻击的 Alakazam，或者第二只进化能立即完成确定 Prize/恢复失败，
  才允许突破这个默认保留规则。

Rare Candy 在规则上可以一回合使用多张，但 V7 可以出于策略目标限制其使用；这是一
个需要用户确认的策略边界，不应伪装成 TCG 规则。

### 2.4 “有攻击 option”不等于“现在应该攻击”

在 Mega Venusaur/Ogerpon 对局 `86877253` 中，V6 在 t2、t5、t7、t9、t11 先后用
Abra、Kadabra、Alakazam、Kadabra、Dunsparce 攻击，但没有一次推进 Prize。

这说明攻击候选必须至少分成三类：

1. `guaranteed_prize_attack`：当前 Active 能确定 KO，或 Boss 后能确定 KO；
2. `alakazam_pressure_attack`：Active Alakazam 已经准备好，当前没有更高价值的
   进化、接力、恢复或过牌动作；即使不能立即 KO，也可以正常攻击；
3. `fallback_attack`：Kadabra 或 Dunsparce 的非 KO 攻击，只能在本回合没有能增加
   未来有效攻击概率的准备动作时使用；Abra 不属于任何攻击候选。

Abra 的 Teleportation Attack (`attackId=1070`) 在 V7 中直接排除。它虽然合法，但会
强制把另一只 Pokémon 换到 Active，消耗当前回合的攻击机会，不符合这套卡组的能量
节奏。因此只要 Active 是 Abra，就不选择任何 Abra attack option，不仅仅是把某个
attack ID 降权。

### 2.5 Item Lock 时没有切换到第二优先级

当对手的 Budew 造成 Item Lock，Rare Candy 路线不可用时，V7 不应继续等待 Item，
也不应把 Hilda 找到的 Kadabra 留在手里。

此时的优先级应切换为：

1. 对合法的 Active Abra 做自然进化到 Kadabra；
2. 如果有其他合法 Abra，也继续自然进化和铺场；
3. 把 Enriching Energy 给 Dunsparce，进化为 Dudunsparce 后使用 Run Away Draw；
4. 在牌库保护线允许时，用获得的手牌寻找后续资源；
5. 最后才考虑低阶段攻击。

Item Lock 的识别使用已确认的卡牌效果硬编码：只有在观察到对手的
`Budew (cardId=235)` 实际宣告 `Itchy Pollen (attackId=323)` 后，才将“对手下一个
回合不能从手牌使用 Item”写入状态。不能因为场上存在 Budew，或因为某次 observation
里暂时没有 `Rare Candy` 的合法 PLAY option，就提前判定 Item Lock。

锁定状态只覆盖紧接着的我方一个回合；该回合内把 Rare Candy 视为不可用路线，回合
结束后清除。模拟器仍是最终合法性来源，硬编码只负责把策略从“等待 Rare Candy”切换
到自然进化、Dudunsparce 过牌和恢复资源等第二优先级。

## 3. V7 的状态模型

V7 仍然使用单文件 submission 结构，但策略内部需要把几个互不相同的概念分开。

### 3.1 回合预算

```text
TurnBudget
├─ supporter_available
├─ manual_energy_available
├─ retreat_available
├─ attack_open
└─ evolution_serials_used
```

对手效果另外记录在局内状态中：

```text
OpponentEffectMemory
└─ item_lock_active_for_turn: bool
```

当且仅当上一段对手行动被归一化为
`attacker_id=235 + attack_id=323` 时，将该字段置为 `true`；进入我方下一个回合时
读取，结束该回合后清除。不能用 Budew 是否在 Active、对手手牌数量或 Rare Candy
PLAY option 是否缺失来替代这个事件记录。

这些是硬约束，不能被一个较高的收益分数抵消。

### 3.2 Pokémon 进化状态

每一只 Pokémon 以 serial 作为实例身份，至少记录：

```text
PokemonState
├─ serial
├─ current_id
├─ area: ACTIVE | BENCH | HAND | DISCARD | PRIZE | DECK
├─ was_in_play_at_turn_start
├─ appeared_this_turn
├─ evolved_this_turn
├─ pre_evolution_ids
├─ psychic_energy_attached
└─ legal_attack_now
```

`current_id == Alakazam` 只说明卡面已经是 Alakazam；`legal_attack_now` 还需要同时
考虑能量、Active 位置、攻击 option 和当前对手目标。

### 3.3 攻击路线

每个回合先建立一个或多个 `AttackRoute`，而不是只计算 ready attacker 数量：

```text
AttackRoute
├─ target_serial
├─ source: ACTIVE | BENCH | HAND | DISCARD
├─ missing_evolution
├─ missing_energy
├─ next_legal_turn
├─ active_entry: CURRENT | RETREAT | ABILITY_SWITCH | KO_PROMOTION
├─ estimated_damage
├─ guaranteed_prize
└─ handoff_after_attack
```

例如，手里有 Alakazam 不代表存在完整路线；只有当场上存在可合法进化的 Abra，或
存在合法的回收、铺场和进化来源时，才可以建立 `AttackRoute`。

### 3.4 资源账本

从固定 `deck.csv` 自动得到每种卡的总量，并持续核对：

```text
ResourceLedger
├─ hand
├─ active
├─ bench
├─ discard
├─ known_prize
├─ deck_count
└─ unknown_prize_lower_bound
```

Abra、Kadabra、Alakazam 和 Psychic Energy 的位置会影响恢复判断。不能只看到一张
Alakazam 在手里，就认为恢复 Alakazam 能够马上产生攻击。

### 3.5 本回合目标

每次主行动重新选择一个目标：

```text
TurnIntent
├─ TERMINAL_WIN
├─ GUARANTEED_PRIZE
├─ BUILD_PRIMARY_ALAKAZAM
├─ BUILD_BENCH_KADABRA_DRAW
├─ RESTORE_HAND_AND_ATTACK_ROUTE
├─ PREPARE_HANDOFF
├─ XEROSIC_DISRUPTION
├─ ALAKAZAM_PRESSURE
├─ FALLBACK_ATTACK
└─ END_TURN
```

Intent 不是固定状态机。每使用一次 Item、Ability、Supporter、进化或附能，策略都要
重新读取 observation 并重算剩余路线。

## 4. V7 主行动流程

V7 的主流程暂定如下：

```text
读取 observation
  -> 同步 TurnMemory 和 ResourceLedger
  -> 过滤规则上不可用的路线
  -> 先检测本回合的最终 Prize 闭环
  -> 选择最高优先级 TurnIntent
  -> 选择能推进该 Intent 的第一步合法 option
  -> action 后重新计算，不缓存整回合计划
```

### 4.1 第一优先级：终局闭环

低牌库保护不能阻止本回合可以拿最后奖赏的路线。需要检查的不只是当前 Active 的
攻击，而是有限长度的可观察组合路线：

```text
Dudunsparce Active
  -> Run Away Draw
  -> 选择 Bench Alakazam Active
  -> Powerful Hand
  -> 确定拿最后 Prize
```

或者：

```text
带 Retreat 代价的 Active
  -> Retreat
  -> 选择带 Psychic Energy 的 Bench Alakazam
  -> Powerful Hand
  -> 确定拿最后 Prize
```

这里必须按实际卡牌语义区分：`Run Away Draw` 属于 `Dudunsparce (66)`；
`Dunsparce (305)` 只有 Trading Places 和 Ram。如果 Active 确实是 Dunsparce，则
只能评估 Retreat，不能虚构 Run Away Draw。

Trading Places 不属于这条闭环，因为它本身已经消耗攻击提交，不能在其后继续使用
Alakazam 攻击。

### 4.2 第二优先级：确定 Prize

如果 Active 可以 KO，通常直接保留 Active 目标；如果 Active 不能 KO，而 Bench 有
可以被当前攻击确定 KO 的目标：

1. 使用 Boss's Orders；
2. 在所有确定 KO 的 Bench 目标中选择剩余 HP 最高者；
3. 之后发起攻击并取奖。

如果 Boss 不会产生确定 Prize，不能因为 Boss option 合法就消耗 Supporter。

### 4.3 第三优先级：完成 primary attacker

#### Active Kadabra

当 Active Kadabra 已在回合开始前入场、带 Psychic Energy、手里有 Alakazam 时：

1. 先自然进化 Active Kadabra 到 Alakazam；
2. 再处理合法的 Bench Abra -> Kadabra；
3. 使用新的 Kadabra Ability 过牌；
4. 完成其它不消耗攻击提交的准备；
5. 用 Active Alakazam 攻击。

#### Active Abra

当 Active Abra 有 Psychic Energy、手里有 Rare Candy + Alakazam，Bench 还有另一只
合法的 Abra 时：

1. 不先把 Active Abra 自然进化成 Kadabra；
2. 先把另一只 Bench Abra 进化成 Kadabra并使用 Ability；
3. 将 Rare Candy + Alakazam 保留给 Active Abra；
4. 完成 Active Alakazam 的进化和攻击。

如果 Active Abra 没有直接 Alakazam 路线，则再比较 Active 自然进化、Bench 自然
进化和恢复手牌，不允许机械地套用上面的分支。

### 4.4 第四优先级：一次手填能量的分配

默认规则：

- 当前 Active 攻击路线缺 Psychic Energy 时，先补当前 primary attacker；
- Active Alakazam 已经可以有效攻击，且有 Enriching Energy 和 Dunsparce/Dudunsparce
  抽牌路线时，优先把 Enriching Energy 转化为 Dudunsparce 过牌；
- 没有 Enriching Energy 时，把其它可用 Psychic Energy 贴给没有 Psychic Energy 的
  Abra 线宝可梦；
- 不把 Enriching Energy 当作 Abra/Kadabra 的 Psychic 攻击能量。

Enriching Energy 的过牌净消耗需要按实际附着卡计算：Dudunsparce 抽三张，再把自身
和所有附着卡洗回牌库。若带一张 Enriching Energy，牌库净减少可以为零；这使它在
牌库安全线以上时成为低风险的手牌恢复手段。

### 4.5 第五优先级：恢复与过牌

恢复动作从下一次现实攻击反向判断：

- 没有可进化 Basic 时，恢复 Alakazam 单卡不能算完成路线；
- 需要同时恢复 Pokémon 和 Psychic Energy 时，优先 Lana's Aid；
- 只有 Night Stretcher 时，才比较 Supporter 搜索加 Night Stretcher 的组合；
- Fezandipiti ex、Kadabra Ability、Dudunsparce Ability 和其它非 Supporter 过牌，
  只要能提高当前攻击路线的完成概率，就应在非 KO 弱攻前执行；
- 牌库明显高于 10 张时，因 Unfair Stamp 导致手牌过少，应优先恢复手牌和伤害所需的
  资源，而不是用 Kadabra 30 点结束回合。

### 4.6 第六优先级：Xerosic

当满足以下条件时，Xerosic 可以在攻击前使用：

- Active Alakazam 已经可以攻击；
- 对手 Active 已受到伤害但无法被当前攻击击倒；
- 对手手牌至少 6 张；
- 没有 Boss 能制造确定 KO；
- 本回合没有更高价值的 primary attacker、接力或恢复动作。

对手手牌只有 4 张左右时，默认保留 Xerosic，不为了轻微干扰消耗唯一 Supporter。

### 4.7 最后才是攻击

攻击排序暂定为：

1. 终局/确定 Prize；
2. Active Alakazam 的正常攻击；
3. Kadabra 或 Dunsparce 能确定 KO 的攻击；
4. 没有任何能增加未来有效攻击概率的准备动作时，才允许 Kadabra/Dunsparce 的低阶段
   非 KO 攻击；
5. Abra 的所有攻击 option、Teleportation Attack 和 Dunsparce Trading Places 永不
   选择。

“Active Alakazam 可以攻击”不是绝对要求它必须先攻击；如果仍有明确的 Bench
Kadabra Ability、Primary route 或恢复动作，先完成这些动作。但 Hilda/Dawn 不能只
为了“后场看起来更完整”而消耗本回合唯一 Supporter。

## 5. 用户反馈对应的验收样例

测试不应断言 simulator option 的数组下标，因为 option 顺序可能改变；应将选中的
option 归一化为 `type + cardId/attackId + target serial/area`。

### Case V7-01：Active 与 Bench 的进化目标

**局面：** Active Abra 与 Bench Abra 都已在回合开始前入场；Active 有 Psychic Energy；
手里有 Rare Candy、Alakazam、Kadabra；Bench Abra 没有能量。

**V6 问题：** 直接选择 Active Abra -> Kadabra，或者在完成第一只 Alakazam 后继续
消耗 Rare Candy。

**V7 断言：**

- 第一个自然进化目标是 Bench Abra -> Kadabra；
- 使用 Kadabra Ability；
- Active Abra 保留为 Rare Candy + Alakazam 目标；
- 第一只 Alakazam 完成后，不再默认进化第二只 Alakazam。

### Case V7-02：Active Kadabra 的自然进化

**局面：** Active Kadabra 带 Psychic Energy，Bench 有合法 Abra，手里有 Alakazam。

**V7 断言：**

- 先 Active Kadabra -> Alakazam；
- 再 Bench Abra -> Kadabra；
- 使用 Kadabra Ability；
- 最后使用 Active Alakazam 攻击。

### Case V7-03：Item Lock 下的自然进化和 Enriching Energy

**局面：** 上一段实际观察到对手 `Budew (235)` 使用 `Itchy Pollen (323)`；Active
Abra 可自然进化；Hilda 找到 Kadabra 和 Enriching Energy；Bench 有 Dunsparce；牌库
高于保护线。当前回合的 Item Lock 使 Rare Candy 路线不可用，合法 option 的变化只是
引擎层结果，不是识别 Item Lock 的依据。

**V6 问题：** 找到 Kadabra 后没有立即自然进化，或者没有把 Enriching Energy 转给
Dunsparce 进行 Dudunsparce 过牌。

**V7 断言：**

- Hilda 的搜索目标要服务于自然进化和过牌；
- 下一步自然进化 Active Abra；
- Enriching Energy 贴给 Dunsparce；
- Dudunsparce 的 Run Away Draw 在安全时执行；
- 不因为当前有弱攻击 option 就提前攻击。

### Case V7-04：终局换位优先于牌库保护

**局面：** 我方剩余最后 Prize；当前 Active 是 Dudunsparce 或可 Retreat 的非攻击者；
Bench 有带 Psychic Energy 的 Alakazam；Alakazam 对对手目标可以确定 KO。

**V6 问题：** 牌库 gate 阻止 Run Away Draw/换位，放弃本回合胜利。

**V7 断言：**

- 识别 `Run Away Draw -> Alakazam -> Powerful Hand` 的终局闭环，或识别 Retreat
  -> Alakazam -> Powerful Hand 的闭环；
- 允许这条路线越过牌库保护线；
- 不选择 Trading Places。

### Case V7-05：禁止 Abra 攻击和 Teleportation Attack

**局面：** Active Abra 带 Psychic Energy，合法 option 中包含 `attackId=1070`，
Bench 存在其它 Pokémon，手里存在 Enriching Energy 或其它准备动作。

**V7 断言：** 不选择任何 Abra attack option，包括 Teleportation Attack；继续评估
进化、附能、过牌、合法 Retreat 和结束回合等动作。

### Case V7-06：被 Unfair Stamp 后先恢复攻击路线

**局面：** 已牺牲一只 Alakazam；当前没有 Alakazam，手牌较少；手里有 Fezandipiti
ex，场上或手里存在 Dudunsparce/Enriching Energy 路线；对手 Active 的 HP 需要更高
的 Powerful Hand 伤害才能 KO。

**V6 问题：** 直接用 Kadabra 的低伤害攻击结束回合，放弃 Fezandipiti 和其它过牌。

**V7 断言：**

- 先使用 Fezandipiti Ability 或其它非 Supporter 过牌；
- 继续寻找 Alakazam、Psychic Energy 和提高 Powerful Hand 伤害的手牌；
- 只有这些动作不可能改善当前或下一次攻击时，才允许 Kadabra 非 KO 攻击。

### Case V7-07：没有 Abra 底座时恢复 Dudunsparce 过牌

**局面：** Active 已经是 Alakazam，Bench 有 Dunsparce，但场上没有 Abra；手牌资源
被 Unfair Stamp 压低，手里有 Hilda、Kadabra/Alakazam 等不能单独形成新攻击的卡；
牌库仍高于保护线。

**V6 问题：** 用 Hilda 搜索孤立的 Alakazam，或在没有改善手牌伤害的情况下结束回合。

**V7 断言：** Hilda 优先寻找 Dudunsparce 与 Enriching Energy；将 Enriching Energy
贴给 Dunsparce，完成进化并使用 Run Away Draw，先恢复手牌和后续攻击资源。

### Case V7-08：Active Shaymin 下的两只 Bench Abra 目标分工

**局面：** Active 是不能有效攻击的 Shaymin，需要 Retreat；Bench 有两只已经在回合
开始前入场的 Abra，其中一只已经贴 Psychic Energy，另一只没有能量；手牌有
Alakazam，并且可能通过过牌获得 Rare Candy。

**V6 问题：** 先把可能承担 Rare Candy + Alakazam 直通的 Abra 进化成 Kadabra，或
把手填能量和自然进化错误地消耗在同一只目标上。

**V7 断言：** 先决定带能量的 Abra 作为直通 Alakazam 目标；让另一只无能量 Abra
自然进化到 Kadabra并使用 Ability；Retreat Shaymin 后，使用 Rare Candy + Alakazam
完成可攻击的 Bench 目标。若直通资源尚未齐全，则保留该分工并继续寻找资源，不提前
把直通目标变成 Kadabra。

### 5.1 Replay case 映射与实现后的汇报格式

V7-01 至 V7-08 直接来自 `IMPROVEMENT.md` 中前五局 replay 的具体反馈，不是脱离
对局的理论样例。实现和验收时保留以下映射：

| Replay case | 来源反馈 | V7 必须纠正的行为 |
|---|---|---|
| `R1-A` / `V7-01` | 第一局第 1 条、第二局第 1 条 | 无能量的 Bench Abra 先自然进化并使用 Kadabra Ability；保留带 Psychic Energy 的 Active Abra 给 Rare Candy + Alakazam。 |
| `R1-B` / `V7-01 + V7-02` | 第一局第 2、3 条 | 第一只 Alakazam 能攻击后，不连续消耗 Rare Candy 做第二、第三只；已有可攻击的 Bench Alakazam 时，Retreat 后用它攻击，不用 Active Kadabra 弱攻。 |
| `R1-C` / `V7-07` | 第一局第 4 条 | 没有 Abra 基础目标时，不用 Hilda 继续找孤立 Alakazam；优先把 Dudunsparce 与 Enriching Energy 转化为过牌资源。 |
| `R2` / `V7-08` | 第二局第 1 条 | 不把贴能量和自然进化 Kadabra 机械地放在同一只可能的直通目标上；先决定哪只 Abra 走 Rare Candy + Alakazam。 |
| `R3` / `V7-03` | 第三局第 1 条 | 实际观察到 Budew 使用 Itchy Pollen 后，放弃 Rare Candy 路线，立即自然进化，并把 Enriching Energy 给 Dunsparce/Dudunsparce 过牌。 |
| `R4` / `V7-04` | 第四局第 1 条 | Dudunsparce 的 Run Away Draw 或合法 Retreat 能把带能量的 Bench Alakazam 送到 Active 且完成最后 KO 时，越过牌库保护线完成胜利。 |
| `R5-A` / `V7-05` | 第五局第 1 条 | Active Abra 永不使用 Teleportation Attack；即使它是合法 option，也继续评估进化、附能、过牌和 Retreat。 |
| `R5-B` / `V7-06` | 第五局第 2 条 | Unfair Stamp 后手牌不足时，先用 Fezandipiti、Enriching Energy 和 Dudunsparce 过牌找回 Alakazam/伤害，不直接用 Kadabra 非 KO 攻击结束回合。 |

这里的 `R1` 至 `R5` 指用户反馈中的第 1 至第 5 局；不在文档中臆造 episode ID 或
回合号。若同一条反馈在多个 replay 中出现，以 observation 的实际状态作为 fixture
输入，而不是以对手名称作为判断条件。

实现完成后的每个 case 均按以下格式汇报，且动作使用归一化描述，不依赖 option 数组
下标：

```text
Case: R3 / V7-03
局面: Active/Bench、手牌关键资源、牌库数量、对手最近一次效果
V6 实际动作: <动作链>
V7 首选动作: <type + card/attack + target>
V7 后续计划: <下一次重规划后的动作链>
验收: PASS/FAIL + 未通过的具体优先级
```

这样可以区分“第一步是否纠正”和“完整动作链是否完成”，也能在某一步合法 option
发生变化时，继续验证策略意图而不是被数组顺序干扰。

## 6. 本地评测指标与回归要求

V6 基线：完整动作 trace 中，第二个我方回合使用 Powerful Hand 为 `11/64 = 17.2%`；
排除第二回合前结束的对局后为 `11/62 = 17.7%`。

V7 每次评测至少记录：

- 全样本 `turn_2_powerful_hand_rate`；
- 进入第二回合后的条件概率；
- 第二回合是否选择了合法的 Powerful Hand；
- V7-01 至 V7-08 的行为断言通过率；
- Abra Teleportation Attack、Trading Places 和非 KO 低阶段攻击次数；
- 首只 Alakazam 后是否存在现实的下一只 handoff route；
- agent error、非法 action 和跨局状态泄漏。

第二回合 Powerful Hand 是全局观察量，不是唯一目标。只有在铺场、进化节奏、攻击
连续性和规则合法性没有明显回退时，指标提升才算 V7 改进。

## 7. 已确认的设计边界

以下边界已经结合规则文档、V6 replay 和用户反馈逐项确认，V7 实现应直接按此执行，
不再把它们留作隐含的策略分支。

### 7.1 已确认：Active Abra 与 Bench Abra 的能量分配边界

用户已确认：当 Active Abra 已有 Psychic Energy、Bench Abra 没有能量、手里有 Rare
Candy + Alakazam 时，始终优先按以下顺序安排：

```text
Bench Abra -> Kadabra -> Ability
Active Abra -> Rare Candy + Alakazam
```

来安排。Active Abra 保留为本回合的直接 Alakazam 攻击目标，不能因为它当前也有自然
进化 option，就先把它变成 Kadabra。

只有当 Active Abra 不具备完整的 Rare Candy + Alakazam 路线时，才重新比较 Active
自然进化、Bench 自然进化和其它攻击线的能量需求。

### 7.2 已确认：第二只 Bench Abra 保留 Rare Candy

用户已确认：当第一只 Alakazam 已经可以攻击，而第二只 Bench Abra 也可以使用
Rare Candy 直接进化时，第二只 Bench Abra 不使用 Rare Candy。

如果这只 Bench Abra 可以通过手牌自然进化到 Kadabra，则本回合按以下顺序处理：

1. Bench Abra -> Kadabra；
2. 使用 Kadabra Ability 增加手牌；
3. 保留 Rare Candy 和 Alakazam；
4. 等下一回合再将这只 Kadabra 自然进化成 Alakazam。

Rare Candy 只有在第一只 Alakazam 不再可用、第二只进化能在本回合立即制造确定
Prize，或没有其它更有价值的自然进化、Ability、恢复和过牌动作时，才进入候选。

### 7.3 已确认：第四局前场是 Dudunsparce

用户已确认第四局反馈中的前场 Pokémon 是 `Dudunsparce (66)`，因此该场面可以建立
以下终局路线：

```text
Dudunsparce Active
  -> Run Away Draw
  -> 选择带 Psychic Energy 的 Bench Alakazam Active
  -> Powerful Hand
```

这条路线可以在牌库保护线以下被允许，但必须确认最终攻击能够确定完成 Prize 闭环。
`Dunsparce (305)` 仍然不具备 Run Away Draw；Trading Places 仍然不作为这条路线的
替代方案。

### 7.4 已确认：Abra 永不攻击，Kadabra 弱攻是最后手段

用户已确认：Abra 不进入任何攻击候选。原因是 Abra 的攻击会强制进行换位，而我们
不希望为了 10 点伤害破坏下一回合的 Active、能量和接力规划。

Kadabra 的非 KO 攻击仍然只在本回合没有任何能增加当前或下一次有效攻击概率的合法
动作时作为最后手段。对手下一回合可能击倒当前 Active，不能单独把 Kadabra 的弱攻
提升为优先动作；除非该攻击能够确定 KO 或形成已经确认的 Prize 闭环。

### 7.5 已确认：Budew 的 Itchy Pollen 是 Item Lock 硬编码来源

用户已确认：当前环境中限制 Rare Candy 的主流方式可直接按 Budew 的
`Itchy Pollen` 处理，因此 V7 使用窄范围硬编码，而不是试图从缺失的合法 option
反推所有可能的锁牌效果。

固定映射为：

```text
BUDEW_CARD_ID = 235
ITCHY_POLLEN_ATTACK_ID = 323
```

只有观察到对手实际使用这张攻击后，才在紧接着的我方回合设置 Item Lock。该回合：

1. Rare Candy 不进入策略候选，即使它在手牌中；
2. Active/Bench Abra 优先走合法的自然进化；
3. Enriching Energy 优先服务于 Dunsparce/Dudunsparce 的过牌路线；
4. 其它恢复、过牌和终局判断继续按通用策略执行。

锁定状态不是 matchup 分支，也不因 Budew 留在场上而持续；它只表示这一个真实可见的
卡牌效果窗口。

## 8. 实现前的验收标准

V7 实现需要满足：

1. `deck.csv` 与 V6 完全一致；
2. V7-01 至 V7-08 都能在构造的 observation fixture 中产生预期动作；
3. 所有动作仍经过 simulator option 合法性过滤；
4. 不选择任何 Abra attack option、Teleportation Attack 或 Trading Places；
5. 终局换位路线不会被普通牌库 gate 阻断；
6. 第二回合 Powerful Hand 指标相对 V6 有可解释的变化；
7. 不能以第二回合指标提升为由牺牲后续 handoff、牌库生存或 Prize 推进；
8. 完成完整本地验证后，再进行 Kaggle 实验，不在设计阶段提交 V7。

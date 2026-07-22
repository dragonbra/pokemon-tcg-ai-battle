# Alakazam Agent 重构设计

## 0. 文档定位

本文档定义 `alakazam_v8_luna_deck_opt` 的目标策略形态，用作下一轮 Agent 重构和
自动迭代的指导规格。它描述的是 Agent **应该如何思考**，不是当前
`main.py` 的实现解释，也不要求保留现有函数、分数或分支结构。

本文档依赖以下资料：

- [`DECK_NOTES.md`](DECK_NOTES.md)：V8 实际卡表和逐张卡牌语义；
- [`STRATEGY.md`](STRATEGY.md)：现有 V7 AutoIter 继承策略和 Item Lock 边界；
- [`CURRENT_OBJECTIVES.md`](CURRENT_OBJECTIVES.md)：V8C-01 至 V8C-10 的任务与验证指标；
- [`pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`](../../../docs/reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md)：规则硬约束和 V6 已确认语义。

目标不是让 Agent 更频繁地使用某一张牌，而是让它在每个己方回合结束时，都已经
完成本回合仍然有价值的准备，并把攻击作为经过检查的不可逆提交。

---

## 1. 最终目标形态

### 1.1 卡组真正要建立的状态

这副牌的最终形态不是“尽快把一只 Alakazam 做出来”，而是建立一条可持续的攻击流水线：

```text
当前可攻击的 Alakazam
        |
        |  本回合攻击前完成所有有价值的准备
        v
下一只可进化、可附能、可接班的 Abra 线
        |
        v
牌库和弃牌区中仍然存在现实的恢复与搜索路径
```

一个理想回合结束时，Agent 应尽量同时满足：

1. 当前 Active 能够按照既定目标攻击，或者已经完成胜利条件；
2. 下一只攻击者有明确的底座、进化时机和能量路径；
3. 本回合的一次性资源没有被无意义地浪费，也没有因为提前攻击而放弃；
4. 手牌数量足以产生需要的 `Powerful Hand` 伤害，但没有为了伤害盲目耗尽牌库；
5. Prize、Active、Bench、手牌、弃牌区和牌库的资源账本已经更新；
6. 没有把本回合刚放下的 Pokémon 错误地视为可以进化的 Pokémon；
7. 没有让一个本回合无法攻击的宝可梦占据 Active，除非换位本身无法合法完成攻击。

### 1.2 Agent 每回合要维护的计划

Agent 不应直接从所有合法 option 中挑一个“分数最高”的动作，而应先生成一个小型
`TurnPlan`。计划至少包含：

| 计划字段 | 含义 |
|---|---|
| `victory_route` | 本回合是否能拿完 Prize、击倒关键目标或避免立即输局 |
| `primary_attacker` | 本回合真正准备攻击的 Active Pokémon |
| `attack_target` | 对手 Active 或 Boss 拉出的目标，以及预计伤害/Prize |
| `handoff_attacker` | 当前攻击者被击倒后或攻击结束后的下一条攻击线 |
| `must_actions` | 不完成就会损失现实价值的动作，例如必要进化、Fezandipiti、Hammer 或换位 |
| `supporter_choice` | 本回合唯一 Supporter 的目的和机会成本 |
| `energy_choice` | 本回合手填 Energy 的目标与理由 |
| `deck_budget` | 本回合允许继续从牌库抽取或检索多少资源 |
| `attack_commit_condition` | 满足哪些条件后才允许宣告攻击 |

动作执行后，Agent 只需要更新事实并重新规划剩余部分，不需要机械地执行旧计划中
已经失效的动作序列。

### 1.3 三种计划状态

每回合的整体目标只应属于以下三类之一：

1. **终局计划**：本回合可以拿完最后 Prize、击倒最后可战斗 Pokémon，或完成确定的
   胜利闭环。终局计划可以越过牌库保护线，但仍须在攻击前完成所有必要动作。
2. **攻击计划**：本回合完成一条可验证的攻击路线，同时保留下一只打手和下一回合的
   资源。攻击不一定马上拿 Prize，但必须改善真实的胜利路线。
3. **建设/生存计划**：本回合攻击不能带来足够价值，重点是建立进化链、恢复资源、
   保护牌库、处理对手攻击压力，或避免下一回合无法继续攻击。

Agent 不应在“建设计划”中为了结束回合而强行使用低阶段攻击，也不应在“攻击计划”
中因为当前存在一个合法攻击 option 就跳过能让下一只打手继续成长的动作。

---

## 2. 不可违反的规则闸门

这些条件不是动作分数，而是策略的硬边界。任何重构都必须先通过这些闸门，再比较
动作价值。

### 2.1 攻击是回合终止提交

一旦选择攻击：

- 本回合不能再进化；
- 不能再手动附 Energy；
- 不能再使用 Supporter、Item 或普通 Ability；
- 不能再铺 Bench；
- 不能把攻击理解成普通的最后一个排序动作。

因此，攻击前必须先完成所有本回合仍有价值的主行动。即使当前 Alakazam 能够 KO，
也要先确认本回合是否还能合法完成：

- Fezandipiti ex 的抽牌；
- Kadabra 或 Alakazam 的进化 Ability；
- Dudunsparce 的抽牌与换位；
- 下一只 Abra 线的铺场或进化；
- Enhanced Hammer 的拆能；
- 唯一 Supporter 的恢复、搜索、Boss 或干扰；
- 本回合手填 Energy 的最佳分配。

“能 KO”只是攻击可行性，不是攻击优先级。

### 2.2 进化必须使用进化时钟

Agent 至少要区分：

- Pokémon 本回合开始时是否已经在场；
- Pokémon 是否本回合才放下；
- Pokémon 是否本回合已经进化；
- 是否可以自然进化；
- 是否可以使用 Rare Candy。

本回合刚放下的 Basic 不能进化；同一只 Pokémon 不能在同一回合连续进化；Rare
Candy 只能跳过 Stage 1，不能绕过这些时间规则。

带 Psychic Energy 且可以走 `Rare Candy -> Alakazam` 的 Abra，不能为了“顺手进化”
被自然进化成 Kadabra。没有能量、且不承担本回合主攻击路线的其它 Abra，才是自然
进化成 Kadabra 并使用 `Psychic Draw` 的优先对象。

### 2.3 一回合一次的资源必须先规划

下列资源各自有独立机会成本：

- Supporter：每回合最多一张；
- 从手牌手动附 Energy：每回合最多一次；
- Retreat：每回合最多一次，并要支付 Retreat Cost；
- Stadium：每回合最多一次；
- 卡牌文本中自己的 Ability 次数限制。

Supporter 不能因为“现在也能用”就立即消耗；手填 Energy 不能因为“有合法目标”
就随意贴上；Retreat 只有在换位后能够让已经准备好的攻击者在本回合攻击，或能
明确完成胜利闭环时，才有直接价值。

### 2.4 牌库风险不是通用手牌上限

牌库使用分为三个策略区间：

- `> 15`：可以较自由地检索和过牌；
- `11-15`：开始比较每一次 deck -> hand 的收益；
- `<= 10`：默认保护牌库，只有能完成本回合最后 Prize 的闭环才继续消耗。

这些是策略保护线，不是规则中的手牌上限。Alakazam 需要手牌产生伤害，但不能为了
多几个伤害指示物而损失下一只打手或让自己的牌库在未来回合开始时无法抽牌。

### 2.5 两个永久边界

1. `Dunsparce` 的 `Trading Places` 永远不作为策略动作提交。它是攻击，不是普通
   换位；选择它后本回合立即结束，不能再让 Bench Alakazam 攻击。
2. 不预设 opponent ID 或完整构筑。只有 observation 真实看到对手的特殊能量、Tera
   Pokémon、Stadium、手牌、伤害和攻击压力时，才启用对应的干扰或目标策略。

---

## 3. 一个回合的完整节奏

规则上的节奏是“抽牌 -> 主行动任意顺序 -> 攻击并结束”。Agent 的内部规划也应保持
这个节奏，但主行动内部不应被写成一条固定的卡牌清单，而应由当前计划决定。

### 阶段 A：回合开始，读取事实

回合开始后先不提交动作，完成一次状态同步：

1. 记录己方和对手的 Active、Bench、手牌数量、牌库数量、Prize 数量、能量、伤害
   和弃牌区；
2. 识别上一回合是否发生己方 KO、对手 KO、对手换位或攻击效果；
3. 更新每只己方 Pokémon 的进场回合、进化回合和本回合可用状态；
4. 更新 Supporter、手填 Energy、Retreat、Stadium、Ability 和 Item Lock 状态；
5. 按牌库保护线确定本回合的 deck budget；
6. 清除已经结束的临时效果，不把旧日志或旧的 effect serial 当成当前事实。

这一步只产生事实，不做动作价值判断。事实账本和动作计划必须分离。

### 阶段 B：先判断胜负和生存

在安排普通动作前，先回答三个问题：

1. 本回合是否存在确定的胜利路线？
2. 如果不攻击，对手下一次行动是否会让己方失去可战斗 Pokémon、Prize 竞争或关键
   资源？
3. 如果攻击，是否会因为没有接班人而立即失去后续攻击节奏？

如果本回合可以拿完最后 Prize，终局计划优先，但仍要先完成所有不会破坏终局且能增加
成功确定性的动作。比如先使用必要的 Boss、完成必要换位或补足攻击能量，然后才攻击。

如果当前 Active 不能 KO，而对手 Bench 有确定可以击倒的目标，则把 Boss 纳入本回合
唯一 Supporter 的候选；多个目标都能 KO 时，优先剩余 HP 更高者，除非 Prize 或对手
下一回合威胁改变了这个结论。

### 阶段 C：分配当前攻击者和接班攻击者

Agent 必须先决定“谁负责本回合攻击”，再决定“谁负责下一回合攻击”。推荐的主攻击
者判断顺序如下：

1. Active Alakazam 已有 Psychic Energy，且可以形成有效的 `Powerful Hand`；
2. Active Kadabra 已有 Psychic Energy，手里有 Alakazam，且自然进化后能攻击；
3. Active Abra 已有 Psychic Energy，并且已在场、手里有 Alakazam 和合法 Rare Candy
   路线；
4. Active 是 Dunsparce，而 Bench Alakazam 已有 Psychic 或可以在本回合获得 Psychic，
   此时优先寻找 `Dudunsparce -> Run Away Draw -> Alakazam` 的换位路线；
5. 其它 Active 只有在本回合无法建立以上路线时，才考虑 Retreat 或低阶段攻击。

接班攻击者不是“Bench 上有一只 Basic”这么简单。它至少需要有一条现实路径：

```text
Basic Abra 已经在场
    -> 合法自然进化或 Rare Candy 时机
    -> Alakazam / Kadabra 可以获得 Psychic
    -> 下一次攻击前仍有搜索或恢复入口
```

如果当前攻击会击倒 Active，但 Bench 没有任何可延续的 Abra 线，Agent 应把建立接班
线视为攻击前的高优先级任务。除非这是最后 Prize 的终局闭环，否则不能因为当前能打
出伤害就放弃接班资源。

### 阶段 D：执行攻击前的高价值动作

主行动的执行顺序由计划决定，但以下动作类别应按“是否改变当前或下一次攻击路线”
进行处理。

#### D.1 触发型 Ability

- 如果己方上一回合有 Pokémon 被击倒，且 Fezandipiti ex 的 `Flip the Script` 合法，
  应先完成它的抽 3。它是 Ability，不占 Supporter 机会；不能因为有其它普通 Bench
  就默认跳过。
- Kadabra 和 Alakazam 的 Psychic Draw 应在不破坏主攻击 Abra 路线的前提下使用。带
  Psychic 的 Abra 如果准备走 Rare Candy，不应先自然进化；其它无能量 Abra 可以先
  进化为 Kadabra 过牌。
- Active Dunsparce 需要让 Bench Alakazam 接班时，应把 Dudunsparce 的 Ability 当成
  换位资源，而不是单纯的抽 3。合法路线是：进化 Dudunsparce、使用 `Run Away Draw`、
  将 Dudunsparce 和附着卡洗回牌库、让可攻击的 Bench Pokémon 来到 Active。
- Active Dudunsparce 且 Bench 为空时，不使用会把唯一 Active 洗回牌库的 Ability。

#### D.2 立即有效的干扰

- 只要对手有合法 Special Energy 目标且 Item 未被锁定，就使用 Enhanced Hammer。四
  张 Hammer 的构筑目的就是降低对手能量密度，收益可以体现在后续回合，而不要求本回合
  立即 KO。
- Hammer 目标优先级为：Active 上的保护性/关键特殊能量，Bench 上的保护性/关键特殊
  能量，Active 上的其它特殊能量，Bench 上的其它特殊能量。Rock Fighting Energy、
  Mist Energy 等需要优先处理；同类目标中 Active 优先于 Bench。
- Nighttime Mine 只在合法且具有真实场面价值时使用：覆盖对手 Stadium，或 observation
  中存在 Tera Pokémon、增加一张无色费用能改变对手攻击节奏。它不是通用增伤卡，也不
  应为了“打出一张 Stadium”而牺牲更重要的攻击路线。
- Item Lock 只覆盖实际观察到的紧接着的己方回合。锁定时不能使用 Hammer、Poffin、
  Poké Pad、Rare Candy、Night Stretcher 或 Sacred Ash，但仍可自然进化、使用 Ability、
  使用 Supporter 和手填 Energy。

#### D.3 建立 Bench 和进化链

铺场的目标是“下一只打手”和“完成当前换位”，不是无条件填满 Bench：

- Poffin 优先建立 Abra 线和至少一条 Dunsparce 线；
- Telepath Psychic Energy 在附能对象为 Psychic Pokémon 时，同时承担铺 Bench 的
  价值；但低牌库时要计算搜索成本；
- 已有现实的两只 Abra 和一只 Dunsparce、且当前没有立即进化或换位收益时，第一回合
  不因习惯使用 Poké Pad；保留它寻找 Kadabra、Alakazam 或 Dudunsparce；
- 不把 Shaymin 当作 Abra 线，也不把 Fezandipiti ex 当作可以由 Poffin/Poké Pad 找到
  的普通 Pokémon；
- 进化时优先保护带能量的主攻击路线，再让无能量的备用 Abra 进入 Kadabra 阶段并使用
  Psychic Draw。

#### D.4 使用唯一 Supporter

Supporter 的选择应服务于当前 `TurnPlan`，而不是按卡牌固定排序：

| 场景 | 首选方向 | 约束 |
|---|---|---|
| 当前 Active 无法 KO，Bench 有确定 KO 目标 | Boss’s Orders | 只有目标和 Prize 价值确定时消耗唯一 Supporter |
| 下一只攻击者断档，需要从弃牌区恢复 | Lana’s Aid | Pokémon-first；最多选择三张合法无 Rule Box Pokémon，只有 Basic Psychic 是必要缺口时才加入 Energy |
| 需要同时补 Basic、Stage 1、Stage 2 | Dawn | 三个类别都必须有现实用途，优先补 Dunsparce/Dudunsparce 换位或 Abra 进化链 |
| 需要 Evolution Pokémon 和 Energy 形成攻击路线 | Hilda | 不拿孤立的 Stage 2；先确定底座和进化时机 |
| Active 可攻击、对手 Active 已受伤但不能 KO、对手手牌至少 6 张且无 Boss KO | Xerosic’s Machinations | 先确认压到 3 张比其它 Supporter 和直接攻击更有价值 |

如果没有 Supporter 能改善当前或下一回合的现实计划，可以保留 Supporter。Supporter
机会本身不是必须消耗的计数器。

#### D.5 使用手填 Energy 和换位

手填 Energy 的判断顺序：

1. 给本回合主攻击者补足攻击所需的 Psychic；
2. 如果主攻击者已经准备好，给下一只确定能进化和攻击的 Abra 线；
3. 如果主攻击者已经准备好、Dudunsparce 路线能产生现实过牌或换位价值，再把
   Enriching Energy 给 Dunsparce/Dudunsparce；
4. Telepath Psychic Energy 的附能同时考虑它提供 Psychic、铺 Bench 和消耗牌库的
   三重影响；
5. Enriching Energy 不能当作 Psychic，不能为了抽 4 而破坏当前攻击所需的 Psychic。

Retreat 只在换位后已有攻击条件的 Pokémon 能在本回合攻击时执行。不能先 Retreat
   再期待用攻击后的动作补能，也不能用 Retreat 替代 Dudunsparce 的合法换位路线。

### 阶段 E：攻击前最终检查

宣告攻击前，Agent 必须重新计算一次，不使用动作执行前的旧计划：

- 当前 Active 的攻击是否合法，能量是否真实存在；
- 本回合是否已经使用 Supporter、手填 Energy、Retreat 或 Stadium；
- 是否还有合法且有现实价值的 Ability、进化、检索、恢复、Hammer 或换位动作；
- 目标是否仍然能 KO，预计拿几张 Prize，是否需要 Boss；
- 手牌是否已经达到所需伤害，继续过牌是否会损失牌库或接班资源；
- 当前攻击结束后，Bench 是否有下一只可持续进化的攻击线；
- 这次攻击是否是最后 Prize、最后可战斗 Pokémon 或必须避免下一回合输局的终局提交；
- 是否误把 `Trading Places`、低阶段非 KO 攻击或无法接班的攻击当成了正常路线。

只有“没有更高价值的合法准备动作”且攻击能改善胜利路线时，才允许提交攻击。

### 阶段 F：选择攻击并结束回合

攻击选择的优先级是：

1. 完成最后 Prize 或确定胜利条件的攻击；
2. 在 Boss 或直接攻击之间选择确定 KO 且 Prize 价值更高的路线；
3. 用 Alakazam 的 Powerful Hand 完成当前目标所需的伤害；
4. 其它能维持攻击节奏的有效攻击；
5. Kadabra 的非 KO 攻击只在没有更高价值的主行动、无法建立接班线且不攻击更差时
   才考虑；
6. Abra 不主动攻击；Dunsparce 的 Trading Places 永不提交。

选择攻击后，Agent 立即将本回合状态标记为已提交，不能再期待后续主行动。

### 阶段 G：攻击结算和下一回合准备

攻击结算后只更新事实，不在同一个回合追加动作：

1. 更新伤害、击倒、Prize、对手新 Active 和双方区域；
2. 记录己方是否失去攻击者，以及当前是否存在 ready attacker；
3. 为下一回合保存 Active/Bench 快照和进化时钟；
4. 更新资源账本中的 Pokémon、Energy、Supporter 和牌库变化；
5. 下一回合开始时重新生成计划，不沿用已经过期的 option index 或效果序列。

---

## 4. 卡组关键资源的目标角色

这不是逐卡规则说明；卡牌文本和边界以 `DECK_NOTES.md` 为准。这里仅定义它们在
最终策略中的职责。

| 资源 | 目标角色 |
|---|---|
| Abra | 四条攻击线的底座；带 Psychic 时保护其 Rare Candy 直通路线 |
| Kadabra | 无能量备用 Abra 的自然进化过牌节点；不是为了进化数量而消耗的终点 |
| Alakazam | 当前主攻击者；进化后的 Psychic Draw 和 Powerful Hand 都要纳入攻击前计划 |
| Dunsparce | Bench 过牌底座和低 Prize 资源；不提交 Trading Places |
| Dudunsparce | 抽牌与换位资源；Active Dunsparce 需要把它视为让 Bench Alakazam 接班的必要步骤 |
| Fezandipiti ex | 己方上一回合被击倒后的重要补手牌 Ability；Ability 不占 Supporter |
| Basic Psychic Energy | 稀缺攻击能量；优先确保当前和下一只 Alakazam 的 Psychic |
| Telepath Psychic Energy | 攻击能量、铺 Bench 和牌库搜索的复合资源 |
| Enriching Energy | 在主攻击者已准备好时，转化为 Dudunsparce 的抽牌/换位价值；不能支付 Psychic 攻击 |
| Rare Candy | 保护带 Psychic 的 Abra 直通 Alakazam；必须服从进化时钟 |
| Enhanced Hammer | 四张构筑的长期能量压制；有合法 Special Energy 目标就使用 |
| Buddy-Buddy Poffin | 建立 Abra 和 Dunsparce 底座；按接班计划使用，不为填 Bench 而使用 |
| Poké Pad | 进化链检索；基础 Bench 已经够用时延迟，保留关键进化或 Dudunsparce |
| Dawn | 一次补齐 Basic、Stage 1、Stage 2 的完整路线；每一类选择都要有用途 |
| Hilda | 把 Evolution Pokémon 和 Energy 组合成具体攻击路线 |
| Lana’s Aid | Pokémon-first 的弃牌区接班入口；Energy 只是完成攻击的必要补充 |
| Sacred Ash | 尽可能装满至多五只 Pokémon，优先恢复两条 Abra 进化链，再补 Dunsparce/其它资源 |
| Night Stretcher | 单张窄恢复工具；只在能闭合一条现实攻击路线时使用 |
| Boss’s Orders | 把确定 KO 目标拉到 Active；不是普通换位工具 |
| Xerosic’s Machinations | 在可攻击但不能 KO 的受伤局面压低对手手牌；不抢走确定 Boss/恢复机会 |
| Nighttime Mine | 覆盖 Stadium 或提高实际 Tera 对手攻击门槛；不是通用增伤卡 |

---

## 5. 重构后的职责边界

Agent 代码应围绕四个独立层次组织，而不是把所有卡牌判断塞进一个巨大排序函数。

### 5.1 Facts：事实层

只负责把 observation 变成稳定的事实：

- 区域和卡牌位置；
- Pokémon 实例、序列号、进场/进化时点；
- 能量、伤害、Prize、牌库和手牌数量；
- 本回合一次性资源和实际观察到的对手效果；
- 动作历史、KO 事件和 effect serial。

事实层不回答“这张牌值不值得用”。

### 5.2 Plan：计划层

只负责决定：

- 当前是终局、攻击还是建设/生存计划；
- 谁是当前攻击者；
- 谁是下一只接班攻击者；
- 本回合必须完成哪些动作；
- Supporter、手填 Energy 和牌库预算分别服务什么目的；
- 什么条件满足后允许攻击。

计划层不依赖 option 数组下标，也不直接操作引擎 API。

### 5.3 Actions：动作层

把引擎提供的合法 option 映射为语义动作，例如：

- `play_item(card_id=1081)`；
- `evolve(source=305, target=66)`；
- `attach_energy(card_id=19, target=741)`；
- `use_supporter(card_id=1184, purpose="handoff")`；
- `use_ability(card_id=66, purpose="promote_attacker")`；
- `attack(attack_id=1072, target=...)`；
- `end_turn`。

动作层只负责合法 option 的匹配、提交和记录；不在此处重新计算整套策略。

### 5.4 Effects：选择层

对同一张卡的多目标、多张选择和数量选择单独处理：

- Hammer 的目标能量排序；
- Lana’s Aid 的 Pokémon-first 选择；
- Sacred Ash 的最多五张恢复；
- Dawn 的 Basic/Stage 1/Stage 2 三类选择；
- Poffin、Telepath、Poké Pad 的搜索对象；
- Boss 的 KO 目标；
- Xerosic 的使用条件。

效果选择器接收 `TurnPlan` 和事实状态，不能使用一个与卡牌语义无关的通用排序。

### 5.5 每次动作都重新规划，但不重新定义目标

推荐的主循环是：

```text
读取 observation
  -> 更新 Facts
  -> 生成 TurnPlan
  -> 从合法 option 中选择一个符合计划的语义动作
  -> 提交并记录动作
  -> 读取新的 observation
  -> 更新 Facts，重新评估剩余 must_actions
  -> 直到 attack 或 end_turn
```

重新规划是为了应对新的手牌、牌库、效果和对手状态；不是允许每一步都回到一个
没有上下文的全局分数比较器。

---

## 6. 自动迭代的验证接口

重构后的 Agent 应让每一个策略目标都能被单独测量，而不是只能看最终胜负。

### 6.1 正确性指标

- 非法 action 数量：必须为 0；
- `Trading Places` 选择次数：必须为 0；
- 空 Bench 的无意义 `Run Away Draw`：必须为 0；
- agent error、错误目标数量选择和错误 Supporter/手填 Energy 次数：必须为 0；
- Item Lock 下使用 Item 的次数：必须为 0。

### 6.2 计划执行指标

- 第二个己方回合实际选择 `attackId=1072` 的次数；同时报告全样本分母和到达目标回合
  的分母；
- “合法 Powerful Hand 存在但没有选择”的次数；
- 攻击前仍有未执行高价值动作的次数；
- Fezandipiti 在己方上一回合被击倒后的 Ability 使用率；
- 有合法 Special Energy 目标时的 Hammer 使用率和目标优先级正确率；
- Active Dunsparce 通过 Dudunsparce 完成换位并让 ready Alakazam 接班的次数；
- Alakazam 被击倒后下一只 ready attacker 的形成率；
- Lana’s Aid 和 Sacred Ash 的恢复对象数量、Abra 线覆盖率；
- 低牌库状态下错误抽牌、错误检索和错误延迟终局攻击次数。

### 6.3 结果指标

结果指标只在 correctness gate 通过后使用：

- raw win rate 和 meta-weighted win rate；
- 各 opponent 的胜率和失败类型；
- post-KO break rate；
- first Alakazam turn；
- 终局攻击错误、牌库保护错误和对手侧 evaluator error。

单个本地对局不能晋级策略。每次候选修改应使用相同对照批次、明确分母和可复核
trace，先判断行为指标，再判断结果是否出现稳定改善。

---

## 7. 明确禁止的重构方向

以下实现方式与目标策略冲突：

1. 用一个全局 score 对所有合法 option 排序，并让攻击自然成为“分数最高”的普通动作；
2. 写死“看到某个 opponent ID 就执行一套完整脚本”；
3. 用 option 数组下标代表卡牌身份、目标或效果阶段；
4. 把 Supporter、Ability 和 Item 混成同一个资源计数器；
5. 把 `Enriching Energy` 当作 Psychic，或把 `Fezandipiti Ability` 当作 Supporter；
6. 把有合法目标等同于值得使用，不检查本回合和下一回合攻击路线；
7. 给牌库设一个未经规则或策略证明的通用手牌上限；
8. 为了追求当前 `Powerful Hand` 伤害，牺牲下一只接班打手；
9. 把 `Trading Places` 当成换位 Ability，或在攻击后期待继续行动；
10. 在一个函数里同时维护区域账本、进化时钟、计划、效果选项和动作提交。

---

## 8. 最终决策口诀

每次 Agent 即将做动作时，按以下顺序回答：

```text
我现在是否必须赢，或必须避免立即输？
谁负责当前攻击，谁负责下一次攻击？
本回合还有没有能改变这两条路线的合法动作？
Supporter、手填 Energy、Retreat 和牌库预算分别要留给什么？
如果现在攻击，攻击之后的接班线是否仍然成立？
只有答案都清楚后，才宣告攻击并结束回合。
```

这套策略的核心不是“尽可能多做动作”，而是**在不可逆的攻击提交之前，把所有
仍然能创造现实胜利价值的资源转化为当前攻击和下一只打手的确定性**。

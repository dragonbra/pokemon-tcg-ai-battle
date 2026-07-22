# Alakazam V6 完整策略重构设计

> 状态：已形成 V6 实验实现；本文仍保留设计推导、假设和待用 replay 校准的参数。
> 基线：`submission/alakazam_v5_auto_iter/`。
> 边界：固定 `deck.csv`，只重构策略与策略说明。
> 原则：先把“我们每一回合到底想最大化什么”讨论清楚，再决定如何修改代码。

## 0. 这份文档要解决什么

V6 暂时不应被理解为“在 V5 auto_iter 上继续加几个 if”。我们需要重新描述一套
完整策略：从开局铺场、第一只 Alakazam、攻击手接力、Supporter/能量分配，到
牌库预算和 Prize race 的统一决策顺序。

这份文档作为我们接下来逐项讨论的工作台。下面分成三类内容：

- **已确认事实**：来自现有代码、官方 replay 和迭代报告，可以直接作为设计输入。
- **暂定判断**：目前有证据支持，但还需要我们讨论优先级或用新 replay 验证。
- **待讨论决策**：没有得到双方确认前，不应该写进 v6 代码。

## 0.1 用户确认的 V6 总目标（原始要求）

> 总目标：我认为目前动作执行的优先级，有一点不符合我们常规玩宝可梦卡牌时计算的一些逻辑。
>
> 这里面的原因，一方面是你可能对宝可梦的规则不够了解，另一方面可能有些点我们之前没有充分讨论和表述清楚。
>
> 因此，我希望 V6 是一个基于真实环境的完全重构。具体要求如下：
>
> 1. 保持在宝可梦卡牌的语义下：我们构建策略，仍然是要在宝可梦卡牌语义下的一个子集来进行。
> 2. 遵循宝可梦卡牌的专属规则：在针对卡组做最优操作等策略时，绝对不能忘记宝可梦卡牌的专属规则。比如我之前叮嘱过的一些要点：
>    - 要提前准备进化。
>    - 要控制手牌，不能把牌库抽空。
>
> 这些规则不是零散的经验法则，而应该成为 V6 决策器的硬约束和状态模型的一部分。

## 0.2 “完全重构”的具体含义

V6 不会把现有 `_main_action.score()` 继续当成唯一的策略骨架，再向其中追加更多
分数。我们需要重新建立四层边界：

1. **规则语义层**：把进化时机、攻击成本、能量保留、抽牌/牌库风险、Prize、Active
   与 Bench 的关系表达成可验证的事实。
2. **局面状态层**：从 observation 提取“现在能做什么、下一回合能做什么、哪些资源
   正在被占用”，而不是只统计 ready attacker 数量。
3. **阶段策略层**：识别当前处于铺场、首攻、连续攻击、恢复或收尾阶段，并使用不同
   的动作目标。
4. **动作价值层**：只在规则合法且语义有意义的候选动作中排序；score 只是实现手段，
   不能替代策略定义。

这里的“规则语义子集”不是要在 agent 内重写完整宝可梦引擎，而是要把本卡组真正会
遇到的规则和卡牌互动显式化，避免把“模拟器给了合法 option”误当成“这就是好动作”。

## 0.3 官方规则学习后的硬约束

官方规则书与本地引擎审计已经把 V6 的规则边界明确下来，完整证据见：
`docs/reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`。

- **攻击是终止提交**：宣告攻击后本回合结束，不能再进化、附能、使用 Supporter、
  铺 Bench 或使用普通 Ability；所有准备必须在攻击前完成。
- **进化有时钟**：只能进化本回合开始时已经在场的 Pokémon；本回合刚放下的不能
  进化，进化后的 Pokémon 不能同回合再次进化；Rare Candy 只能跳 Stage 1，不能绕过
  其自身的首回合/本回合放下限制。
- **回合预算是硬约束**：Supporter、手动从手牌附 Energy、Retreat、Stadium 都有
  一回合限制；Item 和 Ability 仍要服从各自卡牌文本。
- **胜负和牌库要准确建模**：胜利来自拿完 Prize、对手无 Pokémon 可战斗，或对手在
  回合开始无法抽牌；规则没有通用手牌上限，效果抽牌超过牌库只抽现有牌并继续。
- **职责分离**：引擎负责 option 合法性，V6 负责比较攻击终止机会、唯一 Supporter
  机会、手动附能目标、进化路线和未来 Prize 价值。

这些内容是 V6 的不可被 score 抵消的全局不变量。后续策略设计必须先证明动作没有
违反它们，再讨论动作之间的收益排序。

## 0.4 规则学习后对过去策略的反思

这次规则学习暴露出的主要问题，不是某几个优先级数字不够准确，而是我们曾经把
“模拟器允许执行”误当成“现在应该执行”。当前引擎会过滤非法 option，但不会替我们
计算一个动作消耗了什么回合资源、关闭了什么后续路线，以及它是否值得终止本回合。

下面把之前 review 中反复出现的具体问题，重新翻译成规则和策略问题。这里的“过去”
包含 V5 及其迭代中曾出现过的行为；即使部分问题已经被局部修复，也要在 V6 中把
它们上升成状态模型，而不是继续依赖某个特例 gate。

### 0.4.1 过去的误判与规则根因

| 过去暴露出的行为 | 真正的规则/策略根因 | V6 的设计结论 |
| --- | --- | --- |
| 已经能用低阶段宝可梦攻击，便过早宣告攻击，之后才发现本回合本来还应该进化、附能、使用 Supporter 或铺下一只 Abra。 | Attack 不是普通的主行动；一旦选择攻击，本回合立即结束。 | 把攻击建模为“回合终止提交”。先规划本回合所有不可逆的准备，再比较攻击与继续准备；任何攻击评分都不能绕过剩余的回合预算。 |
| 第一回合用 Poké Pad 找 Kadabra，随后把它当成当前回合的进化资源；或者把本回合刚放下的 Abra 当成立刻可进化。 | 进化目标必须在本回合开始时已经在场；刚放下的 Basic 不能进化。进化后同回合也不能再连续进化。 | 分开表示“现在能进化”和“为下一回合准备”。进化时钟至少需要 `was_in_play_at_turn_start`、`appearThisTurn` 和 `evolved_this_turn`；不能只看手牌是否有 Kadabra/Alakazam。 |
| 把 Rare Candy + Alakazam 当成所有局面的固定最高路线，宁可让后排 Abra 直进，也不让前场已有能量的 Kadabra 自然进化。 | Rare Candy 只是允许 Basic 跳过 Stage 1 的一种卡牌效果，不是 Alakazam 的唯一正确路线；它的价值取决于进化目标和攻击时机。 | 第二回合 Active Abra 的直通是一个特殊的抢节奏路线；但到了 Active Kadabra 已有能量、手里有 Alakazam 的局面，应优先让 Active 自然进化并攻击，Rare Candy 可以留给真正需要跳级的 Basic。优化对象是“哪个目标形成有效攻击”，不是“哪个组合看起来更高级”。 |
| 把能量贴到已经有 Psychic Energy 的 Kadabra/Alakazam，甚至在一只打手上堆多张能量，同时让下一只攻击线缺能量。 | 每回合只能手动附一张手牌 Energy；进化会保留附着能量，所以能量应该提前贴在线上，而不是等到 Alakazam 阶段重新贴。 | 将能量视为“攻击线资源”而非当前 Pokémon 的分数。默认一条 Abra 线只需要一张能满足攻击的 Psychic Energy；手填优先补没有该能量的现实攻击线。Telepath 的搜索效果只在从手牌附着时触发，进化不会重新触发。 |
| 第一回合或没有明确收益时发生 Retreat，只是为了改变 Active/Bench 位置；或为了撤退支付了本来属于下一只打手的能量。 | Retreat 一回合只能使用一次，并按 Retreat Cost 丢弃能量；它不是免费的换位动作。 | 只有当 Retreat 能立即交给一只可进行有效攻击的打手，或能避免明确的失败，才把它列入行动计划；同时消耗 `retreat_available` 并计算被丢弃的能量。Dunsparce 的 Trading Places 是攻击，不能当作免费 Retreat；V6 初版直接禁用它，因为使用后本回合结束，也不能再让被换到 Active 的 Alakazam 攻击。 |
| 已经使用了 Dawn/Hilda 后，仍然把 Boss/Xerosic 当作本回合可继续使用的备选；或者只看 Supporter 的即时抽牌数，没有比较它与进化、Boss 击倒、干扰的机会成本。 | Supporter 一回合只能使用一次。Item、Ability 等并没有统一的“一回合一次”，必须按具体卡文处理。 | 在回合状态中显式维护唯一的 Supporter 预算。先比较“完成当前/下一回合攻击线、制造 KO、恢复、干扰”的价值，再消耗 Supporter；引擎的合法性过滤不能代替这个比较。 |
| 手里只剩 Xerosic、Enhanced Hammer、Fezandipiti 等低动点牌时，直接用 Kadabra 打 30 点，甚至没有先尝试 Fezandipiti 的过牌。 | 攻击会终止回合；低阶段攻击一旦提交，就放弃了本回合剩余的 Ability、Item、Supporter 和进化机会。Fezandipiti 的 Ability 不是 Supporter，但拍下它会占用 Bench 并承担两 Prize 风险。 | 低阶段攻击必须是有理由的 fallback：通常要么能 KO，要么能保护一条明确的接力路线。若没有有效攻击路线，应先评估 Fezandipiti、Dudunsparce 等过牌是否能补齐下一步；这个判断不能被“上一回合是否刚被 KO”这一单一 gate 限死。 |
| 牌库还有资源却因为固定的手牌/牌库阈值停止过牌；或者场上已经有两只 ready attacker，仍继续过牌直到牌库耗尽。 | 规则没有通用手牌上限；效果抽牌超过剩余牌库时只抽能抽到的牌。真正的牌库失败发生在己方回合开始需要抽牌、却没有牌可抽。手牌在本卡组中同时是 Alakazam 伤害和未来资源。 | 把手牌数量、剩余牌库、下一次回合开始抽牌、Prize 进度和缺失卡件放进同一个预算模型。固定 `hand > 20` 或 `deck <= 10` 只能是经验性的保护线，不能伪装成规则；当过牌能完成 KO 或明确找到下一只打手时，可以在预算内承担风险。 |
| 用 Night Stretcher/Lana's Aid 从弃牌区拿回 Alakazam，却没有场上的 Abra；或者只按高阶段卡牌的 Prize 价值恢复。 | Rare Candy 的目标是场上的 Basic，而且该 Basic 还必须满足进化时钟；没有合法基础目标时，拿回 Alakazam 不能形成进化路线。 | 恢复要从“下一次有效攻击”反向规划：没有可进化的 Basic 时先恢复并铺 Abra，再恢复 Kadabra/Alakazam；有合法自然进化目标时才比较高阶段卡牌。恢复动作必须解释它补的是哪条现实路线。 |
| 用 `field_count` 或 `ready_attacker_count` 判断攻击线已经安全，导致过早把 Poffin/Hilda 的资源转给 Dunsparce；或者为了保护资源把 gate 收得过严，牺牲首只 Alakazam 和攻击节奏。 | “在场”“有一张能量”“本回合能攻击”“下回合能交接”是四种不同状态。ready 数量没有表达进化时钟、换位方式、能量损失和对手击倒后的 Active 风险。 | 用可解释的 `handoff_path` 替代单一 ready 数量：下一只必须有合法进化时机、Psychic Energy、进入 Active 的方式，以及交接后仍可继续攻击或恢复的路径。资源保护必须服务于 handoff，而不是以数量 gate 为目标本身。 |

### 0.4.2 从这些错误提炼出的全局状态

V6 至少要把下面四组状态分开，不能继续让一个 `_main_action.score()` 隐式猜测：

```text
回合预算
├─ supporter_available       # 本回合是否还可以使用 Supporter
├─ manual_energy_available   # 本回合是否还可以手填 Energy
├─ retreat_available         # 本回合是否还可以 Retreat
├─ stadium_available         # 本回合是否还可以使用 Stadium
└─ attack_open               # 仍可准备；宣告攻击后变为 false 并结束回合

进化时钟
├─ was_in_play_at_turn_start
├─ appeared_this_turn
└─ evolved_this_turn

攻击计划
├─ current_effective_attack  # 当前是否能造成有意义的伤害/KO
├─ next_attacker              # 下一回合的现实攻击者，而不是抽象 ready 数量
└─ handoff_path               # Retreat/被动换位是否可行

资源预算
├─ hand_value                 # Alakazam 伤害与关键卡资源
├─ deck_reserve                # 到下一次回合开始抽牌前必须保留的牌库
├─ energy_deadlines            # 哪条攻击线最晚何时必须完成附能
└─ prize_progress              # 当前动作是否缩短真实 Prize race

资源账本
├─ card_totals                # 例如 Abra/Kadabra/Alakazam 各自的 4 张总量
├─ prize_zone                 # 已知或推定落入 Prize 的数量
├─ hand_zone                  # 手牌中的数量
├─ field_zone                 # Active 与 Bench 中的数量
├─ discard_zone               # 弃牌区中的数量
└─ deck_zone                  # 牌库中的数量，和 deckCount 交叉核对

动作历史
├─ turn_start_snapshot        # 每个己方回合开始时的 Pokémon 实例和位置
├─ played_at_turn             # 每只 Pokémon 进入场面的己方回合
├─ evolved_at_turn            # 每次进化发生的己方回合
├─ supporter_used             # 本回合使用的 Supporter
├─ manual_energy_used         # 本回合是否已经手填 Energy
├─ retreat_used               # 本回合是否已经 Retreat
└─ attack_or_end              # 宣告攻击或主动结束，作为回合终止标记
```

其中，`supporter_available`、`manual_energy_available`、`retreat_available` 和
`attack_open` 是回合级硬约束；`current_effective_attack`、`handoff_path`、
`deck_reserve` 和 `resource_ledger` 是策略级判断。两者不能混成一个可被其他收益
抵消的分数。

### 0.4.3 对 V6 目标函数的修正

我们之前容易把目标写成“尽快做出一只可以攻击的 Alakazam”。规则学习后，更准确的
目标应该是：

> 在不违反回合预算和进化时钟的前提下，最大化未来 1–2 回合完成有效 Prize 推进的
> 概率；若本回合不能推进 Prize，则保护一条明确的进化、能量和交接路线，同时保留
> 足够的手牌/牌库生存预算。

这会带来几个具体的优先级变化：

1. **先完成回合计划，再决定是否攻击。** 攻击不再是“当前合法就与其他 option
   并列竞争”的普通候选。
2. **先问攻击目标，再问进化卡。** Rare Candy、自然进化和回收都只是实现目标的
   手段；目标可能是 Active Alakazam，也可能是为下一回合准备 Bench handoff。
3. **先分配一次手填能量，再考虑过牌。** 只有不存在更紧迫的攻击线缺口时，才把
   Enriching Energy 等转化为 Dudunsparce 的抽牌机会；但这不是“永远先做第二只
   打手”，而是比较能量最晚需求与过牌带来的下一回合命中概率。
4. **把低阶段攻击当作需要证明的选择。** “能打 30 点”本身不构成价值；必须说明
   它能 KO、保护接力、处理威胁，或比所有准备/过牌路线更能改善 Prize race。
5. **把牌库风险写成时序风险。** 不是机械地看到牌库较少就停止，而是计算抽牌后能否
   安全度过下一次回合开始抽牌，以及当前收益是否足以承担这个风险。

这一节目前是 V6 设计讨论的规则底稿，不等于已经批准的实现规格。实现前我们还需要
逐项确认“有效攻击”“handoff_path”“牌库 reserve”和 Supporter 机会成本的具体
判定方式。

### 0.4.4 第一次讨论结论：低阶段攻击是最后手段

用户确认：本卡组不需要把“当前这只宝可梦能否撑到下一回合”作为低阶段攻击的主要
判断依据。我们的攻击者本来就是可替换资源，不执着于让同一只 Pokémon 连续留场。

因此，若本回合仍能确保 Alakazam 的合法进化路线，并能继续铺设后续 Abra/攻击线，
即使 Active Kadabra 或 Abra 可以造成少量伤害，也不应立即攻击。低阶段攻击属于最后
手段，只有在本回合已经没有更有价值的进化、铺场、附能或过牌路线时才考虑。

这会把 V6 的判断重点从“保护当前 Active”转移到：

- 当前是否仍能推进一条合法的 Alakazam 进化路线；
- 是否已经为后续攻击线完成必要的铺场与附能；
- 继续准备是否比 30 点低阶段攻击更接近下一次有效 Prize 推进。

这里的 handoff path 仍然重要，但它表示“是否有下一只攻击者可以接班”，而不是
“是否必须保护当前攻击者不被击倒”。

### 0.4.5 第二次讨论结论：先完成 Active 自然进化，再发展后场 Kadabra

用户确认了以下场面判断：当 Active 是带 Psychic Energy 的 Kadabra，手里有
Alakazam 和 Rare Candy，Bench 有可以合法进化的 Abra 时，优先级是：

1. 先把 Active Kadabra 自然进化成 Alakazam；
2. 再把合法的 Bench Abra 进化成 Kadabra；
3. 使用 Kadabra 的 Ability 增加资源；
4. 在所有准备动作完成后，用 Active Alakazam 攻击。

这里不急着对 Bench Abra 使用 Rare Candy。Rare Candy + Alakazam 不是必须马上完成
的 combo，而是可以保留到中后期需要从回收的 Abra 重新建立 Alakazam 攻击路线时再用。

这说明 V6 的进化计划必须按“每只 Pokémon 的进化机会”分别安排，而不是在看到
Rare Candy + Alakazam 时就把它们视为一个必须立即消耗的组合：Active 的自然进化负责
本回合的 Prize 推进，Bench 的 Abra → Kadabra 负责本回合攻击前的资源增长，Rare
Candy 则保留为未来路线的弹性。

### 0.4.6 第三次讨论结论：有 Enriching Energy 时优先启动 Dudunsparce 过牌

用户确认：当 Active 已经具备本回合的有效攻击，手里有 Enriching Energy，且场上有
Dudunsparce 可以使用其抽牌 Ability 时，V6 第一版默认优先把 Enriching Energy 给
Dudunsparce，完成抽牌并将其及附着卡洗回牌库。

这里暂时不建立复杂的“下一只打手最晚何时必须完成附能”预测。原因是：能拿到
Enriching Energy 本身通常意味着当前手牌和检索资源已经比较充足，Dudunsparce 的
过牌大概率可以找到下一回合所需的攻击线。这个假设以后可以用 replay 验证，但不在
V6 初版中提前增加过多例外。

因此 V6 的简化能量规则是：

1. 当前 Active 尚未具备有效攻击时，先满足当前攻击线的 Psychic Energy 需求；
2. 当前 Active 已经具备有效攻击，且有 Enriching Energy + Dudunsparce 时，默认
   优先过牌；
3. 没有 Enriching Energy 时，其他可用 Energy 优先贴给没有 Psychic Energy 的
   Abra 系列 Pokémon；
4. 只有后续 replay 明确显示这条默认规则造成误判，才增加针对 matchup 或时序的
   特殊判断。

### 0.4.7 第四次讨论结论：Boss 优先制造确定 KO，并按 HP 选择目标

用户确认：如果对手 Active 不能被当前攻击击倒，而对手 Bench 存在可以被当前攻击
确定击倒的 Pokémon，应优先使用 Boss's Orders 把目标拉到 Active，再完成本回合攻击
并取得 Prize。

当 Bench 中有多个可以被击倒的目标时，优先选择其中剩余 HP 最高的那一只。这个规则
只在候选目标都已经满足“当前攻击可以确定 KO”时作为目标选择策略使用；它不允许
为了拉出更高 HP 的目标而放弃一个无法击倒的 Active，也不改变 Supporter 一回合只能
使用一次的规则。

因此，V6 的 Boss 判断顺序暂定为：

1. 当前 Active 能否被击倒；如果能，通常不为了换目标消耗 Boss；
2. 如果不能，枚举 Bench 中当前攻击可以确定击倒的目标；
3. 如果存在候选，选择剩余 HP 最高者，使用 Boss 后攻击；
4. 如果不存在候选，再比较 Dawn/Hilda 的进化铺场价值、Xerosic 的干扰价值和其他
   Supporter 机会成本。

### 0.4.8 第五次讨论结论：攻击已准备好时，Xerosic 取决于对手手牌规模

用户进一步确认：当 Active Alakazam 已经可以攻击、对手没有合适的 Boss 击倒目标时，
不需要为了提前发展后场而急着使用 Hilda 或 Dawn。只要这些牌留在手里，下一回合仍然
可以使用；本回合应优先保留攻击节奏。

如果对手 Active 已经受到伤害但还没有昏厥，且对手手牌至少有 6 张，则 Xerosic's
Machinations 的预期收益上升，可以在攻击前消耗本回合唯一的 Supporter。相反，对手
只有约 4 张手牌时，Xerosic 通常不能制造明确的限制，应避免为了轻微干扰提前消耗
Supporter。

因此，在 Active Alakazam 已经准备好攻击、且 Boss 无法制造确定 KO 时，V6 第一版的
Supporter 顺序暂定为：

1. 对手手牌至少 6 张，且 Xerosic 能对已受伤的对手 Active 或其资源形成明确限制：
   使用 Xerosic，然后攻击；
2. 否则不为了后场“提前进化”消耗 Dawn/Hilda，直接用 Alakazam 攻击；
3. 将 Dawn/Hilda 留到下一回合，除非它们能改变本回合是否能形成有效 Alakazam 攻击。

### 0.4.9 第六次讨论结论：牌库低于安全线时，只有终局闭环才允许过牌

用户确认：牌库数量不是一看到变少就立即停止，而是分成普通区间和中局保护线：

- 牌库剩余超过 15 张时，可以相对自由地把牌库资源转化为手牌和本回合可用资源；
- 牌库剩余 11–15 张时进入轻度警戒，开始关注每次检索会消耗多少牌库；
- 牌库剩余正好 10 张时进入中局保护，严格审查所有会把卡牌从 deck 放入手牌的
  抽牌、检索和查看后取回动作；
- 10 张是 V6 初版的测试参数，后续可以根据 replay 中的具体操作调整，不是官方规则；
- 进入中局保护后，普通的过牌、增加手牌伤害或寻找一般资源，都不再自动证明继续
  消耗牌库是正确的。

低于 10 张时，只有下面的完整终局路线能够成立，才允许继续过牌：

```text
继续过牌
  → 把足够的牌拿到手里，增加 Alakazam 的伤害
  → 如有需要，使用 Boss 拉出可以击倒的 Bench 目标
  → 本回合攻击并完成 KO
  → 取得最后的 Prize，直接获胜
```

因此，低牌库时不能因为“这次过牌可以让普通 KO 更稳定”就继续抽，也不能因为
“手牌越多，Alakazam 伤害越高”就无限增加伤害。V6 必须判断的是完整的本回合胜利
闭环：抽牌后的手牌是否足够、是否需要且能够使用 Boss、目标是否能被击倒，以及
击倒后是否真的拿完最后 Prize。只有这些条件同时成立，过牌的牌库风险才值得承担。

这套 10 张保护线是 V6 的策略性风险控制，不是官方规则：官方规则只规定在回合开始
无法抽牌时因牌库耗尽输；卡牌效果抽不到足够数量本身不会立即判负。

Dudunsparce 的牌库净消耗要按实际附着卡计算，而不能把一次 Run Away Draw 简化为
“固定消耗 3 张”：

```text
净牌库变化 = 抽出的 3 张 - (Dudunsparce 本体 1 张 + 所有附着卡数量)
```

例如没有附着卡时净减少 2 张；有 1 张附着卡时净减少 1 张；有 2 张附着卡时净减少
0 张。Enriching Energy 是否让这次抽牌接近零消耗，要以 observation 中实际附着卡
数量计算，而不能只看到卡名就写死一个结果。

### 0.4.10 第七次讨论结论：恢复是资源寻找流程的最后手段

用户确认：Night Stretcher 和 Lana's Aid 不应该因为弃牌区里有高阶段宝可梦就被立即
使用。弃牌区回收的正确时机，是我们已经明确知道下一条路线缺什么，并且这次回收能
直接补齐该路线。

当同时缺少宝可梦和能量时，V6 的资源寻找顺序暂定为：

1. 先识别当前路线缺少的组件：Basic/进化宝可梦、Psychic Energy、Rare Candy，或
   其他必要的检索牌；
2. 先使用不占本回合 Supporter 次数的过牌和检索手段，例如 Dudunsparce 的 Ability
   以及其他可用的 Item；
3. 如果弃牌区只需要回收一只 Pokémon 或一张 Basic Energy，且 Night Stretcher
   可以补齐这个缺口，可以使用一个 Supporter 完成另一部分检索，再使用 Night
   Stretcher。Night Stretcher 是 Item，不会消耗第二次 Supporter 机会；
4. 如果需要同时从弃牌区拿回 Pokémon 和 Basic Energy，Lana's Aid 是首选。它本身
   就是 Supporter，可以用一次 Supporter 机会完成原本需要“Supporter + Night
   Stretcher”才能完成的组合；此时不能再把 Hilda/Dawn 等另一个 Supporter 接在
   前面或后面；
5. 如果回收仍不能形成明确的下一步，就暂时保留回收牌，不为了“让弃牌区变干净”而
   消耗它。

如果缺口已经明确，Lana's Aid 的最多三张回收能力可以一次性补齐组合，例如同时拿回
Alakazam 和 Energy，或拿回 Kadabra、Alakazam 和 Energy。即使本回合无法直接完成
Alakazam，也可能先完成合法的 Abra → Kadabra 自然进化并获得后续资源；这时是否
立即附能不再是唯一的判断重点。

因此，恢复动作的评价不是“哪张卡的阶段最高”，而是：

> 这次回收是否在当前回合或下一回合补齐一条已经确认的有效攻击路线？

### 0.4.11 第八次讨论结论：V6 不使用 Trading Places

用户指出了之前 handoff 判断中的关键规则错误：Dunsparce 的 Trading Places 是一项
攻击，而不是“先换位、再继续攻击”的普通 Ability。使用它会消耗本回合的攻击行动，
因此即使它把 Bench 的 Alakazam 换到 Active，也不可能在同一个回合再使用 Alakazam
进行有效攻击。

V6 初版明确禁用 Trading Places，不把它计入任何可靠的 `handoff_path`。原因不是它
不合法，而是它在这套卡组的计划中没有正收益：

1. 如果 Bench 的 Alakazam 已经带有 Psychic Energy，且 Retreat 后本回合可以由它
   进行有效攻击，就应立即 Retreat 并完成 Alakazam 攻击；不能为了让 Dunsparce
   “挡一回合”而放弃当前的攻击机会；
2. 只有在 Retreat 后 Alakazam 仍然不能攻击，例如缺少能量且本回合没有额外的附能
   或其他补充方式时，才考虑让 Dunsparce 留在 Active 承受对方攻击；
3. 使用 Trading Places 既不能让 Alakazam 在本回合攻击，又会增加 Alakazam 被对手
   处理的风险，因此不应作为“没有其他事情可做时”的兜底动作。

后续 handoff 只考虑合法 Retreat 或引擎提供的被动换位路径，不再考虑 Trading Places。

### 0.4.12 第九次讨论结论：普通 Retreat 是攻击前的换位动作

用户进一步明确了普通 Retreat 与 Trading Places 的区别：普通 Retreat 虽然会消耗
本回合的 Retreat 机会并丢弃 Retreat Cost 对应的能量，但它不会结束回合。只要换到
Active 的 Alakazam 已经具备攻击条件，或者可以在本回合完成必要附能，就应先 Retreat，
再用 Alakazam 攻击。

因此，V6 的 Retreat 判断顺序是：

1. 先检查 Bench 是否有带 Psychic Energy 的 Alakazam；
2. 如果有，且 Retreat 后可以攻击，立即执行 Retreat，再攻击；
3. 如果 Alakazam 缺能量，检查本回合是否还有合法附能或其他补充能量的方式；
4. 只有换位后仍不能攻击时，才把 Dunsparce 留在 Active，利用它较低的牺牲价值
   承受对手攻击。

这条规则再次说明：V6 不能只看“Active 被击倒的风险”，而必须先判断 Retreat 是否
能在本回合产生有效的 Alakazam 攻击。能攻击时，攻击节奏优先于让 Dunsparce 留场。

### 0.4.13 第十次讨论结论：Land Collapse 只作为已观察到的牌库压力

用户确认 V6 不允许预设对手类型或完整对手构筑。`Land Collapse` 只能在 observation
中实际看到 Great Tusk、其攻击条件或相关攻击压力后，作为当前局面的事实参与决策。

官方卡表中的 Great Tusk TEF 97 文本是：其 `Land Collapse` 攻击通常弃掉对手牌库顶
1 张；如果对手本回合使用过 Ancient Supporter，则额外弃掉 3 张。因此它的威胁是
连续多个回合把牌库消耗掉，而不是一张攻击牌立即造成规则上的牌库失败。

V6 对这种已观察到的牌库压力不建立独立 matchup 分支，也不额外要求比通用规则更保守，
而是保持正常的进攻节奏，只收紧已有的通用原则：

1. 优先把当前有效攻击转化为 Prize，尽快缩短对局；如果 Active 不能击倒而 Bench
   有确定 KO 目标，按 Boss 规则制造 Prize；
2. 牌库进入 10 张中局保护线后，停止没有直接收益的可选抽牌和牌库检索；只有能通过
   “过牌增加伤害 → 必要时 Boss → 本回合拿完最后 Prize”完成终局闭环时，才继续
   消耗牌库；
3. 如果 observation 显示对手有特殊能量或其他可见的攻击条件问题，仍按普通的
   Enhanced Hammer、Xerosic 等动作价值判断处理；不能仅因为对手可能是某种构筑就
   提前使用这些牌。Xerosic 本身只压低对手手牌，Enhanced Hammer 也只处理特殊能量，
   它们都不是 Land Collapse 的默认直接反制；
4. 如果我们没有及时建立 Alakazam、没有拿 Prize，或者只是继续抽牌，不能把这类败局
   简化成“Land Collapse 太强”。现有 replay 的主要教训仍然是：先完成攻击线并把
   攻击轮次转化为 Prize。

因此，Land Collapse 的准备不是“预先针对 Great Tusk 改变整套策略”，而是：观察到
牌库压力后，缩短胜利时间、收紧可选过牌，并只针对当时可见的能量和攻击状态做处理。

### 0.4.14 第十一次讨论结论：用资源账本和动作历史恢复规则时序

用户进一步确认：V6 不能只依赖 observation 当前这一帧来猜测进化是否合法，而要在
agent 内记录每个己方回合发生过的动作和关键时间节点。

资源账本以固定卡组总量为起点。对于本卡组的进化线，Abra、Kadabra、Alakazam 各有
4 张，总数只能分布在 Prize、手牌、Active、Bench、弃牌区和牌库这六类位置。第一次
看到牌库或可见卡牌区时，初始化这些数量；之后每次抽牌、检索、放置、进化、附能、被
击倒、拿 Prize、弃牌和洗回牌库都更新账本，并用总量和 `deckCount` 交叉核对。

如果某些 Prize 卡的具体身份暂时不可见，就记录“已知数量”和“未知数量”的边界，
不能把未知卡错误地当成仍在牌库；一旦牌库查看、Prize 暴露或其他 observation 使
身份可知，再把账本收紧到具体卡牌。

动作历史至少包含：

- 每个己方回合开始时 Active/Bench Pokémon 实例和位置的快照；
- 每只 Pokémon 进入场面的回合，以及每次进化发生的回合；
- 本回合已经使用的 Supporter、手动附能和 Retreat；
- 本回合是否已经宣告攻击或主动选择结束。

回合边界由规则动作确定：宣告攻击，或在所有计划动作完成后选择主动结束，都标记
当前己方回合结束；下一次己方回合开始时建立新的场面快照。这样，策略可以判断一只
Pokémon 是否在回合开始时已经在场、是否本回合刚放下、是否已经进化过，而不是只看
当前卡牌 ID。

simulator 暴露的 `supporterPlayed`、`energyAttached`、`retreated`、`appearThisTurn`
等字段用于合法性和账本校验；agent 自己维护的动作历史用于补充“什么时候发生”的
语义。两者不互相替代，出现不一致时应优先停止高风险动作并记录诊断信息。

## 1. 已确认的基线：v5_auto_iter 现在在做什么

### 1.1 执行模型

`agent(obs)` 的入口和策略骨架已经比较清楚：

1. `select is None`：清空多步效果 serial，返回 60 张卡组。
2. `type == 0 and context == 0`：进入 `_main_action()`，为所有合法主行动计算
   `(priority, tie_break, option_index)`，选择最低分。
3. 其他选择：进入 `_select_effect()`，按 context 选择进化目标、检索卡牌、能量、
   伤害目标、切换目标或数量，然后推进 effect serial。

因此 v6 的核心问题不是“能不能返回合法 option”，而是如何给当前合法 option
定义更好的长期价值。

### 1.2 V5 基线行为与 V6 的处理方向

以下内容是 V5 auto_iter 的历史行为和可复用经验，不等于 V6 已批准的实现。V6 会把
它们重新放入规则语义、资源账本和动作历史中判断。

- **攻击线保护**：V5 曾用场上数量和 ready 数量决定 Poffin/Hilda 何时转向
  Dudunsparce；V6 改用真实的进化路线和 handoff path。
- **一条进化线通常只需要一张 Psychic Energy**：这条能量不重复附着原则保留。
- **自然进化与 Rare Candy 并存**：Active Kadabra 已有能量且手中有 Alakazam 时，
  优先自然进化；Active Abra 在合法的第二回合直通时保留 Rare Candy 路线。
- **Mist Energy 视作当前 runtime 的伤害阻断**：Alakazam 对带 Mist 的对手 Active
  不应继续把攻击当成有效伤害，是否使用 Enhanced Hammer 仍须看可见的特殊能量和
  实际攻击价值。
- **Boss / Fezandipiti 有条件使用**：Boss 是 Supporter，Fezandipiti 是 Pokémon
  Ability；二者都不能被当作无条件过牌。
- **每局清除 effect serial**：这是生命周期正确性修复，必须保留。

### 1.3 官方十场 replay 给出的事实

来源：`docs/reports/kaggle/alakazam-v5-auto-iter-episodes-2026-07-19/analysis.md`。
这组样本不是同一对手、同一先后手的 paired experiment，且 V5 与 auto_iter 的
`deck.csv` 顺序不同，所以只能作为方向证据，不能把差异全部归因于代码。

| 指标 | v5_auto_iter | 设计含义 |
|---|---:|---|
| 胜 / 负 | 5 / 5 | 样本表现优于同批 V5 的 2 / 8，但仍不够稳定 |
| 首只 Alakazam | 9/10，平均 turn 5.56 | “完全做不出主攻”的比例下降 |
| 首次任意攻击 | 10/10，平均 turn 3.40 | 有早期动作，但不等于有效 Prize 推进 |
| Abra 线攻击次数 | 平均 2.50 | 连续攻击比 V5 的 1.90 更充分 |
| 最大 ready attacker | 平均 2.00 | 场面数量变好，但不保证下一回合能交接 |
| 牌库耗尽败局 | 2 场 | 抽牌 / 场面资源还没有和胜利节奏绑定 |

报告中最重要的警告是：`86828026` 在有两只 ready attacker 的情况下仍然先把牌库
抽空，而 `86820080` 则完全没有完成 Alakazam；所以“ready 数量”不能继续作为
完整策略的唯一代理指标。

## 2. 为什么需要重新设计，而不是继续堆叠局部 gate

### 2.1 “有攻击”与“有效攻击”没有被区分

当前策略可以很早选择 Kadabra 或 Dunsparce 的低阶段攻击，但官方复盘显示，早攻
并不必然让我们更接近胜利：有些局只是消耗了攻击轮次，随后仍然没有 Alakazam、
没有 Prize 转换，或把 Active 留在了无法继续接力的位置。

V6 需要明确区分至少三件事：

- **立即价值**：这次攻击能否击倒、拿 Prize 或处理明确的威胁？
- **接力价值**：这次攻击后，下一回合是否仍有可攻击的 Alakazam 路径？
- **节奏代价**：选择低阶段攻击是否放弃了本回合能找到 Rare Candy / Alakazam、
  自然进化或形成更高 Prize 价值的动作？

### 2.2 “Rare Candy 优先”也不能成为另一种僵化规则

历史 review 中已经指出两个看似相反、但其实都必须成立的条件：

- 第二回合从 Active Abra 直通 Alakazam 时，Rare Candy + Alakazam 是重要路线；
- 第三回合以后，Active Kadabra 已带能量且手里有 Alakazam 时，自然进化往往更
  简单，也更有利于维持攻击线。

所以 v6 不应问“Rare Candy 还是自然进化谁永远优先”，而应问：
**哪条路线能以最低资源成本，在本回合或下一回合形成可连续攻击的 Active？**

### 2.3 “ready attacker 数量”不等于真正的接力能力

一只 Bench Alakazam 只有在以下条件同时满足时，才应被计入可靠接力：

- 它带有能满足攻击需求的 Psychic Energy；
- 当前 Active 能通过合法 Retreat 或被动换位交给它；
- 交接不会浪费关键能量或把两奖目标暴露给对手；
- 交接后仍有一只下一顺位的攻击线，或牌库/弃牌区有现实恢复路径。

V6 需要从 `ready_count` 升级为可解释的 **handoff path** 状态。

### 2.4 抽牌保护需要服从胜利节奏，而不是只看牌库张数

`_draw_is_blocked()` 当前主要看“抽牌是否直接创造击倒、手牌是否超过 20、牌库
是否 ≤ 10、当前是否已经能击倒”。这能防止一部分无意义过牌，但还没有回答：

- 当前抽牌是不是为了下一只打手的明确缺件？
- 当前已有两只 ready attacker，却还没有拿到任何 Prize 时，继续抽牌是否合理？
- 牌库检索动作和抽牌动作是否应该消耗同一份牌库预算？
- Dudunsparce 抽牌并按附着卡数量洗回自身的净消耗是多少？

V6 需要一个“牌库预算 + Prize 进度 + 下一只打手缺件 + 净牌库变化”的联合 gate，
10 张只是第一版保护参数，而不是继续单独调整 `DECK_DRAW_STOP` 就能解决的问题。

## 3. 暂定的完整策略状态模型

以下不是代码接口，而是讨论时统一语言的状态向量：

```text
turn_context
├─ 立即攻击：Active 能否攻击？能否击倒？攻击是否只是低阶段 fallback？
├─ 主攻路线：Active/Bench 的 Abra → Kadabra → Alakazam 合法进化路径
├─ 交接路线：下一只攻击手是否有 Psychic、能否 Retreat/接班
├─ 资源缺口：手牌、弃牌区、牌库中缺少什么，Supporter 是否能补齐
├─ Prize 节奏：我方剩余 Prize、对方剩余 Prize、当前目标的奖赏价值
├─ 牌库预算：本回合过牌收益、抽牌后剩余牌库、是否存在 deck-out 约束
└─ 对手威胁：Mist、特殊能量、墙、两奖目标、手牌伤害、Land Collapse 等
```

每个合法 action 的价值都应该能回答一句话：

> “执行它之后，距离下一次有意义的 Prize 推进更近了吗？如果没有，它是否在
> 保护一个明确的攻击/接力路线？”

## 4. 已确认的第一性方向：先从规则语义重建目标

基于上面的要求，V6 的总目标暂定为：

> 在宝可梦卡牌规则允许的语义下，最大化未来 1–2 回合形成有效 Prize 推进的概率，
> 同时避免破坏进化时机、攻击接力、手牌控制和牌库生存。

这比“本回合能不能攻击”更准确，也比“永远等 Rare Candy + Alakazam”更灵活：

- 可击倒时，立即击倒仍然是最高收益；
- 不可击倒时，低阶段攻击必须证明它能保护接力、处理威胁或缩短 Prize race；
- 如果低阶段攻击只是“合法且有一点伤害”，抽牌、提前进化、铺场或 END 可以胜过它；
- 但如果 Active Kadabra 已能自然进化并攻击，就不能因为等待 Rare Candy 而浪费这回合；
- 任何抽牌动作都必须同时回答“抽完之后如何赢”和“抽完之后牌库是否仍然安全”。

因此接下来不再先讨论某张卡的 score，而是按以下顺序重建：

```text
宝可梦卡牌规则不变量
        ↓
当前局面与未来一回合的可执行状态
        ↓
当前阶段与胜利路线
        ↓
动作是否创造有效 Prize / 接力 / 生存价值
        ↓
在合法 option 中确定性选择
```

**实现边界已经确认：V6 固定 v5_auto_iter 的 `deck.csv`，只重构 `main.py` 策略与
说明。** 本轮不重新讨论牌表；这样可以把行为变化归因到策略重构，并让 replay 复盘
真正回答“规则语义重建是否有效”。

## 5. 当前设计状态与剩余问题

前面的核心问题已经通过讨论形成初步决定：低阶段攻击是最后手段；Active Kadabra
优先自然进化；有 Enriching Energy 且 Active 已能攻击时默认启动 Dudunsparce；Boss
优先制造确定 KO；Supporter 只使用一次；Lana's Aid 负责明确的 Pokémon + Energy
回收组合；牌库 10 张进入保护；Trading Places 禁用；对手策略只根据 observation
中的事实触发。

V6 实验已经把这些决定落成 `main.py` 的统一顺序。主行动使用
`preparation_is_due()` 作为攻击前硬性 gate，再在准备阶段内做确定性排序；它不是
完整引擎，而是本卡组规则语义的最小实现。当前仍需要用官方 replay 校准：

1. **动作计划校准**：Boss、Xerosic、Enriching Energy、Dudunsparce Ability、进化、
   Retreat 和攻击的固定优先级是否在不同对局节奏中稳定；
2. **handoff path 细化**：当前实现已经使用 Pokémon serial、附着能量和 Retreat
   可用状态，仍需观察对手击倒后交接是否过于保守；
3. **牌库账本边界**：当前只把 observation 可见卡牌记入已知区，Prize/隐藏手牌继续
   保留未知数量，需用 replay 检查交叉核对；
4. **10 张保护线的 replay 校准**：确认它是否应保持为 10，或根据真实操作调整。

### 5.1 实验实现与设计的对应关系

| 设计约束 | V6 实现入口 |
| --- | --- |
| 攻击是终止提交 | `_main_action()` 的 `preparation_is_due()` gate 与攻击分支 |
| 进化时钟与历史 | `TurnMemory`、`_active_evolution_is_legal()`、`_has_evolvable_abra()` |
| Supporter/附能/Retreat 预算 | `TurnMemory.record_main_action()` 与 `_main_action()` 的预算 gate |
| Boss 高 HP 确定 KO | `_boss_ko_targets()`、`_choose_switch_option()` |
| Xerosic 六张手牌条件 | `_xerosic_is_worth_playing()` |
| Dudunsparce/Enriching 净牌库变化 | `_dudunsparce_net_deck_change()`、`_v6_draw_is_blocked()`；Enriching 使用抽 4 张的牌库预算 |
| Trading Places 禁用 | 攻击分支直接将 `TRADING_PLACES_ATTACK` 放到 END 之后 |

## 6. 讨论纪律与验收标准

- 每个新规则先写成“触发条件 → 期望收益 → 可能副作用”，再决定是否实现。
- 不以单场 replay 证明策略正确；至少要能解释它改善哪一类失败，并设计可比较的
  replay 指标。
- 指标至少包括：首只 Alakazam、首只 Alakazam 攻击、有效攻击次数、每回合是否有
  handoff path、首次 Prize 回合、连续 Prize 间隔、终局牌库、剩余 Prize 和
  deck-out 原因。
- 评测时尽量固定 deck 顺序，或做同一对手/先后手的 paired replay，避免把抽牌顺序
  与策略代码混为一谈。
- V6 实验目录已经包含必要的策略运行文件；`deck.csv` 仍必须与 v5_auto_iter 保持一致。

## 7. 参考材料

- 基线实现：`submission/alakazam_v5_auto_iter/main.py`
- 基线说明：`submission/alakazam_v5_auto_iter/STRATEGY.md`
- auto_iter 迭代记录：`submission/alakazam_v5_auto_iter/CHANGELOG.md`
- 最新官方 replay 复盘：`docs/reports/kaggle/alakazam-v5-auto-iter-episodes-2026-07-19/analysis.md`
- 流程可视化：`submission/alakazam_v5_auto_iter/strategy-flow.html`
- 官方规则学习：`docs/reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`

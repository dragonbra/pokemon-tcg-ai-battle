# Alakazam V8 卡组笔记

这份笔记以 V8 的实际 `deck.csv` 为准，结合 V7 AutoIter iter-46 延续下来的进化、
附能、过牌、恢复、干扰和攻击策略。它是给人复核和下一轮策略迭代使用的说明，
当前不会被 agent 自动读取。

## 阅读方式与总边界

- **卡面事实**：以 `data/official/EN_Card_Data.csv` 和模拟器实际提供的合法 option
  为准。卡牌效果改变区域时，要同步考虑 Prize、手牌、Active、Bench、弃牌区和牌库。
- **V7 已关注/已实现的语义**：代表当前继承策略明确要求的行为，例如攻击是回合终止
  提交、Active Abra 不使用攻击、低阶段非 KO 攻击最后再考虑，以及 Item Lock 下改走
  自然进化和过牌。
- **V8 建议**：根据这份新卡表和规则提出的候选理解，还需要后续 replay、fixture 或
  Kaggle 结果确认后，才能转成代码规则。这里不会把候选建议写成当前 agent 已经自动执行。

### 卡组级规则

1. **攻击是本回合的终止提交。** 先处理合法的进化、Ability、过牌、附能、Supporter、
   换位和终局路线；宣告攻击后不能再做主行动。没有伤害的 `Trading Places` 也仍然是
   攻击，不能把它当作普通换位。
2. **进化看回合开始时点。** 本回合刚放下的 Basic Pokémon 不能进化，同一只 Pokémon
   同回合不能连续进化；`Rare Candy` 不能绕过这些时机限制。
3. **资源预算独立计算。** 每回合 Supporter、手填 Energy、Retreat 各一次；Item、
   Ability 和进化没有统一次数上限，但要遵守各自卡文和合法 option。Supporter 之间要
   比较 Boss、搜索、恢复和 Xerosic 的机会成本。
4. **先分配攻击者角色。** 带 Psychic 的 Abra 只要能验证 `Rare Candy + Alakazam`
   直通路线，就不要先把它自然进化成 Kadabra；让其它无能量、合法入场的 Abra 先自然
   进化并使用 `Psychic Draw`。Active Kadabra 已带 Psychic 且手里有 Alakazam 时，
   则优先自然进化 Active，再处理 Bench 过牌和攻击。
5. **控制牌库和手牌。** 牌库超过 15 张时动作相对自由，11–15 张开始警戒，10 张及以下
   默认保护牌库。所有从 deck 到 hand 的检索、抽牌和 `Run Away Draw` 都要计入预算；
   只有能够在本回合闭合最后 Prize 的终局路线，才允许越过保护线。Alakazam 的伤害依赖
   手牌数量，所以过牌不能只看抽了几张，还要看是否仍有能完成攻击的能量和进化资源。
6. **对手相关效果只按实际观察触发。** 不预设对手 ID 或完整构筑；只在 observation
   中看到特殊能量、Tera Pokémon、对手手牌数量、伤害和可击倒目标时，才启用相应的
   Hammer、Nighttime Mine、Xerosic 或 Boss 判断。

## V8 实际卡表逐卡说明

| 类型 | 卡牌 | ID | 数量 | 说明状态 | 你的特殊用法说明 |
|---|---|---:|---:|---|---|
| Pokémon | Abra | 741 | 4 | 已填写 | **卡面事实：** Basic Pokémon，`Teleportation Attack` 需要一张 Psychic，攻击后把自己和一只 Bench Pokémon 互换。<br>**V7 已关注：** Active Abra 带 Psychic 且手里有 `Rare Candy + Alakazam` 时，优先保留它作为本回合的直通主攻击者；如果 Bench 有其它合法入场、无能量的 Abra，先让那只自然进化成 Kadabra 过牌。<br>**V8 建议：** 四张 Abra 是 Poffin 和 Telepath 的主要落点；不要为了 10 点伤害使用 `Teleportation Attack`，换上来的宝可梦通常还没有 Psychic，会破坏下一回合的能量和攻击节奏。 |
| Pokémon | Kadabra | 742 | 4 | 已填写 | **卡面事实：** 从手牌进化时可使用一次 `Psychic Draw` 抽 2；`Super Psy Bolt` 需要一张 Psychic，造成 30 点。<br>**V7 已关注：** 当带能量的 Abra 要走 `Rare Candy + Alakazam` 时，优先让其它无能量 Bench Abra 自然进化并使用 Ability；但 Active Kadabra 已有 Psychic、手里有 Alakazam 且自然进化后能形成确定 KO 时，先处理 Active。<br>**V8 建议：** 四张 Kadabra 让自然进化过牌成为稳定的非 Item Lock 路线；非 KO 的 `Super Psy Bolt` 只在没有任何能增加当前或下一次攻击成功率的动作时使用。 |
| Pokémon | Alakazam | 743 | 4 | 已填写 | **卡面事实：** Stage 2，进化时可使用 `Psychic Draw` 抽 3；`Powerful Hand` 需要一张 Psychic，并为手牌中的每张卡在对手 Active 放置 2 个伤害指示物，也就是每张手牌 20 点伤害。<br>**V7 已关注：** 先完成所有仍有明确收益的进化、Ability、过牌、Boss、Xerosic 和接力准备，再提交攻击；不要连续消耗 Rare Candy 制造没有即时收益的第二、第三只 Alakazam。<br>**V8 建议：** 每次攻击前重新计算手牌和牌库，不要把 `Enriching Energy` 当作 Psychic；如果已能确定 KO 或完成最后 Prize，攻击价值高于可选的过牌。 |
| Pokémon | Dunsparce | 305 | 3 | 已填写 | **卡面事实：** 70 HP Basic，能被 Buddy-Buddy Poffin 找到；`Trading Places` 用一张无色交换自己与一只 Bench Pokémon，但这是攻击提交，`Ram` 需要两张无色造成 20 点。<br>**V7 已关注：** Dunsparce 的主要职责是铺在 Bench 上，等待 `Dudunsparce` 和 `Enriching Energy` 形成过牌引擎；不把 `Trading Places` 当普通换位。<br>**V8 建议：** 只有其它动作不能改善局面且它能明确 KO，或交换后能让已经带 Psychic 的 Alakazam 在本回合完成终局攻击时，才考虑 Dunsparce 的非 `Trading Places` 攻击；通常让它留在 Bench 比送出低价值攻击更好。 |
| Pokémon | Dudunsparce | 66 | 2 | 已填写 | **卡面事实：** Stage 1，`Run Away Draw` 每回合抽 3；只要因此抽到卡，就把这只 Dudunsparce、进化堆叠和所有附着卡洗回牌库。<br>**V7 已关注：** 只有存在合法接班者时才使用 Ability；Bench 为空时不能把唯一 Active 洗回牌库。洗回数量要计入牌库模型：牌库净减少量按抽 3 减去洗回的本体和全部附着卡计算。<br>**V8 建议：** 本卡只有两张，比 V7 少一条循环资源；应把它当成一次有成本的手牌恢复，而不是每回合默认使用。终局时可以识别 `Run Away Draw → 换上带 Psychic 的 Alakazam → Powerful Hand` 闭环，但非终局仍要保留接力和牌库安全。 |
| Pokémon | Fezandipiti ex | 140 | 1 | 已填写 | **卡面事实：** Basic Pokémon ex，带 Rule Box、通常给对手两张 Prize；若我方 Pokémon 在对手上一回合被击倒，可使用一次 `Flip the Script` 抽 3，每回合最多使用一次。<br>**V7 已关注：** 手牌被干扰后、没有立即 KO 且仍有 Alakazam 或高伤害路线时，可以先 Bench 再使用 Ability；不要把它的 Ability 当作 Supporter。<br>**V8 建议：** 它不能被 Poffin、Poké Pad 或其它“无 Rule Box Pokémon”搜索直接找到，也不应为了多抽 3 张而延误最后 Prize 攻击；把两 Prize 风险和 Bench 空间一起计算。 |
| Pokémon | Shaymin | 343 | 1 | 已填写 | **卡面事实：** Basic、无 Rule Box、80 HP；`Flower Curtain` 只防止对手 Pokémon 的攻击对 Bench 上无 Rule Box Pokémon 造成的伤害，不保护 Active，也不防止攻击的其它效果。<br>**V7 已关注：** 它通常是低 Prize 的备用 Active 或起手 Pokémon，不是 Psychic 攻击线；需要 Retreat 时才考虑给它支付撤退费用。<br>**V8 建议：** 80 HP 高于 Poffin 的 70 HP 上限，不能用 Poffin 找；不要为了它消耗攻击线 Psychic，也不要把 Bench 保护误判为 Active 保护。 |
| Energy | Basic {P} Energy | 5 | 2 | 已填写 | **卡面事实：** 基础 Psychic，能够支付 Abra、Kadabra 和 Alakazam 的一张 Psychic 攻击费用，也能支付其它 Psychic Pokémon 的合法费用。<br>**V7 已关注：** 每回合手填只有一次；当前 Active 的确定攻击路线优先，其次给可见的 Bench 接力 Abra。需要同时恢复 Pokémon 和 Basic Psychic 时，Lana's Aid 通常优先于分开使用其它回收。<br>**V8 建议：** 本卡只有两张，且移除了 `Wondrous Patch`，不要把手填浪费在已经有 Psychic 的同一只 Pokémon 上；保留至少一条现实的 Alakazam 接力线比给 Dunsparce 贴普通 Psychic 更重要。 |
| Energy | Enriching Energy | 13 | 1 | 已填写 | **卡面事实：** ACE SPEC，附着后提供一张无色；从手牌附着到 Pokémon 时抽 4。<br>**V7 已关注：** 它不能支付 Abra 线的 Psychic 攻击，主要价值是给 Dunsparce/Dudunsparce 过牌；在 Active Alakazam 已经能攻击、且过牌不会错过确定 Prize 时，优先转化为 `Enriching Energy → Run Away Draw`。<br>**V8 建议：** 这是整副牌唯一一张，附着抽 4 后再由 Dudunsparce 抽 3，且 Dudunsparce 会把自身和附着的 Enriching Energy 洗回牌库；要把新增手牌、洗回数量和牌库保护一起计算，不能把它当普通无色能量使用。 |
| Energy | Telepath Psychic Energy | 19 | 4 | 已填写 | **卡面事实：** 提供一张 Psychic；从手牌附到 Psychic Pokémon 时，可从牌库找最多两只 Basic Psychic Pokémon 放到 Bench，然后洗牌。<br>**V7 已关注：** 它既是攻击能量又是铺场动作，优先给 Abra/Kadabra/Alakazam 线中能形成接力的 Psychic Pokémon；检索效果找的是 Basic Psychic Pokémon，不是 Basic Psychic Energy。<br>**V8 建议：** 四张 Telepath 是 V8 的主要 Bench 扩展手段，但每次附能和检索都消耗牌库；低于保护线时不要为了多铺一只没有后续进化或 Retreat 计划的 Basic 而盲目使用。 |
| Trainer | Rare Candy | 1079 | 3 | 已填写 | **卡面事实：** Item，把合法在场的 Basic Pokémon 直接进化为对应 Stage 2；它不能绕过“本回合刚放下不能进化”或同回合连续进化的规则。<br>**V7 已关注：** 优先用于带 Psychic、手里已有 Alakazam、当前回合确实能攻击的 primary attacker；第一只 Alakazam 已经可攻击后，不默认继续消耗 Rare Candy 做第二只。<br>**V8 建议：** 使用前先确认 Alakazam、Psychic、攻击 option 和手牌伤害都能闭合；对手实际触发 Budew 的 Item Lock 后，Rare Candy 路线暂停，改用自然进化和过牌。 |
| Trainer | Enhanced Hammer | 1081 | 4 | 已填写 | **卡面事实：** Item，只能弃掉对手一只 Pokémon 身上的一张 Special Energy，不能处理 Basic Energy。<br>**V7 已关注：** 只有在实际看到特殊能量，且弃除后能改变 Powerful Hand 的伤害、KO 或对手攻击门槛时，才提高优先级；面对 Mist 等保护性特殊能量时，先拆除再重新判断伤害。<br>**V8 建议：** 数量由 2 增至 4，说明构筑更重视特殊能量干扰，但不等于任何特殊能量都应立即拆；无合法目标、对手只有基础能量或拆除不改变本回合结果时保留。Item Lock 时不能使用。 |
| Trainer | Buddy-Buddy Poffin | 1086 | 4 | 已填写 | **卡面事实：** Item，从牌库找最多两只 HP 70 或以下的 Basic Pokémon 放到 Bench；可以找 Abra 和 Dunsparce，不能找 80 HP 的 Shaymin 或 210 HP 的 Fezandipiti ex。<br>**V7 已关注：** 优先补足三只 Abra 攻击线，再建立至少一条 Dunsparce 底座；不为了“铺满”而让 Bench 没有接力空间。<br>**V8 建议：** 每次搜索都消耗牌库，牌库接近保护线时要比较手牌伤害和未来攻击者数量；Bench 已满、没有具体进化/过牌路线或当前终局更重要时不要盲目使用。Item Lock 时不能使用。 |
| Trainer | Night Stretcher | 1097 | 1 | 已填写 | **卡面事实：** Item，从弃牌区把一只 Pokémon 或一张 Basic Energy 放回手牌；不能拿回 Special Energy。<br>**V7 已关注：** 只有它能补齐现实的下一只攻击者、Basic Psychic 或明确的恢复链时才使用；需要同时拿回 Pokémon 和 Psychic 时，优先比较能一次完成两类资源的 Lana's Aid。<br>**V8 建议：** 本卡只有一张，既可以在同回合和另一个 Supporter 组合，也可以作为 Item Lock 之外的窄恢复工具；拿回孤立的 Stage 2 或没有 Basic 底座的卡，不算完成攻击路线。 |
| Trainer | Sacred Ash | 1129 | 1 | 已填写 | **卡面事实：** Item，把弃牌区最多五只 Pokémon 洗回牌库，不会直接把它们放回手牌。<br>**V7 已关注：** 多只 Abra 线被击倒、后续搜索资源不足时用它恢复进化链密度；它不是当前回合立即拿到 Alakazam 的替代品。<br>**V8 建议：** 由于牌库保护线存在，使用后还要通过 Poffin、Poké Pad、Hilda 或 Dawn 把洗回的 Pokémon 找出来；只洗回一张没有现实搜索路线的 Pokémon，通常不值得消耗 Item 和增加牌库。Item Lock 时不能使用。 |
| Trainer | Poké Pad | 1152 | 4 | 已填写 | **卡面事实：** Item，只能从牌库找一只没有 Rule Box 的 Pokémon；可以找 Abra、Kadabra、Alakazam、Dunsparce、Dudunsparce 和 Shaymin，不能找 Fezandipiti ex。<br>**V7 已关注：** 搜索目标要服务本回合可进化或下一次可攻击的具体阶段，不能用“找一只 Pokémon”代替完整攻击路线；它不能找 Energy。<br>**V8 建议：** 四张提供较高的进化链检索密度，但每次都会消耗牌库；低牌库或 Item Lock 时分别执行保护线和禁用 Item 的判断。 |
| Trainer | Boss’s Orders | 1182 | 3 | 已填写 | **卡面事实：** Supporter，把对手一只 Bench Pokémon 换到 Active；每回合最多使用一次 Supporter。<br>**V7 已关注：** 当前 Active 不能 KO、Bench 有确定 KO 目标时优先考虑 Boss；多个都能确定击倒时，优先剩余 HP 较高的目标。若 Active 已能攻击且没有 Boss KO 目标，不要只为了换位消耗 Supporter。<br>**V8 建议：** 在使用前重新计算拉出目标后的 Powerful Hand 伤害和 Prize 数量，并与 Hilda、Dawn、Lana's Aid、Xerosic 的即时收益比较；对手没有可确定 KO 的 Bench 目标时保留。 |
| Trainer | Lana’s Aid | 1184 | 1 | 已填写 | **卡面事实：** Supporter，从弃牌区以任意组合拿最多三张无 Rule Box Pokémon 和 Basic Energy 到手牌；不能拿回 Fezandipiti ex 或 Special Energy。<br>**V7 已关注：** 需要同时恢复 Pokémon 与 Basic Psychic 时优先于 Night Stretcher；恢复动作必须能构成当前或下一次现实攻击，而不是只让手牌看起来更多。<br>**V8 建议：** V8 没有 `Wondrous Patch`，Lana's Aid 是重要的弃牌区 Psychic 接力入口；但它占用唯一 Supporter，若本回合有确定 Boss KO 或终局攻击，就不能为了非必要恢复而延后攻击。 |
| Trainer | Xerosic’s Machinations | 1197 | 3 | 已填写 | **卡面事实：** Supporter，让对手弃牌直到手牌剩 3 张；它不抽我方牌，也不直接造成伤害。<br>**V7 已关注：** 只有 Active Alakazam 已能攻击、对手 Active 已受伤但当前不能 KO、对手手牌至少 6 张且没有 Boss 确定 KO 时，才在攻击前使用；对手手牌较少时通常保留。<br>**V8 建议：** 不要因为它是三张投入就每回合寻找使用机会；先完成 primary attacker、接力和恢复，再判断压到 3 张是否比直接 Powerful Hand 更有价值。 |
| Trainer | Hilda | 1225 | 4 | 已填写 | **卡面事实：** Supporter，从牌库找一只 Evolution Pokémon 和一张 Energy；不能直接找 Basic Abra 或 Basic Dunsparce。<br>**V7 已关注：** 可找 `Alakazam + Psychic` 形成 Rare Candy 路线，也可找 `Kadabra + Enriching Energy` 走 Item Lock 下的自然进化与 Dudunsparce 过牌；没有 Abra 底座时，不要只找孤立 Alakazam。<br>**V8 建议：** 先确定要服务的路线，再选择 Evolution Pokémon 和 Energy；使用 Hilda 后仍要检查回合进化时机、手填 Energy 是否已用以及是否需要保留 Boss/Lana/Xerosic。 |
| Trainer | Dawn | 1231 | 4 | 已填写 | **卡面事实：** Supporter，从牌库各找一只 Basic Pokémon、Stage 1 Pokémon 和 Stage 2 Pokémon；不能找 Energy。<br>**V7 已关注：** 适合一次补齐 `Abra → Kadabra → Alakazam` 的进化链，也可以按合法路线准备 Dunsparce/Dudunsparce；但不能只为了让场面看起来完整而消耗唯一 Supporter。<br>**V8 建议：** 先确认三类进化卡都有现实用途和合法入场时间；如果 Boss 能立刻拿 Prize、Lana 能补齐下一只攻击或 Xerosic 能改变本回合结果，Dawn 不一定是最高优先级。 |
| Trainer | Nighttime Mine | 1266 | 2 | 已填写 | **卡面事实：** Stadium；场上双方每一只 Tera Pokémon 使用的攻击费用增加一张无色。它不是增伤卡，也不是只作用于对手。<br>**V7 已关注：** V7 没有这张牌，不能从对手 ID 或构筑预设它的价值。<br>**V8 建议：** 只在 observation 中实际看到 Tera Pokémon，且增加一张无色确实能改变对手攻击节奏或阻止其攻击时优先使用；若对手没有 Tera，通常没有即时收益。V8 自身没有 Tera Pokémon，因此对我方常规 Alakazam 攻击没有额外费用，但仍要注意 Stadium 替换和对手场面变化。 |

## V8 相对 V7 的构筑影响

V8 的核心 Alakazam 攻击线和大部分过牌、搜索、Supporter 骨架没有改变，变化集中在
以下五项：

| 变化 | V7 | V8 | 对策略理解的影响 |
|---|---:|---:|---|
| Enhanced Hammer | 2 | 4 | 提高实际处理特殊能量的密度，但仍然只在看到合法 Special Energy 目标且能改变结果时使用。 |
| Nighttime Mine | 0 | 2 | 新增针对 Tera Pokémon 的 Stadium；必须按实际观察判断，不把它当作通用 Stadium 或增伤卡。 |
| Dudunsparce | 3 | 2 | `Run Away Draw` 的可重复次数下降，每一次 Ability 和 Bench 接班都更需要有具体收益。 |
| Wondrous Patch | 1 | 0 | 不能再用 Item 直接把弃牌区 Basic Psychic 附到 Bench Psychic Pokémon；要依靠手填、Telepath、Lana's Aid 或其它 Basic Energy 回收。 |
| Battle Cage | 2 | 0 | 没有原先的后场伤害保护，Bench 资源和低 Prize Pokémon 的暴露风险需要通过铺场数量、Shaymin 的实际保护和攻击节奏管理。 |

因此，V8 不是把 V7 的攻击逻辑完全改写：依旧先建立可持续的 Alakazam 接力线，
再在不影响 Prize 的前提下用 Dudunsparce 过牌和 Xerosic 干扰。下一步值得单独验证的
问题是：四张 Hammer 是否提高了实际 KO 路线、Nighttime Mine 在真实 Tera 对局中的
触发价值，以及两张 Dudunsparce 是否增加接力断档或迫使牌库保护更早收紧。

## 使用边界

- 当前版本不会自动读取本文件，也不会因为 `已填写` 状态改变动作选择。
- 这里的数量以 `deck.csv` 为准；每行代表一种卡牌，不代表单张实体。
- 后续把说明转成策略代码时，应先为每条建议补充 observation fixture 或官方 replay
  证据，再决定是否改变动作优先级。

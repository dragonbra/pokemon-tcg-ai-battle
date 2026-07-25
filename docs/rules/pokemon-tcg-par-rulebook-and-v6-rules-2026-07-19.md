# Pokémon TCG 规则学习与 V6 重构约束

## 结论先行

这次学习最重要的不是记住更多卡牌文本，而是把 V6 的决策边界改正确：

1. **Attack 是本回合的终止提交**：选择攻击后，不能再进化、附能、使用 Supporter、
   铺 Bench 或使用普通 Ability。所有本回合想做的准备都必须在攻击前完成。
2. **进化看“本回合开始时是否已经在场”**：本回合刚放下的 Pokémon 不能进化；进化
   后又成为本回合新进入的 Pokémon，不能在同一回合再次进化。Rare Candy 可以跳过
   Stage 1，但不能绕过它自己的首回合/本回合放下限制。
3. **Supporter、手动附能和 Retreat 都是有限资源**：Supporter 一回合一次，手动从
   手牌附 Energy 一回合一次，Retreat 一回合一次。选择其中一个动作会消耗本回合的
   机会，不是“option 还合法就以后再说”。
4. **没有通用手牌上限**：控制手牌是 Alakazam 的伤害与资源策略，不是规则硬上限。
   牌库真正的规则性失败是：在自己的回合开始需要抽牌时，牌库没有牌可抽。
5. **引擎已经负责大多数合法性，agent 要负责语义价值**：simulator 会过滤不能做
   的选项，但不会告诉 agent 这次攻击是否值得放弃剩余主行动，也不会替 agent 比较
   Dawn、Boss、Xerosic 和 Hilda 的唯一 Supporter 机会。

因此，V6 不应继续把“最低 score 的合法 option”当作完整策略，而要先建立回合资源
预算、进化时钟、攻击终止提交和未来攻击路线，再比较动作价值。

## 1. 来源与核验方法

### 官方规则书

- 来源：[Pokémon Trading Card Game Rules](https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/par_rulebook_en.pdf)
- 文件：`par_rulebook_en.pdf`
- 版本信息：PDF 41 页，PDF 1.5，创建时间 2023-07-08；以下页码均为 PDF 页面右下角的印刷页码。
- 使用 `pypdf` 与 `pdfplumber` 提取文字，使用 Poppler 渲染印刷页 9、11、12、13、20
  进行视觉核验。重点页面文字完整，未发现布局截断或提取错位。

### 当前仓库与引擎

- 卡牌文本：`data/official/EN_Card_Data.csv`
- API 状态：当时 `submission/alakazam_v5_auto_iter/cg/api.py:367-376`（源包已退役）
- 主阶段/引擎规则：`engine/source/ptcgProgram 22/GameProc.h`
- JSON 状态导出：`engine/source/ptcgProgram 22/ToJson.h:143-155`
- V5 auto_iter 策略：当时 `submission/alakazam_v5_auto_iter/main.py`（源包已退役）

规则书是通用规则的官方快照；具体卡牌效果以当前卡牌数据和 simulator runtime 为准。

## 2. 一个回合的真实结构

规则书印刷页 9 给出的三段结构是：

```text
1. Draw a card
2. 在任意顺序执行主行动
   A. 从手牌把 Basic Pokémon 放到 Bench
   B. 进化 Pokémon
   C. 从手牌手动附一张 Energy
   D. 使用 Trainer
   E. Retreat Active Pokémon
   F. 使用 Abilities
3. Attack，然后结束回合
```

印刷页 13 再次强调：攻击前先确认第 2 步想做的事情都已经做完；一旦攻击，回合就
结束，不能回头。

### 2.1 攻击是终止提交，不只是一个高收益 option

“攻击”在规则语义上不是普通的可逆动作，而是把当前回合提交出去：

- 攻击前可以继续做回合主行动；
- 宣告攻击后不能再做任何普通主行动；
- 之后只处理攻击步骤、伤害、击倒、取 Prize、对手补 Active 和 Pokémon Checkup；
- 攻击即使造成 0 伤害，或攻击效果失败，它仍然是一次攻击提交；
- 先攻玩家的第一回合跳过攻击步骤，完成其他动作后回合结束。

印刷页 19 还明确说明：没有伤害的招式也仍然算 Attack。当前卡组的
Dunsparce `Trading Places` 因此也是攻击提交，不是一个可以在攻击后继续准备的普通
换位动作。

**V6 约束：**在比较攻击与其他主行动时，必须把攻击的机会成本写进动作价值。即使
当前可以击倒，也不能只因为“能拿 Prize”就跳过对下一只打手、Supporter 唯一机会和
手动附能的计算；要比较本回合攻击后的胜利路线与不攻击、先完成准备后的路线。

## 3. 进化规则与进化时钟

规则书印刷页 11 的核心句子是：只有当进化前的 Pokémon 在**本回合开始时已经在场**，
才能把手中的 “Evolves from X” 卡放上去进化。

它还给出三个约束：

- Pokémon 在它进入场上的第一个回合不能进化；
- 进化后它相当于“新进入场上”，所以同一回合不能再次进化；
- Active 和 Bench 都适用同一规则。

进化会保留所有附着卡和伤害指示物，但清除攻击效果与特殊状态；进化后不能继续使用
进化前 Pokémon 的招式或 Ability，除非卡牌文本另有说明。

### 3.1 Rare Candy 的精确语义

当前卡牌数据 `data/official/EN_Card_Data.csv:1854` 的文本是：

> Choose 1 of your Basic Pokémon in play. If you have a Stage 2 card in your hand that
> evolves from that Pokémon, put that card onto the Basic Pokémon to evolve it, skipping
> the Stage 1. You can’t use this card during your first turn or on a Basic Pokémon that
> was put into play this turn.

因此：

- Rare Candy 可以让**已经在场且满足时机**的 Abra 直接成为 Alakazam；
- Rare Candy 不能让本回合刚放下的 Abra 立刻进化；
- Rare Candy 不能在先攻第一回合使用；
- Rare Candy 跳过的是 Stage 1，不是 Pokémon TCG 的回合时机规则；
- 如果本回合先把 Abra 直接进化成 Kadabra，它已经作为“本回合新进入”的 Kadabra，
  不能再在同一回合用 Alakazam 二次进化。

**V6 约束：**必须显式区分 `present_at_turn_start`、`played_this_turn`、
`evolved_this_turn` 和 `can_use_rare_candy`。不能用“当前场上是 Abra”这一项事实
代替进化时钟。

这解释了为什么要提前准备：本回合先把合法的 Abra 进化到 Kadabra，下一回合才有
Kadabra → Alakazam 的自然进化路线；但如果 Active Abra 已经在场一回合，第二回合的
Rare Candy + Alakazam 也可能是另一条合法路线。两者不是永远固定的优先级，而是不同
进化时钟下的路线。

## 4. 每回合一次的资源

### 4.1 Supporter

规则书印刷页 12：Trainer 中 Item 和 Pokémon Tool 可以按需要使用，但 Supporter
每回合只能使用一张；先攻玩家的第一回合不能使用 Supporter。

这意味着以下动作互斥：

- Dawn：搜索 Basic、Stage 1、Stage 2；
- Hilda：搜索 Evolution Pokémon 与 Energy；
- Boss’s Orders：换入对手 Bench 目标；
- Xerosic’s Machinations：把对手手牌压到 3 张；
- Lana’s Aid：从弃牌区恢复最多三张无 Rule Box Pokémon / Basic Energy。

不是“Boss 现在能 KO、Xerosic 也能干扰，所以两个都可以做”，而是本回合必须在一
个 Supporter 机会中比较它们对整条胜利路线的贡献。

当前引擎确实强制了这一点：`State` 有 `supporterPlayed`，生成 Play option 时在
`GameProc.h:824-830` 检查首回合和本回合 Supporter 标志，实际使用后在
`GameProc.h:144-147` 设置标志，并通过 `ToJson.h:151` 暴露给 observation。

### 4.2 手动附能

规则书印刷页 9、11：从手牌手动附 Energy 一回合一次，可以附给 Active 或 Bench。
卡牌效果额外附能不等同于手动附能；本卡组中的 Hilda、Telepath Energy、Enriching
Energy 等效果要按各自卡牌文本处理。

当前引擎在 `State.h:144-148` 保存本回合 `energyPlayed`，在 JSON 中以
`energyAttached` 导出；`GameProc.h:800-814` 生成手动附能 option 时过滤已使用状态。
由卡牌效果产生的附能走另一条路径，不应被策略错误地当成普通手填次数。

**V6 约束：**附能前必须先比较 Active 当前攻击、下一只 Alakazam 的最低准备成本、
Retreat 成本以及 Enriching/Telepath 的效果价值。不能把“找到一个合法目标”当成附能
决策完成。

### 4.3 Retreat、Stadium、Ability

规则书印刷页 12：

- Retreat 一回合一次；要支付 Retreat Cost，交换后仍然可以攻击；
- Stadium 一回合一次；
- Item / Pokémon Tool 可以按需要使用；
- Abilities 按卡牌文本执行，通用规则允许在一回合使用多个，但单张 Ability 的
  “once during your turn”等限制仍然有效。

本卡组没有 Stadium，但 Retreat 是攻击接力的一部分；Retreat 不能被当成攻击后的
补救动作，而必须在攻击前决定。

## 5. 攻击、伤害、击倒和 Prize

### 5.1 攻击步骤

规则书印刷页 13–14、20 给出的核心顺序：

1. 确认 Active 的攻击能量并宣告攻击；
2. 处理会取消或改变攻击的效果；
3. 处理 Confused 等攻击是否发生的检查；
4. 做攻击要求的目标选择；
5. 做攻击要求的 coin flip 等支付/执行；
6. 处理伤害前效果，再放置伤害指示物和其他效果；
7. 检查所有受影响 Pokémon 是否 Knocked Out；
8. 对手为被击倒 Pokémon 选择新的 Active；
9. 检查取 Prize 或对手无 Pokémon 可替换等胜负条件；
10. 回合结束并进入 Pokémon Checkup。

### 5.2 Alakazam 的伤害语义

当前卡牌数据 `data/official/EN_Card_Data.csv:1309-1310`：

- Alakazam 的 `Powerful Hand` 需要 1 Psychic Energy；
- 对手 Active 每有 1 张 Energy，攻击造成 2 个 damage counter，即每张手牌 20 点；
- 因此手牌数是攻击伤害资源，但不是规则要求的手牌上限。

**V6 约束：**动作价值需要同时看“当前手牌带来的攻击伤害”和“继续抽牌带来的牌库
风险”。不能使用一个任意的手牌上限把抽牌全部禁止，也不能为了把伤害数字做大而无
限制抽空牌库。

### 5.3 Prize 与胜利条件

规则书印刷页 8、14、21：标准游戏开始时每方放置 6 张 Prize；对手 Pokémon 被
Knocked Out 后，攻击者拿自己的 Prize。胜利条件有三种：

1. 拿完自己的所有 Prize；
2. 对手场上没有 Pokémon 可继续作战；
3. 对手在自己回合开始时没有牌可抽。

被击倒的对手如果 Bench 为空或没有其他原因可选新的 Active，攻击者直接获胜；如果
同时拿到最后一张 Prize，也立即获胜。

**V6 约束：**不能把“能造成伤害”当成“在推进胜利”。应区分：能否 KO、拿几张
Prize、是否暴露己方高 Prize Pokémon、攻击后对手是否能立刻完成反击，以及不攻击而
先准备下一回合的真实 Prize 期望。

## 6. 牌库与手牌：规则硬约束和策略约束要分开

### 6.1 牌库抽空的真实规则

规则书印刷页 10、21：

- 如果回合开始时牌库没有牌，无法完成“抽一张牌”，该玩家输掉游戏；
- 如果卡牌效果要求抽取/查看的数量超过剩余牌库，只抽/看剩余牌，游戏继续；
- 不能因为一次 Dudunsparce、Alakazam 或其他效果没有足够数量的牌而直接输掉。

因此“不能把牌库抽空”是重要的长期策略要求，但精确说法是：不能让牌库在未来某个
回合开始时变成无法抽牌的状态，除非已经先赢；也不能把所有抽牌都当作同等风险。

### 6.2 没有通用手牌上限

这本规则书没有给出通用的 20 张手牌上限。`handCount > 20` 可以是当前 V5 策略
为了控制 Alakazam 伤害、检索和牌库风险设置的经验 gate，但不是 Pokémon TCG 的
通用规则。

### 6.3 Dudunsparce 的卡牌级限制

当前卡牌数据 `data/official/EN_Card_Data.csv:131`：Run Away Draw 是
“Once during your turn, you may draw 3 cards”；如果因此抽到了牌，就把 Dudunsparce
和它的所有附着卡洗回牌库。

这带来三个不同问题，不能合并成一个“牌库 ≤ N 禁止抽牌”：

- Ability 本身一回合只能用一次；
- 是否值得为了 3 张牌暂时把 Dudunsparce 与附着卡洗回去；
- 抽牌后牌库是否仍能支撑未来回合开始抽牌和下一只攻击手。

## 7. Simulator 映射：它负责什么，V6 负责什么

### Simulator 已负责的合法性

本地引擎审计确认：

- 攻击 option 满足攻击条件，选择后进入 `SelectedAttack`，并在
  `GameProc.h:229-232` 进入 `TurnEnd`；
- Supporter 首回合与一回合一次由 `GameProc.h:824-830` 过滤；
- 手动附能一回合一次由 `GameProc.h:800-814` 过滤；
- 普通进化由 `GameProc.h:782-787` 和 `State.h:1287-1290` 过滤；
- option 的合法目标、数量和索引由引擎与 API 验证；
- `supporterPlayed`、`energyAttached`、`retreated`、`appearThisTurn` 等状态暴露给
  agent。

### V6 必须负责的语义价值

引擎不会替 agent 判断：

- 当前攻击是否应该终止本回合，而不是先完成准备；
- 两个 Supporter 哪一个是本回合唯一正确机会；
- 这一次手动附能应该给当前 Active、下一只攻击手，还是用于 Retreat/效果；
- 合法进化是否会破坏更高价值的路线；
- 抽牌是否服务于明确的下一次 Prize 推进；
- 攻击后对手的 Prize race、反击和己方接力是否会变差。

这正是当前 `main.py` 的主要缺口：例如 `_main_action` 在 `main.py:1310-1338`
把非 KO 攻击在抽牌受保护时仍设置为高优先级；`main.py:1542-1546` 将 END 固定为
99 后按升序选择；Supporter 的未识别 fallback 在 `main.py:1530` 仍可能比 END
优先；这些都是“合法但未必有语义价值”的路径。

## 8. 写入 V6 设计的全局不变量

V6 后续设计必须先满足以下不变量，再谈动作排序：

```text
TURN_PHASE
  draw → main actions in any order → attack or END → checkup

ATTACK_COMMITMENT
  choose attack ⇒ current turn has no further normal setup actions

EVOLUTION_CLOCK
  can evolve ⇔ pre-evolution was in play at turn start and card text permits it
  evolve once ⇒ that Pokémon cannot evolve again this turn

TURN_BUDGET
  supporter: 1 / turn
  manual hand Energy: 1 / turn
  retreat: 1 / turn
  stadium: 1 / turn
  Ability: according to each card's text

VICTORY_AND_DECK
  win by all Prizes, no Pokémon in play, or opponent cannot draw at turn start
  effect draw may exhaust available cards without immediate loss
  no generic hand-size ceiling
```

这些是设计硬约束，不是可通过 score 抵消的软惩罚。动作价值层只能在约束允许的动作
中比较长期收益。

## 9. 对 v6 的直接设计结论

后续我们应按以下顺序重构，而不是继续增加散落的分数：

1. 先从 observation 提取本回合预算：Supporter、手动附能、Retreat、首回合限制、
   当前是否已经攻击提交；
2. 为每个 Active/Bench Abra 线建立进化时钟和未来两回合路径；
3. 先生成“不攻击也能获得的主行动价值”，再比较攻击提交后的 Prize 价值；
4. Supporter 必须作为一次性机会整体比较，不能让未识别 Supporter fallback 偷走
   本回合机会；
5. 手动附能先评估下一只可攻击 Alakazam 和接力路线，再考虑 Dudunsparce 抽牌；
6. 抽牌 gate 用“未来回合开始抽牌风险 + 当前手牌伤害 + 明确缺件 + Prize 进度”
   联合判断，而不是固定手牌/牌库阈值；
7. 评测新增“攻击前是否完成所有准备”“攻击后仍然需要做的动作数量”“首次 Prize
   回合”“连续 Prize 间隔”“回合开始牌库耗尽原因”等指标。

这份报告只记录规则学习和 V6 设计约束，不改变当前任何 submission 策略代码。

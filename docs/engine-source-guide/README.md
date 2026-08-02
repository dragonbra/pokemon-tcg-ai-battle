# 官方 PTCG Engine Source Guide

这是一份针对 `engine/source/ptcgProgram 22/` 的只读代码导览。它回答四个问题：

1. 官方怎样表示一场游戏和一张卡；
2. immutable card prototype 与 mutable card instance 如何分开；
3. 一次 action 怎样暂停在玩家选择上，再恢复并改变状态；
4. 卡文究竟被建模成通用公式、通用指令，还是逐卡硬编码。

可视化入口：[index.html](index.html)。本文适合查实现细节，HTML 适合先建立全局地图。

## 0. 证据边界

- **官方规则**：回合结构、进化时机、胜利条件等，以项目已核验的[规则证据文档](../rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md)为背景。
- **engine 事实**：本文关于对象、字段、调用顺序和隐藏信息的陈述，来自当前官方 C++ source 快照。
- **架构解释**：本文把这套实现称为“有限 DSL / execution IR + continuation state machine”。这是对源码结构的工程概括，不是官方给出的产品术语。

本项目没有修改 `engine/source/`。官方 README 还明确说明该包仅限竞赛使用，不是开源软件；不要把源码或卡牌内容发布到竞赛许可范围之外。

## 1. 一句话结论

engine 不是在运行时理解自然语言卡文，也不是为每张卡写一个 `if (card_id == ...)` 的执行函数。启动时，`CardImpl()` 用 `Chain` fluent builder 把每张卡编译进三张全局 master table：

```text
CardTable<CardId, CardMaster>
  ├─ ability/play/delay ──> SkillTable<SkillId, Skill>
  │                           └─ Effect[]
  └─ attacks[] ───────────> AttackTable<AttackId, Attack>
                              ├─ cost + base damage
                              ├─ preEffects[]
                              └─ postEffects[]

Effect = operation + target query + condition + selection contract + values
```

运行时由统一解释器遍历这些 `Effect`，用 `TargetList` 找目标、用 `SatisfyCondition` 判条件、需要玩家决定时生成 `SelectOption` 并暂停，选择返回后继续执行。状态变化集中落到 `CardMove`、`SetProperty` 和 effect handlers。

因此准确答案是：

- **大量常见语义是参数化的**：如“目标数量 × 50”“抽 N”“治疗 N”“移动到某区域”“下回合不能攻击”。
- **不是纯数学表达式树**：`EffectType` 对应 C++ `switch` 分支，字段含义依赖 effect type。
- **不是完全通用 DSL**：枚举中存在具体卡族或特例词汇，如 `MysteryGarden`、`LoveBall`、`KoffingOrWeezing`；主阶段也有个别 card-ID 特判。
- **CardImpl 基本是声明数据，不持有逐卡 callback**：特例通常通过新增 enum/builder helper/解释器分支实现，而不是把 lambda 塞进 card prototype。

## 2. 目录结构地图

官方实现是 C++20、几乎 header-only；`Export.cpp` 是唯一主要 translation unit，`All.h` 聚合依赖。

| 层 | 文件 | 职责 |
|---|---|---|
| 构建/入口 | `Export.cpp`, `All.h`, `Framework.h`, `Core.h` | C ABI、总 include、标准库、常量与基础 helper |
| prototype 类型 | `Types.h`, `Skill.h`, `Card.h` | effect/target/condition 枚举；CardMaster/Skill/Attack/Effect；Card instance |
| prototype 构建 | `CreateCard.h`, `CardImpl.h`, `InitializeCard.h` | fluent builder、全部卡牌声明、Stage 2 链补全与 attack 后处理 |
| 核心状态 | `Game.h`, `PlayerState.h`, `State.h`, `ActivateInfo.h` | 配置/RNG/scratch；双方 zone；全局真值；当前 effect/trigger context |
| continuation | `GameFunction.h`, `BattleData.h`, `FixedList.h` | 可序列化函数栈；Game+State owner；固定容量容器 |
| 对局流程 | `SetupProc.h`, `GameProc.h`, `SelectProc.h` | 开局、回合、主行动、攻击、checkup、各类选择恢复函数 |
| effect 执行 | `EffectProc.h`, `EffectInstant.h`, `EffectContinual.h` | effect 调度、瞬时操作、持续属性重算、trigger/KO/refresh |
| 查询与变更 | `TargetList.h`, `SatisfyCondition.h`, `CardMove.h`, `SetProperty.h`, `GameUtil.h` | 目标查询、条件、zone 移动、伤害/状态、能量与伤害计算 |
| trigger/输出 | `PullTrigger.h`, `AddOption.h`, `AddLog.h` | 收集触发能力、构造合法 option、记录结构化日志 |
| API/序列化 | `Api.h`, `ApiData.h`, `ApiType.h`, `ApiJson.h`, `ToJson.h`, `JsonBuilder.h`, `Binary.h` | battle/search API、observation JSON、隐藏信息裁剪、state blob |
| 分支搜索 | `Search.h` | 复制已裁剪 state，为不同选择建立后继 state |
| 工程元数据 | `game.sln`, `game.vcxproj*`, `REUSE.toml`, `LICENSES/` | VS2022 构建与许可 |

依赖方向大致为：

```text
Types -> Skill -> Card -> PlayerState -> State
                                    -> query/mutation helpers
                                    -> EffectProc -> GameProc -> SetupProc
Export -> All -> BattleData + CardImpl + InitializeCard + Api
```

头文件互相 include 较深，不能把它误读为严格的模块边界；上表是职责分层。

## 3. 谁表示“一场游戏”

### 3.1 `BattleData` 是 owner

`BattleData` 同时持有一个 `Game game` 和一个 `State state`，并令 `state.game = &game`。`ApiData` 继承它，再加 JSON buffer、binary reader/writer、search memory 和可视化记录。

```text
ApiData : BattleData
  ├─ Game game
  │   ├─ GameConfig: decks, seed, time limit, logging flags
  │   ├─ RNG, remaining time
  │   └─ reusable scratch vectors (energyList, targetList, ...)
  ├─ State state
  │   ├─ authoritative match state
  │   └─ Game* game
  ├─ Search search
  └─ JSON / binary / visualization buffers
```

`Game` 并不是全部游戏状态。它主要保存配置、随机数、计时和临时工作区；真正会被复制、序列化、搜索分支化的事实在 `State`。

证据：`BattleData.h:11-30`、`ApiData.h:12-24`、`Game.h:32-84`。

### 3.2 `State` 是权威可推进快照

`State` 同时包含：

- 全局时序：`turn`、`phase`、`gameResult`、`firstPlayer`；
- 每回合预算：`supporterPlayed`、`stadiumPlayed`、`energyPlayed`、`retreated`、`turnEnd`；
- 当前 interaction：`selectType/context/player/min/max/options/selected`；
- 当前 effect/attack：`effectState`、`triggerInfo`、`currentAttackId`、`attacker`、damage accumulator；
- 实体池：固定 `allCard[128]`；
- 双方 zone：`players[2]`；
- 临时目标、KO、trigger、历史与日志列表；
- `functionStack`：下一步要执行的 continuation。

`State::serialize()` 把固定字段区和变长列表都写入 binary；`Search` 也直接复制 `State`。所以它不仅是 board snapshot，还是“执行到一半”的完整 continuation snapshot。

证据：`State.h:110-236`、`State.h:243-279`。

### 3.3 `PlayerState` 用 zone list 表示每位玩家

每个玩家持有 `active`、`bench`、`prize`、`hand`、`deck`、`trash`，以及独立的 attachment storage：`energy`、`tool`、`preEvolution`。特殊状态在 player 上，因为它们属于当前 Active；换位/进化流程会相应清理或迁移。

值得注意的是：zone list 里不是完整 `Card`，而是 `CardRef`。真实实例统一在 `State::allCard` 中。这避免同一张实体卡在移动时复制自身状态。

证据：`PlayerState.h:32-60`、`State.h:208-218`。

## 4. 一张卡的两层表示

### 4.1 `CardMaster`：不随对局变化的原型

`CardMaster` 保存：

- identity：`cardId`、编号、日英名称；
- card class：Pokémon/Item/Tool/Supporter/Stadium/Energy；
- Pokémon stats：stage、HP、类型、弱点、抗性、撤退费用；
- tags/flags：ex、Mega ex、Tera、Ancient、Future、人物/Team Rocket 等；
- 行为引用：`ability`、`play`、`delay`、`attacks[]`；
- 进化来源。

所有 master object 存在全局 `CardTable/SkillTable/AttackTable`，由 `GameInitialize -> InitializeAll -> CardImpl` 构建一次。

证据：`Card.h:41-160`、`All.h:14-20`、`Export.cpp:43-45`。

### 4.2 `Card`：对局中的物理实例

`Card` 保存某张实体卡的：

- `cardId` 与 `CardMaster` 的关联；
- `playerIndex`、`area/preArea`、朝向；
- `moveCounter`：该实体最近一次进入区域/在场身份的 serial；
- `attachMoveCounter`：Energy/Tool/PreEvolution 绑定到哪个 Pokémon instance；
- damage、once-per-turn ability usage；
- this-turn、next-turn、next-enemy-turn、continual modifiers；
- KO、进场、进化、攻击伤害历史等 flags。

`CardRef` 只有一个 `allCard` 索引。`AreaRef = CardRef + moveCounter`，用来防止“同一物理卡离场再回来后，旧 effect 仍错误指向新实例”。这是非常关键的 identity 设计。

证据：`Card.h:11-39`、`Card.h:192-399`。

### 4.3 附着与进化不是嵌套 object

Energy、Tool、PreEvolution 仍放在各自 player list 中；它们用 `attachMoveCounter == target.moveCounter` 表示归属。Pokémon 进化时，新卡继承原 Pokémon 的 `moveCounter`，旧卡进入 `PreEvolution` 并以该 serial 绑定，因此伤害、Energy 和 Tool 可保留。

证据：`CardMove.h:135-170`、`CardMove.h:310-359`、`CardMove.h:413-448`。

## 5. Card DSL：自然语言怎样变成执行语义

### 5.1 builder 只在初始化时运行

`CreateCard(id, name, type, no)` 建立 `CardMaster`；链式调用再建立 `Skill`、`Attack`、`Effect`。文本被保存用于展示，不参与规则解释。

```cpp
CreateCard(861, ..., Pokemon, 1233)
  .nameEn(u8"Mega Froslass ex")
  .pokemon(MegaEx, Stage1, Water, 310, 1)
  .evolvesFrom(...)
  .attack(1240, ..., "50x", { Water })
  .preEffect(AttackDamageChangeTargetCount, Enemy)
    .eVal(50)
    .targetHand();
```

证据：`CreateCard.h:182-203`、`CreateCard.h:363-404`、`CreateCard.h:623-719`、`CreateCard.h:919-1083`、`CardImpl.h:10530-10540`。

### 5.2 `Effect` 是 typed instruction record

一个 `Effect` 不是独立 class hierarchy，而是一条带大量 tagged fields 的记录：

```text
effectType              operation opcode
values[2]               opcode-specific numeric operands
target                  player + areas + TargetCondition[]
conditionType           precondition opcode
effectSelectType        all / exact N / up to N / energy / evolve...
selectContext           UI/action semantic context
flags                   random, enemy selects, per-target loop, separator...
skill / attack pointer  owning prototype
```

`EffectType::ContinualEffectSeparator` 还把 effect opcode 分成持续效果与瞬时效果；`Effect::isInstantEffect()` 直接按枚举顺序判断。

证据：`Skill.h:14-94`、`Types.h:12-292`。

### 5.3 用户例子：对手手牌数量乘 50

Mega Froslass ex 的 `Resentful Refrain` 是最干净的 proof chain：

```text
CardImpl
  operation = AttackDamageChangeTargetCount
  target    = Enemy.Hand
  value     = 50
        |
TargetList
  enumerate enemy hand => targetList.size = H
        |
EffectInstant
  attackDamageChange = 50 * H
        |
GameProc::AttackDamage
  baseDamage = attack.damage + attackDamageChange
  damage = CalcDamage(... weakness/resistance/modifiers/protection ...)
        |
SetProperty::AddDamage
  mutate target Card.damage, mark KO when applicable
```

Whimsicott ex 的类似攻击再加一个 `TargetType::Trainer` filter，所以是“对手手牌里的 Trainer 数量 × 50”。这说明 area query 和 card predicate 是可组合的。

证据：`CardImpl.h:6772-6785`、`CardImpl.h:10530-10540`、`EffectInstant.h:1353-1363`、`GameProc.h:334-393`。

### 5.4 Attack、Ability、Trainer、Stadium 如何统一

- `Attack`：energy cost + base damage + `preEffects[]` + `postEffects[]`。pre effects 可改伤害、掷币或令攻击失败；统一伤害后再跑 post effects。
- `Ability`：一种 `Skill`，带合法区域、main/once-per-turn、trigger 与 effects。
- Item/Supporter：`CardMaster::play` 指向 `SkillType::Play`；主阶段合法性额外检查 card class 与每回合预算。
- Tool/Special Energy：本身作为附着 card instance 存在，其 `play/attach` skill 和 continual effects 可在 refresh 时生效。
- Stadium：唯一全局 `State::stadium` zone；其 ability/continual effects也走 Skill/Effect 系统。
- 延迟效果：`CardMaster::delay` 指向一个带 trigger 的 Skill，并在 `delayTriggerStack` 中保留未来触发上下文。

统一点是 `Skill/Effect`；差异点主要在合法行动生成、有效区域、trigger timing 与 zone lifecycle。

## 6. 状态机不是一个大 `while(turn)`

### 6.1 `functionStack` 是显式 continuation

每个流程函数不会同步执行到底，而是把“之后做什么”压入 `functionStack`。`State::step()` 反复调用栈顶，直到：

- 遇到 `selectType != None`，此时把 option 暴露给调用者并返回；或
- 游戏结束。

`GameFunction` 不序列化裸函数指针，而是存 `functionIndex + 最多三个参数 + callCount/calledCount`。启动时 `InitializeBattleFunction()` 以固定顺序注册所有 continuation。这样 state 可以在 selection boundary 被序列化，然后恢复到同一流程位置。

证据：`GameFunction.h:13-49`、`State.h:1650-1752`、`BattleData.h:85+`。

### 6.2 外部 action 只是 option index

engine 先用 `AddOption*` 构造 typed option；JSON 对外发出 `type` 与 `param` 的解释字段。agent 返回的是 option indices。`ApiSelect` 校验 index、数量、重复等条件，然后调用 `next()` 恢复栈。

这就是 action contract 的本质：agent 不直接提交任意 `attackId + target` mutation，而是在 engine 当前 selection node 上选择合法边。

证据：`ApiType.h:9-163`、`ApiJson.h:11+`、`Api.h:122-140`、`State.h:1755+`。

### 6.3 回合主流程

```text
SetupGame
  shuffle -> choose first -> draw 7 -> mulligan -> Active -> Bench -> Prize
      |
TurnStart
  advance turn/history -> rotate timed state -> deck-out check -> draw 1
      |
MainSelect <--------------------------------------------------+
  build all legal Play/Attach/Evolve/Ability/Retreat/Attack/End |
      | ordinary main action -> resolve -> Refresh -> ToMain --+
      | Attack
      v
  preEffects -> damage -> postEffects -> trigger/KO/Prize -> TurnEnd
      |
PokemonCheckup
  status + triggers -> Refresh -> next TurnStart
```

官方规则的“攻击是回合终止提交”在 engine 中体现为 attack resolution 直接走 `AfterAttack -> TurnEnd`，不会再压回 `ToMain`。

证据：`SetupProc.h:226-251`、`GameProc.h:680-713`、`GameProc.h:749-996`、`GameProc.h:315-431`、`GameProc.h:16-119`。

## 7. 一条 action 的完整转换链

以主阶段攻击为例：

1. `MainSelect` 读取 `State + master tables`，过滤能量、状态、首回合和 attack condition，生成 `Attack` option。
2. `SelectedMain` 读取 agent 选择，转入 `SelectedAttack`，设置 `currentAttackId/srcAttackId/attacker`。
3. `SelectedAttack3` 依次安排 pre-effects、`AttackDamage`、post-effects。
4. 每个 effect 经 `ActivateEffectLoop -> ActivateEffect`：检查 condition、枚举 target、必要时产生嵌套 selection。
5. `ActivateEffect2` 分流：continual effect 进入属性应用，instant effect 进入 `EffectInstant` 大 switch。
6. zone 改变走 `MoveCard`，伤害/特殊状态走 `SetProperty`，都更新 serial、日志和 trigger context。
7. `Refresh` 重算所有持续效果，处理 bench/tool 容量、trigger stack、KO、新 Active、Prize 与胜利条件。
8. 下一处 selection boundary 由 API 输出成新 observation。

这条链解释了为何一次“动作”可能需要多次 agent 回调：先选牌，再选目标，再决定 Yes/No，再选择弃哪张 Energy。每次都是同一 continuation state 的下一节点。

## 8. 瞬时状态与持续状态如何维护

### 8.1 zone mutation 是集中式的

`MoveCard` 负责从源 list 取出 `CardRef`、推入目标 list、更新 `area/preArea/moveCounter/reverse`、处理 attachment、写日志和触发移动语义。`EvolveProc`、`AttachProc`、`SwitchPokemon` 建立在它之上。

这种设计比散落的 `hand.erase(); bench.push()` 更可审计，因为 identity 与 attachment invariants 有统一入口。

### 8.2 持续效果采用“清空后重算”

`RefreshEffect` 会清空全局、玩家和在场卡的 `continualState`，收集当前有效的 continual skills，按 priority/skill order/move counter 排序，再重新应用 `EffectContinual`。

优点是 Stadium/Ability/Tool 离场后不必逐项撤销旧 modifier；下次 refresh 自然消失。代价是许多 derived fields 是缓存，不是独立事实源，阅读 snapshot 时要知道它们来自最近一次 refresh。

证据：`EffectProc.h:157-235`、`EffectContinual.h:23-504`。

### 8.3 定时状态采用分层字段轮转

Card 与 Player 都有 `thisTurn/nextTurn`；回合开始把 next 搬到 this，回合结束清除对应窗口。另有 `nextEnemyTurnEndState`、turn-local flags 和 history ring，分别承载不同卡文时限。

这不是通用 duration object，而是为常见 TCG 时间窗口设计的 packed state。它很快、可序列化，但每出现新时间语义可能需要新增字段。

### 8.4 trigger 是收集后解析

`PullTrigger` 扫描 Active/Bench/Stadium/attached cards 等有效区域，将匹配 `TriggerType + Target` 的能力放入 temporary/real trigger stack。`Refresh` 再处理顺序、选择、KO 和后续 trigger。

延迟 trigger 独立存在 `delayTriggerStack`，例如“下个对手回合结束时”。

## 9. Hidden truth 与 observation 不是同一个对象

engine 内部 `State` 知道双方完整 deck、hand 和 Prize。对 agent 输出时有两道边界：

1. `ToJson` 按观察者过滤：对手 `hand` 是 null、Prize card 可为 null、deck 通常只给 count；公开区给 card ID/serial/HP/attachments。
2. `ApiGetBattleData` 复制 state 后调用 `erasePlayerData(selectPlayer)`，清掉搜索 blob 中不应直接知道的卡牌身份，再序列化。

之后 agent search API 必须显式提供自己对未知 zone 的假设 card IDs，`Search::start` 才把这些 identity 填回 clone 并分支推进。

所以不能因为 simulator 内部有全知真值，就把它当作 actor-visible feature。`State`、observation JSON、sanitized search state 是三个不同边界。

证据：`ToJson.h:12-188`、`Api.h:99-120`、`State.h:281-315`、`Search.h:89-163`。

## 10. 这套设计的真实取舍

### 做得好的地方

- prototype 与 instance 分离，数千张同类卡共享通用执行代码；
- `AreaRef` 防止 stale target 指向离场重进后的新实例；
- continuation stack 让任意深度选择可暂停、序列化和分支搜索；
- zone mutation、target query、condition 和 effect execution 分层；
- continual effect 重算避免复杂的增量撤销；
- engine 合法 option 是唯一 action contract，agent 无法任意篡改状态。

### 明显的工程代价

- `Effect` 是“大记录 + opcode-dependent fields”，类型安全弱；
- `EffectInstant.h` 的大 switch 与 `CardImpl.h` 的大表会持续增长；
- 某些 enum/helper 已经带具体卡牌或卡族名字，抽象并不纯；
- timed state 由大量 packed flags 表示，新语义常要求改 struct；
- include graph 深、实现集中在 header，编译边界和单元隔离较弱；
- JSON `AllCard/AllAttack` 只导出展示所需子集，不等于完整 execution IR。

这不是否定设计。对一个规则范围固定、要求确定性和高吞吐的竞赛 simulator，这些取舍很务实。

## 11. 阅读源码的推荐路线

第一次阅读按下面顺序，避免直接掉进 13K 行卡表和 2K 行 effect switch：

1. `README.md`, `Export.cpp`, `All.h`：知道入口和初始化。
2. `Skill.h`, `Card.h`, `State.h:110-236`：建立 prototype/instance/state 三层模型。
3. `BattleData.h`, `State.h:1650-1752`：理解 continuation executor。
4. `GameProc.h:680-996`：看 legal option 与 turn loop。
5. `EffectProc.h:352-819`：看 effect 的 target/select/loop 调度。
6. `EffectInstant.h:1353-1363` + `CardImpl.h:10530-10540`：闭环用户的 `hand x 50` 例子。
7. `CardMove.h`, `SetProperty.h`, `PullTrigger.h`：看 mutation 与 trigger invariants。
8. `ToJson.h`, `Api.h`, `Search.h`：最后看 actor boundary 与分支搜索。

## 12. 源码索引

- [`Export.cpp`](../../engine/source/ptcgProgram%2022/Export.cpp)：C ABI 与静态 JSON cache。
- [`All.h`](../../engine/source/ptcgProgram%2022/All.h)：初始化顺序。
- [`Card.h`](../../engine/source/ptcgProgram%2022/Card.h)：CardMaster、Card、CardRef、master tables。
- [`Skill.h`](../../engine/source/ptcgProgram%2022/Skill.h)：Target、Effect、Skill、Attack。
- [`Types.h`](../../engine/source/ptcgProgram%2022/Types.h)：规则词表。
- [`CreateCard.h`](../../engine/source/ptcgProgram%2022/CreateCard.h)：prototype builder。
- [`CardImpl.h`](../../engine/source/ptcgProgram%2022/CardImpl.h)：官方卡牌执行定义。
- [`State.h`](../../engine/source/ptcgProgram%2022/State.h)：权威状态与 continuation executor。
- [`PlayerState.h`](../../engine/source/ptcgProgram%2022/PlayerState.h)：玩家 zones 与状态。
- [`GameProc.h`](../../engine/source/ptcgProgram%2022/GameProc.h)：turn/main/attack/checkup。
- [`EffectProc.h`](../../engine/source/ptcgProgram%2022/EffectProc.h)：effect/trigger/refresh/KO 调度。
- [`EffectInstant.h`](../../engine/source/ptcgProgram%2022/EffectInstant.h)：瞬时 effect opcode 实现。
- [`EffectContinual.h`](../../engine/source/ptcgProgram%2022/EffectContinual.h)：持续 effect opcode 实现。
- [`TargetList.h`](../../engine/source/ptcgProgram%2022/TargetList.h)：area 与 target predicate 查询。
- [`SatisfyCondition.h`](../../engine/source/ptcgProgram%2022/SatisfyCondition.h)：condition 与合法性。
- [`CardMove.h`](../../engine/source/ptcgProgram%2022/CardMove.h)：zone/attachment/evolution mutation。
- [`SetProperty.h`](../../engine/source/ptcgProgram%2022/SetProperty.h)：伤害与特殊状态 mutation。
- [`Api.h`](../../engine/source/ptcgProgram%2022/Api.h)：battle/search API 与隐藏状态裁剪。
- [`ToJson.h`](../../engine/source/ptcgProgram%2022/ToJson.h)：actor-visible observation。
- [`Search.h`](../../engine/source/ptcgProgram%2022/Search.h)：state clone 与分支推进。

## 13. 最后回答用户原始判断

你的判断在核心方向上是对的：卡文不会在运行时被解析；“数量 × coefficient”“移动区域”“选择 N 张”“状态持续到下回合”等都已被拆成通用字段和执行原语。

需要修正的一点是：它不是一个可以任意组合所有规则的优雅公式语言。它更像为当前 card pool 手工演进的 bytecode：常见模式高度参数化，罕见模式通过增加 opcode、flag、target predicate 或显式特判纳入。因此研究它时，最有价值的不是只数 `EffectType`，而是同时看四层：

```text
prototype declaration
  -> target/condition query
  -> effect interpreter
  -> mutation + refresh + trigger
```

缺任何一层，都无法从 card declaration 准确推断最终运行时语义。

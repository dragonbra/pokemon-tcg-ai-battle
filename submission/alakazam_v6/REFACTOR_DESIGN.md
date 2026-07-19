# Alakazam V6 完整策略重构设计

> 状态：重构设计讨论中，尚未形成实现规格。
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

### 1.2 当前值得保留的方向

- **攻击线保护**：攻击线不足时，Poffin 的效果选择继续优先 Abra；达到三条线且
  ready 线数量够高后才转向 Dunsparce。
- **一条进化线通常只需要一张 Psychic Energy**：不再给已经有 Psychic 的
  Abra/Kadabra/Alakazam 无意义地重复贴能量。
- **自然进化与 Rare Candy 并存**：Active Kadabra 已有能量且手中有 Alakazam 时，
  允许直接自然进化；Active Abra 在合法的第二回合直通时保留 Rare Candy 路线。
- **Mist Energy 视作真实伤害阻断**：Alakazam 对带 Mist 的对手 Active 不应继续
  把攻击当成有效伤害，Enhanced Hammer 才可能成为高优先级动作。
- **Boss / Fezandipiti 有条件使用**：不把 Supporter 或两奖宝可梦当作无条件过牌。
- **每局清除 effect serial**：这是生命周期正确性修复，必须保留。

### 1.3 官方十场 replay 给出的事实

来源：`reports/kaggle/alakazam-v5-auto-iter-episodes-2026-07-19/analysis.md`。
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
- 当前 Active 能通过 Retreat、Trading Places 或被动换位交给它；
- 交接不会浪费关键能量或把两奖目标暴露给对手；
- 交接后仍有一只下一顺位的攻击线，或牌库/弃牌区有现实恢复路径。

V6 需要从 `ready_count` 升级为可解释的 **handoff path** 状态。

### 2.4 抽牌保护需要服从胜利节奏，而不是只看牌库张数

`_draw_is_blocked()` 当前主要看“抽牌是否直接创造击倒、手牌是否超过 20、牌库
是否 ≤ 10、当前是否已经能击倒”。这能防止一部分无意义过牌，但还没有回答：

- 当前抽牌是不是为了下一只打手的明确缺件？
- 当前已有两只 ready attacker，却还没有拿到任何 Prize 时，继续抽牌是否合理？
- 对手是 Land Collapse / wall / mirror 时，牌库预算是否应该不同？
- Dudunsparce 抽牌并回收自身的价值，是否高于保留一只攻击线？

v6 需要一个“牌库预算 + Prize 进度 + 下一只打手缺件”的联合 gate，而不是继续
只把 `DECK_DRAW_STOP` 从 10 调成别的数字。

## 3. 暂定的完整策略状态模型

以下不是代码接口，而是讨论时统一语言的状态向量：

```text
turn_context
├─ 立即攻击：Active 能否攻击？能否击倒？攻击是否只是低阶段 fallback？
├─ 主攻路线：Active/Bench 的 Abra → Kadabra → Alakazam 合法进化路径
├─ 交接路线：下一只攻击手是否有 Psychic、能否撤退/Trading Places/接班
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

## 5. 当前暂不拍板的设计问题

这些问题会在确认总目标后一次讨论一个：

1. **低阶段攻击门槛**：只有能击倒才攻击，还是允许用攻击换取手牌/场面节奏？
2. **进化路线**：第二回合直通、第三回合自然进化和 Bench 进化如何排序？
3. **接力定义**：哪些 Retreat / Trading Places / 能量附着算作可靠 handoff？
4. **Supporter 预算**：Dawn、Hilda、Fezandipiti、Xerosic、Boss 的价值如何与
   当前攻击轮次比较？
5. **能量预算**：什么时候允许给 Dudunsparce 贴 Enriching Energy 过牌，什么时候
   必须把能量留给下一只 Abra 线？
6. **恢复顺序**：没有场上 Abra 时，Lana's Aid / Night Stretcher 是否必须先补
   Abra，再考虑拿回 Kadabra / Alakazam？
7. **牌库预算**：对普通 Prize race、墙、mirror、Land Collapse 是否采用同一抽牌
   gate？
8. **匹配对手信息**：只使用 observation 中可观察的事实，还是为已知 opponent
   archetype 保留有限的 matchup policy？

## 6. 讨论纪律与验收标准

- 每个新规则先写成“触发条件 → 期望收益 → 可能副作用”，再决定是否实现。
- 不以单场 replay 证明策略正确；至少要能解释它改善哪一类失败，并设计可比较的
  replay 指标。
- 指标至少包括：首只 Alakazam、首只 Alakazam 攻击、有效攻击次数、每回合是否有
  handoff path、首次 Prize 回合、连续 Prize 间隔、终局牌库、剩余 Prize 和
  deck-out 原因。
- 评测时尽量固定 deck 顺序，或做同一对手/先后手的 paired replay，避免把抽牌顺序
  与策略代码混为一谈。
- 在设计确认前，`submission/alakazam_v6/` 只保存讨论材料；设计获批后才复制必要
  的策略运行文件，且 `deck.csv` 必须与 v5_auto_iter 保持一致。

## 7. 参考材料

- 基线实现：`submission/alakazam_v5_auto_iter/main.py`
- 基线说明：`submission/alakazam_v5_auto_iter/STRATEGY.md`
- auto_iter 迭代记录：`submission/alakazam_v5_auto_iter/CHANGELOG.md`
- 最新官方 replay 复盘：`reports/kaggle/alakazam-v5-auto-iter-episodes-2026-07-19/analysis.md`
- 流程可视化：`submission/alakazam_v5_auto_iter/strategy-flow.html`

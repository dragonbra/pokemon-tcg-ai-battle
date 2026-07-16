# 两个 Alakazam Notebook 的联合研究

研究对象：

- https://www.kaggle.com/code/heiseimikiko/why-alakazam-is-a-good-baseline-for-ai
- https://www.kaggle.com/code/ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th

研究日期：2026-07-17

## 一、先说结论

这两个 Notebook 共同表达的不是“Alakazam 是最强卡组”，而是：

> Alakazam 是一个非常适合把卡牌游戏转化为规则决策问题的 archetype。

它的强项在于策略结构清楚：

1. 建立 Abra → Kadabra → Alakazam 进化线；
2. 持续准备 Psychic Energy；
3. 让下一只 Alakazam 在当前攻击手被击倒前准备好；
4. 通过手牌数量决定 Powerful Hand 的伤害；
5. 按 Prize race 选择攻击目标；
6. 用抽牌、检索和恢复卡维持连续性。

这恰好把一个复杂的 TCG agent 拆成了若干可以明确编码、测试和逐步增强的子问题。

## 二、Heisei 的 Notebook：为什么 Alakazam 适合作为 baseline

来源：
https://www.kaggle.com/code/heiseimikiko/why-alakazam-is-a-good-baseline-for-ai

作者介绍说，他曾经用 Alakazam 达到过当时 leaderboard 的前三名，然后总结这个卡组为什么适合规则型 AI。

### 1. 自己的回合有清晰、可重复的计划

作者概括的核心计划是：

- 完成 Alakazam 进化线；
- 贴 Psychic Energy；
- 保持后续攻击手；
- 把手牌数量转化为伤害。

也就是说，agent 初期不需要理解对手所有可能的 combo，只要先把自己的回合做连贯。

这是一个非常重要的 baseline 特征：

- 状态目标明确；
- action priority 可以解释；
- 失败原因容易记录；
- 可以从简单规则逐渐扩展到概率和搜索。

### 2. 基础 game plan 是“连续输出”

作者列出的基本计划包括：

- Abra 进化成 Kadabra，再进化成 Alakazam；
- 保证每回合有足够 Psychic Energy 攻击；
- 当前 Alakazam 被 Knock Out 前，准备下一只；
- 使用抽牌和检索保持足够大的手牌；
- 不要过早或过晚消耗 recovery 卡。

作者认为最重要的不是某一个回合打出最大伤害，而是保持连续攻击，避免后期没有后续攻击手。

这对应我们的状态特征：

```text
当前攻击手是否可攻击
下一只攻击手还需要几回合准备
手牌数量
本回合可增加多少手牌
剩余 Psychic Energy
牌库中可用攻击手数量
剩余 Prize
对手各 Pokémon 的 Prize value
```

### 3. 这是规则型 agent 的天然形状

Alakazam 的决策可以写成明显的层次结构：

```text
先保证生存和连续性
    ↓
准备进化线
    ↓
准备能量
    ↓
增加手牌/保留关键资源
    ↓
计算 Powerful Hand 伤害
    ↓
选择能够带来最大 Prize 收益的目标
```

这比一个依赖大量条件联动的 combo deck 更容易先写出能运行的 baseline。

### 4. 早期强，不等于最终上限最高

Heisei 的文章主要回答“为什么适合 baseline”，并没有证明 Alakazam 的卡组上限一定最高。

它更准确的贡献是：

> Alakazam 是一个能快速暴露 agent 基本能力的测试平台。

如果 agent 连 Alakazam 的连续攻击、手牌伤害和 Prize 目标都处理不好，直接训练更复杂的 deck 也很难解释结果。

## 三、Ryota 的 Notebook：一个具体的规则型实现

来源：
https://www.kaggle.com/code/ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th

Notebook 标题是“Rule-based, not psychic: Alakazam”，作者说明：

- 这是比赛第二天的 agent；
- 当时 Alakazam 卡组位于 leaderboard 第 5；
- Notebook 打开了那个 agent；
- 该方法复制自另一个公开的 Crustle bot 工作；
- 页面显示的最佳分数为 726.3（V2）；
- Notebook 采用 Apache 2.0 license。

这里必须注意时间因素：这是比赛早期的历史成绩，不应和当前分数直接比较。

### 1. 第一回合的 setup 规则

作者的第一回合优先级非常具体：

1. 先把一只 Abra 和一只 Dunsparce 放到 Bench；
2. 再放两只 Abra；
3. 如果可以，放置 Abra 时附上 Telepath Psychic Energy；
4. 如果不能直接放置，则用 Buddy-Buddy Poffin；
5. 如果还有 Buddy-Buddy Poffin，再放一只 Dunsparce；
6. 如果有 Poke Pad，就检索 Dudunsparce 和 Kadabra；
7. 至少保留一个 Bench 空位。

这套规则揭示了一个关键点：

> Alakazam baseline 的核心不是“见到 Alakazam 就进化”，而是同时铺设攻击手、抽牌引擎和后备资源。

Dunsparce/Dudunsparce 在这里承担的是 hand engine 或 setup engine 的角色；Abra/Kadabra/Alakazam 承担攻击线。

### 2. 第二回合及以后：小伤害先用 Kadabra 收割

作者给出了一个明确阈值：

- 如果对手 Active 剩余 HP ≤ 30，使用 Kadabra 的 Super Psy Bolt；
- 如果剩余 HP > 30，围绕 Alakazam 的 Powerful Hand 计算最大和最小伤害。

这说明 agent 不是简单地永远用 Alakazam，而是在做：

```text
低血量目标 → 用低成本的 Kadabra 收割
正常目标 → 计算 Alakazam 的 hand-size damage
```

这对 baseline 设计很有价值，因为它体现了“动作成本”和“资源保留”，而不是只看当前最高伤害。

### 3. Powerful Hand 的核心计算

作者把 Powerful Hand 的伤害抽象为：

```text
手牌数量 × 20
```

然后实际决策不是只看当前手牌，而是估计：

```text
本回合最多还能把手牌增加多少
```

因此 action selection 需要考虑：

- 先抽牌还是先攻击；
- 先使用检索还是先使用会减少手牌的卡；
- 哪些卡应当留在手里作为伤害；
- 能否在本回合达到击倒阈值；
- 为了多拿 Prize 是否值得使用 Boss's Orders。

这是一个非常适合从规则 baseline 逐步升级到搜索/网络的局面价值问题。

### 4. Prize race 和目标选择

作者的目标选择逻辑大致是：

1. 如果击倒 Active 后可以拿完剩余 Prize，就优先击倒 Active；
2. 如果当前 Active 即使达到最大伤害也打不掉，就使用 Boss's Orders 把 Bench Pokémon 拉到 Active；
3. 选择能够被当前伤害范围击倒、且能提供最多 Prize 的目标；
4. 如果多个目标 Prize value 相同，则优先选择剩余 HP 更高、但仍在最大伤害范围内的目标。

这里已经不是“攻击血量最低的 Pokémon”，而是一个简化的战术目标函数：

```text
优先级 = 是否完成胜利
       > Prize value
       > 是否在伤害范围内
       > 目标 HP/后续交换价值
```

这就是一个非常适合写成 action scorer 的 baseline。

### 5. 作者自己承认规则还不完整

Ryota 明确说这些 playing principles 没有被准确写进代码，也不是最优方法，还有很多可以改进的地方。

这点很重要：

- 他的历史第 5 名说明“简单规则 + 合适卡组”可以很有效；
- 但它不是一个已经完成的 expert policy；
- 我们可以把它看成一个可读的 policy skeleton，而不是最终方案。

## 四、两个 Notebook 合在一起揭示的真正结构

### 1. Alakazam 不是“简单”，而是“复杂度集中在少数明确变量上”

它并不是没有难点。难点集中在：

- 进化节奏；
- Psychic Energy 的连续供应；
- 手牌数量与 Powerful Hand 伤害；
- 下一只攻击手的准备；
- Prize race 目标选择；
- Boss's Orders 的时机；
- recovery 卡的使用时机；
- Bench 空间。

相比之下，很多复杂 combo deck 的难点是大量卡牌之间的状态依赖，难以快速定义稳定规则。

### 2. 它非常适合作为“分层 agent”

可以拆成四层：

```text
层 1：合法动作和状态解析
层 2：setup planner
     Abra/Kadabra/Alakazam、Dunsparce/Dudunsparce、Energy
层 3：damage/prize planner
     Powerful Hand、Super Psy Bolt、Boss's Orders
层 4：资源连续性
     下一只攻击手、recovery、Bench 空间、牌库/手牌管理
```

这四层可以逐一测试，不需要一开始写一个不可解释的 monolithic policy。

### 3. 它天然适合 imitation learning

Ryota 的规则可以直接产生训练标签：

```text
输入：当前 observation + 合法 actions
标签：规则 policy 选择的 action
```

之后可以训练一个 policy model 去模仿规则 agent，再比较：

- 神经网络是否能压缩规则；
- 网络是否在未见过的局面上泛化；
- 网络是否比规则 agent 更快或更稳定；
- 网络是否能作为 bounded search 的 action prior。

## 五、对我们 baseline 的具体启发

第一版不应该只实现“看到能攻击就攻击”。建议写成以下优先级：

### Setup 阶段

```text
1. 保证至少一只基础攻击线
2. 保证后续攻击手
3. 保证 Dunsparce/Dudunsparce 抽牌引擎
4. 保证 Psychic Energy 计划
5. 保留 Bench 空间
```

### 每回合决策阶段

```text
1. 先判断本回合是否可以直接赢
2. 判断是否可以用 Kadabra 低成本收割
3. 计算 Powerful Hand 最大/最小伤害
4. 判断是否值得先抽牌/检索/增加手牌
5. 判断是否需要 Boss's Orders 改变目标
6. 攻击后检查下一只 Alakazam 是否已在准备
7. 处理 recovery 与 Bench 空间
```

### 第一版状态特征

```text
自己的：
- 手牌数量和关键卡数量
- Active/Bench 上的 Abra、Kadabra、Alakazam
- Dunsparce/Dudunsparce 数量
- Psychic Energy 数量和附着位置
- 当前攻击手是否可攻击
- 下一只攻击手预计几回合完成
- 牌库数量、discard 数量
- Bench 剩余空间
- 剩余 Prize

对手公开信息：
- Active/Bench Pokémon
- 当前 HP
- Prize value
- 是否在 Powerful Hand 伤害范围内
- 是否需要 Boss's Orders
```

## 六、对“Alakazam 为什么高频”的重新判断

这两个 Notebook 让我们可以把之前的结论说得更精确：

Alakazam 在高分区的高频率，可能来自三个层面：

1. 卡组结构本身提供了稳定的 setup 和攻击循环；
2. 它把关键决策压缩成容易写规则的变量；
3. 公开规则 agent 容易复制和改进，形成 baseline 扩散。

所以它可能不是“现实环境里最强的卡组”，但它非常可能是：

> 这个 simulator 中最适合快速写出稳定 agent 的卡组之一。

这也解释了你观察到的现象：现实牌局里它未必是最主流，但 Kaggle 高分榜里它很集中。Kaggle 评估的不是纯卡组强度，而是“卡组与 agent 的组合”。

## 七、限制与待验证点

目前两个 Notebook 的正文可以读取到策略说明，但公开页面没有直接给出我们可以完整复现的精确 60 张 deck list。还需要在本地拿到比赛数据或公开 replay 后确认：

- Alakazam 的完整卡组构成；
- Telepath Psychic Energy 的准确 card ID 和行为；
- Dunsparce/Dudunsparce 的具体 hand engine；
- Powerful Hand 是否在 simulator 中严格按手牌数 × 20；
- Boss's Orders 和 Prize value 的具体 action 表达；
- Ryota 的规则是否已经被后续版本修改。

## 八、最终建议

把这两个 Notebook 当成我们第一版 baseline 的设计文档，而不是直接复制代码。

第一版目标：

```text
Alakazam 规则 agent
= setup planner
+ hand-size damage calculator
+ Prize target scorer
+ continuity/recovery manager
```

然后按顺序做：

1. 手工实现规则；
2. 与随机 agent 对战；
3. 与 Mega Lucario/Archaludon/Crustle 规则 agent 对战；
4. 记录每一步规则命中和失败原因；
5. 再加入 1–3 层有限搜索；
6. 最后训练 policy/value model。

这两个 Notebook 对我们最有价值的地方，不是给出一个神秘的强卡组，而是提供了一个可以逐条转成代码和实验的决策分解。

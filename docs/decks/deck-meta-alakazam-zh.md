# 排行榜卡组元分析与 Alakazam 研究

研究对象：
https://www.kaggle.com/code/myso1987/ptcg-ai-battle-leaderboard-deck-meta-by-score-band

研究日期：2026-07-17

## 一、这个 Notebook 到底做了什么

它确实是在分析排行榜上的主流卡组，但不是用完整 60 张卡组直接估计每张卡的强度。

它的流程是：

1. 从公开 leaderboard 获取队伍和分数；
2. 每个队伍只选择一个 active submission；
3. 选择分数最接近当前队伍 leaderboard 分数的公开 submission；
4. 从该 submission 的公开 completed episode replay 恢复一套 60 张牌；
5. 用卡组中是否出现某些 marker card，按从上到下的规则给卡组分类；
6. 按分数段统计 archetype 的队伍数和占比。

这个 Notebook 的公开输出只包含聚合表、图表和 coverage 信息。它明确不发布：

- 队伍名称和 ID；
- submission ID；
- episode ID；
- 原始 replay；
- 完整 60 张 deck list；
- card-ID 序列。

因此它可以回答“哪些 archetype 在不同分数段出现得多”，但不能单独回答“Alakazam 的标准 60 张具体是什么”。

Notebook 还明确说：这是 metagame snapshot，不是卡组强度、胜率或因果表现的测量；archetype 是按 marker-card 规则分类，混合卡组可能被归类或误分类。

## 二、当前各分数段的结果

Notebook 在 2026-07-16 的快照中，覆盖了 595 支 leaderboard 队伍，恢复 590 套卡组，覆盖率 99.2%。各 band 的主要结果如下：

| 分数段 | 主要 archetype 与占比 |
|---|---|
| 1100+，18 队 | Alakazam 5，27.8%；Crustle Wall 4，22.2%；Marnie Grimmsnarl 4，22.2%；Team Rocket Mewtwo 3，16.7% |
| 1000–1099，77 队 | Alakazam 34，44.2%；Crustle Wall 14，18.2%；Marnie Grimmsnarl 7，9.1%；Mega Lucario 7，9.1% |
| 900–999，99 队 | Alakazam 43，43.4%；Marnie Grimmsnarl 16，16.2%；Crustle Wall 5，5.0%；Dragapult 5，5.0%；Mega Lucario 5，5.0% |
| 800–899，99 队 | Alakazam 29，29.3%；Archaludon 28，28.3%；Cynthia Garchomp 7，7.1%；Dragapult 6，6.1% |
| 700–799，99 队 | Mega Lucario 26，26.3%；Archaludon 20，20.2%；Alakazam 18，18.2%；Dragapult 15，15.2% |
| 600–699，99 队 | Mega Lucario 28，28.3%；Alakazam 20，20.2%；Crustle Wall 14，14.1%；Mega Starmie 11，11.1% |
| 500–599，99 队 | Mega Lucario 26，26.3%；Crustle Wall 15，15.2%；Alakazam 13，13.1%；Mega Starmie 11，11.1% |

## 三、Alakazam 到底说明了什么

你的观察是对的：Alakazam 在高分区非常集中。

- 900–999：43.4%；
- 1000–1099：44.2%；
- 1100+：仍为第一名，但只有 5/18，即 27.8%；
- 600–699：20.2%；
- 500–599：13.1%。

这说明它不是只有一个作者偶然分享的单一 baseline，也不是只存在于一个极小的队伍样本中。至少从这个快照看，Alakazam 是一个在 900–1099 分段特别常见的 archetype。

但目前不能直接得出“Alakazam 卡组本身最强”的结论，原因有四个：

1. Notebook 统计的是队伍占比，不是 archetype 条件胜率。
2. 分数同时包含卡组、agent 策略、代码质量、对手池和提交年龄。
3. 一个公开 baseline 可能被大量复制，造成某个 archetype 的高频率。
4. Notebook 用 marker card 分类，Alakazam 可能吸收了多个不同的混合卡组。

更准确的表述是：

> Alakazam 是当前高分环境中强烈过度代表的卡组 archetype；这说明它与高分 agent 组合之间存在很强的相关性，但还不能把相关性分解成“卡组优势”与“公开 agent/baseline 扩散优势”。

尤其要注意 1100+ 只有 18 队，5 个 Alakazam 就能形成 27.8%。这个 band 的统计不确定性明显高于 900–999 和 1000–1099。

## 四、目前能可靠说出的“主要卡牌组成”

这里需要严格区分“已从 Notebook 公开验证的内容”和“我们根据 archetype 名称提出的研究骨架”。Notebook 的输出没有完整 60 张卡表，所以目前不能诚实地给出每个 archetype 的精确卡张数。

Notebook 的分类规则实际使用了以下 marker：

- Great Tusk / Crustle：同时有 Great Tusk 和 Crustle；
- Marnie Grimmsnarl：Marnie's Grimmsnarl ex；
- Cynthia Garchomp：Cynthia's Garchomp ex；
- Mega Lucario：Mega Lucario ex；
- Archaludon：Archaludon ex；
- Crustle Wall：Crustle；
- Dragapult：Dragapult ex；
- Mega Starmie：Mega Starmie ex；
- Starmie：Starmie ex 或 Starmie；
- Mega Gardevoir：Mega Gardevoir ex；
- Alakazam：Alakazam ex 或 Alakazam；
- Iono Bellibolt：Iono's Bellibolt ex；
- Festival Lead：Dipplin；
- Hop Trevenant：Hop's Trevenant；
- Hop Snorlax：Hop's Snorlax；
- Mega Kangaskhan：Mega Kangaskhan ex；
- Chandelure：Chandelure ex 或 Chandelure；
- Mega Greninja：Mega Greninja ex；
- Mega Clefable：Mega Clefable ex；
- Team Rocket Mewtwo：Team Rocket's Mewtwo ex。

因此当前可以建立的是“核心 Pokémon + 功能包”的研究骨架：

### Alakazam

已验证核心：Alakazam ex 或 Alakazam。

合理的功能包假设：

- Alakazam 主攻手/终结手；
- Psychic 系能量与能量加速/回收；
- 进化链稳定组件；
- 抽牌和牌库检索；
- 让主攻手持续攻击的替补和撤退组件；
- 针对 Prize race 的目标选择。

这里的功能包是研究假设，不是 Notebook 已公开的精确 deck list。需要通过原始 replay、公开 submission 或比赛数据进一步确认。

### Crustle Wall

已验证核心：Crustle；与 Great Tusk 同时出现时分类为 Great Tusk / Crustle。

从名字和高频位置看，它更像耐久/防守/拖节奏 archetype，而不是单纯追求最快击倒：

- 用高耐久或低 Prize 代价的 Pokémon 争取交换优势；
- 通过 Crustle 的防守和持续站场能力拖慢对手；
- 让对手在 Prize 交换、能量和攻击目标上付出代价；
- Great Tusk 版本可能承担额外的高血量或终结压力。

需要通过卡牌文本和 replay 确认 Crustle 的具体能力在 simulator 中如何体现。

### Marnie Grimmsnarl

已验证核心：Marnie's Grimmsnarl ex。

主要研究假设：

- Grimmsnarl ex 作为主力攻击和换奖点；
- Marnie 相关卡牌/效果承担卡组主题或稳定性；
- 通过进化、能量累积和中后期高压攻击赢得 Prize race；
- 可能更依赖稳定的 setup，而不是纯快攻。

### Mega Lucario

已验证核心：Mega Lucario ex。

主要研究假设：

- Fighting 系攻击和能量配置；
- 尽快完成主动位攻击循环；
- 利用攻击效率和对特定类型 Pokémon 的打击能力；
- 用低成本替补或检索组件维持节奏。

它在 500–799 分段都很常见，但在 900+ 的占比下降，可能表示它是易于实现、强 baseline 友好的卡组，也可能是高分区被更复杂 archetype 替代。不能只看出现频率判断强弱。

### Archaludon

已验证核心：Archaludon ex。

主要研究假设：

- 依靠稳定能量和中后期高伤害；
- 通过资源积累换取高质量攻击；
- 可能较适合规则 agent，因为主攻目标和能量计划相对明确；
- 800–899 分段出现 28/99，是这个分段最接近 Alakazam 的 archetype。

### Dragapult

已验证核心：Dragapult ex。

主要研究假设：

- 通过攻击直接压制 active 或同时施加后排压力；
- 重点是目标选择和 Prize race，而不是只计算 active 的单次伤害；
- 更适合搜索/价值模型研究，因为每次攻击的目标分配可能影响未来多回合局面。

### Cynthia Garchomp

已验证核心：Cynthia's Garchomp ex。

主要研究假设：

- 进化链和能量准备是前期核心；
- Garchomp 负责高质量主动位攻击；
- Cynthia 相关卡牌可能提供主题资源或稳定性；
- 适合研究 setup 风险和中后期爆发。

### Mega Starmie

已验证核心：Mega Starmie ex；另外 notebook 单独区分 Starmie archetype。

主要研究假设：

- 通过水系资源和灵活攻击争取 tempo；
- 可能偏向快速完成攻击循环；
- 在低分段的出现率较高，适合作为初期 baseline 对手。

## 五、这对规则 baseline 的意义

我们不应该先做一个“万能 Pokémon agent”，而应该先做一个能解释一个 archetype 的 agent。

我建议第一版按以下顺序：

1. 先实现 Alakazam 的卡组读取和局面日志；
2. 明确它的 setup 目标：起手、进化、能量、主动位和替补；
3. 对所有 `select` 做合法动作过滤；
4. 在每个动作点定义优先级：搜索/抽牌 > 建立主攻手 > 资源分配 > 攻击 > 撤退/换位；
5. 用 replay 记录每次决策和失败原因；
6. 再拿同一套 agent 换成 Mega Lucario、Crustle Wall 或 Archaludon 做对照。

这样可以分离三个问题：

- 卡组是否好；
- 规则 agent 是否会玩；
- 某个卡组是否只是因为有公开强 agent 才显得强。

## 六、模拟器是否支持“我自己和 AI 对战”

支持“自定义两个 agent 对战”，但官方文档给出的直接接口是 AI-vs-AI，不是现成的图形化人类操作模式。

官方 API 包括：

- `battle_start(deck0, deck1)`：用两套 60 张卡组开始 battle；
- `battle_select(select_list)`：推进一步并提交选择；
- `battle_finish()`：结束 battle；
- `visualize_data()`：输出人类可读的当前状态；
- `kaggle_environments.make("cabt", configuration={"decks": [deck0, deck1]})`：启动本地环境；
- `env.run([agent0, agent1])`：运行两个 agent；
- `env.render(mode="html")`：输出可视化结果。

所以可以做三种模式：

### 模式 A：AI vs AI

最简单，直接使用两个 agent：

```python
env.run([alakazam_agent, random_agent])
```

### 模式 B：你的手动 agent vs AI

可以写一个人工决策回调：

1. simulator 给出当前 observation 和合法 `select`；
2. 程序打印当前棋盘、手牌、选项；
3. 你输入要选择的 option index；
4. 程序调用 `battle_select`；
5. AI agent 在另一边自动选择。

这需要我们自己写一个 CLI UI，但 simulator API 从能力上支持这个闭环。

### 模式 C：半自动调试模式

让规则 agent 运行，但每一步暂停并打印：

- 当前 active/bench；
- 自己手牌；
- 牌库数量、discard；
- 对手公开状态；
- 当前合法动作；
- 规则 agent 的排序理由。

你可以手动覆盖某一步动作。这是我最推荐的第一版“自己玩一玩”方式。

## 七、当前结论

你关于 Alakazam 的直觉基本成立，但要精确表述：

- 它在 900–1099 分段明显是主流；
- 它在 1100+ 仍然是第一 archetype，但样本只有 18 队；
- 它在低分段也存在，说明不是某个单独高分作者的孤立结果；
- 但 notebook 没有提供 archetype 条件胜率和完整卡组，因此不能证明 Alakazam 本身是最强卡组；
- 它很可能同时受益于卡组结构、公开 baseline 扩散、agent 质量和对手池适配。

最重要的下一步不是猜测，而是做两个对照实验：

1. 同一套规则 agent，比较 Alakazam、Mega Lucario、Archaludon、Crustle Wall；
2. 同一套 Alakazam 卡组，比较随机、规则、概率化和搜索 agent。

这样才能把“卡组优势”和“agent 优势”拆开。

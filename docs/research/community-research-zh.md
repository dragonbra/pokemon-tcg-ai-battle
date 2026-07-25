# Pokémon TCG AI Battle：研究摘要（第一版）

研究日期：2026-07-17

本摘要基于比赛 Overview、Code 页面、Discussion 页面、官方 simulator 文档和可公开读取的 Notebook 页面。Discussion 与 Notebook 的作者推断视为线索，不视为已证实事实。

## 一、目前社区在讨论什么

### 1. Simulator 与正式规则的差异

这是最重要的基础问题。主办方已经单独置顶说明：某些攻击在 simulator 中可能直接不可选；Mega Zygarde ex 的部分处理采用自动顺序；双方同时 Knock Out 时 Prize 顺序不同；连续效果由 simulator 自动处理。

这意味着我们不能只按现实卡牌规则写 agent。正确的工程原则是：先读取 `select` 提供的合法动作，再根据 simulator 的状态转移行为建模。

来源：
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586

### 2. 社区主体仍然是规则型 agent

一篇基于 30,000 局顶尖队伍对局时间的分析认为，场上大量 agent 是手写规则或公开示例的变体；很多 agent 每步几乎立即决策。作者把启动时间较长、复杂局面思考时间明显增加的 agent 推测为神经网络或搜索型 agent。

这不是源代码证据，只是根据行为时间反推。但它告诉我们：简单规则 agent 仍然是主流，且不能把“使用了 RL”与“已经有强竞争力”画等号。

来源：
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/724362

### 3. 顶部可能出现“模型 + 有界搜索”的混合路线

同一篇 30,000 局分析推测，当前最靠前的某个 agent 可能加载了较重的模型，并在复杂局面使用更多思考时间，因此可能是 RL 加 bounded search 的混合系统。这个结论需要谨慎：它不是作者公开的实现说明，也不是我们可以直接复现的事实。

研究价值在于，它支持一个合理的长期路线：神经网络负责状态评估/动作先验，有限搜索负责短期战术选择，而不是一开始就试图用纯端到端网络解决所有决策。

### 4. RL、MCTS、概率化 Expectimax 和向量化模拟

当前公开 Code 页面中有几类明显路线：

- 官方/社区的 Reinforcement Learning and MCTS sample code；
- `Improved Probabilistic agent`，公开分数 967.7，描述为严格启发式的 Probabilistic Expectimax agent；
- `Custom Engine with Vectorized Env (2M sample/sec)`，重点是自定义高速向量化环境；
- 多个基于示例规则 agent、卡组更新和 meta 分析的 Notebook。

这四类工作解决的问题不同：

- 规则型/概率型：快速获得可靠 baseline；
- MCTS：在动作分支有限时进行局部前瞻；
- 神经网络/RL：学习复杂状态到策略或价值的映射；
- 向量化环境：提高训练样本吞吐，但会带来与官方 simulator 行为不一致的风险。

来源：
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/code

代表页面：

- https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code
- https://www.kaggle.com/code/aristophanivan/improved-probabilistic-agent
- https://www.kaggle.com/code/abiolatti/custom-engine-with-vectorized-env-2m-sample-sec
- https://www.kaggle.com/code/kaiwalyaatulraut/pok-mon-ai-battle-challenge-simulation-solution

## 二、关于三类 baseline 的判断

### A. 基于卡组的规则 agent

这是第一阶段必须做的 baseline。

它的价值不是只看分数，而是：

- 验证 `deck.csv` 和 submission layout；
- 熟悉 observation、`select`、OptionType 和状态转移；
- 建立卡牌优先级、进化、能量、撤退、Prize race 的可解释策略；
- 作为所有后续模型的对手和回归测试基线。

缺点是规则容易变成大量特殊情况，面对未知手牌和卡组时泛化有限。

### B. 搜索算法

搜索路线值得做，但不应作为最早的唯一目标。

优点：

- 对短期战术有天然解释性；
- 可以直接利用 simulator 的合法动作；
- 不依赖大量标注数据；
- 可以检验某种局面下的动作价值。

困难：

- 手牌、牌库和对手信息存在隐藏状态；
- 随机抽牌、硬币和对手策略导致树很快膨胀；
- 需要定义 rollout policy 和价值函数；
- 官方 simulator 每局有时间约束，线上包还有 CPU/RAM 限制。

因此更适合作为有限深度、动作剪枝或网络辅助的搜索，而不是无限制 MCTS。

### C. 神经网络

我同意把神经网络作为长期目标，但当前不建议直接从纯端到端 RL 起步。

原因：

- 首先要确认 simulator API 和动作语义；
- 需要可靠的 reward/value 定义；
- 自博弈早期可能只会学习到脆弱策略；
- 如果没有规则 baseline，很难判断网络学到的是策略还是环境 bug；
- 训练环境和官方 cabt simulator 如果存在细微差异，可能产生严重 sim-to-real 问题。

更现实的路线是：

1. 规则 baseline；
2. 记录状态、合法动作和结果；
3. 训练 action/value model；
4. 用网络做动作先验或局面评分；
5. 再接有限深度搜索；
6. 最后进行 self-play 和对手池评估。

## 三、对卡组研究的初步结论

当前公开页面已经说明参与者可以基于 Data 页的可用卡牌自己构建卡组。我们不应直接照搬现实世界 meta deck，而应先建立 simulator 卡池的结构化分析：

- 哪些卡可以稳定完成起手和搜索；
- 哪些卡提供抽牌/检索；
- 能量加速和撤退成本如何影响节奏；
- 主动位和后备位的 Prize race 价值；
- 卡牌组合是否因为 simulator 规则差异而改变优先级。

第一套卡组不必追求现实世界最强，而应追求：状态覆盖广、动作类型丰富、失败易解释、适合验证 agent。

## 四、建议的研究路线

### 阶段 1：可解释规则 baseline

目标：完成一套能稳定跑完比赛 simulator 的 agent。

重点：

- 合法动作过滤；
- 选择 active/bench；
- 基础进化和能量分配；
- 支持者/物品的简单优先级；
- 攻击与撤退；
- 日志和 replay 输出。

### 阶段 2：概率化/有限搜索 baseline

目标：在规则 agent 的基础上加入局部前瞻。

重点：

- 估计抽牌与检索成功概率；
- 比较多个可行动作的 Prize race 和 board value；
- 做 1 到 3 层 rollout；
- 记录搜索耗时和收益；
- 与纯规则 agent 做固定对手池比较。

### 阶段 3：神经网络研究

目标：学习策略先验或价值函数，而不是立即端到端控制全部动作。

优先尝试：

- imitation learning：从规则 agent、搜索 agent 或 replay 学习；
- value model：预测当前状态的胜率；
- policy model：在合法动作集合内排序；
- network-guided search：网络给动作排序，搜索做最终选择。

## 五、当前结论

你的方向是合理的，但“先观察优秀公开实现，再开始实现”需要转化成一个明确的短期任务：

1. 先把公开实现按方法分类，而不是盯着某个 notebook 的分数；
2. 明确复现一个简单规则 baseline；
3. 再复现一个公开概率化/Expectimax baseline；
4. 研究官方 RL/MCTS 示例的接口和训练方式；
5. 决定我们是否需要在 5080 上实现向量化环境；
6. 最后把神经网络路线拆成 policy/value/search 三个可验证的小实验。

当前最重要的下一步不是立刻训练大模型，而是弄清楚：什么是卡组优势，什么是 agent 决策优势，什么只是 leaderboard 采样和对手池差异造成的表象。

本轮进一步阅读的 Alakazam Notebook：

- https://www.kaggle.com/code/heiseimikiko/why-alakazam-is-a-good-baseline-for-ai
- https://www.kaggle.com/code/ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th

联合分析见：`docs/decks/alakazam-baseline-notebook-analysis-zh.md`。

# 从官方引擎到 Foundation Agent：可扩展宝可梦 TCG 智能体路线

**日期：** 2026-07-30  
**定位：** 当前竞赛成果的结构化解释，以及面向官方 Scale-up / Solution Write-up 的研究路线  
**当前基石：** `0019-0730-epoch13`，`SourceConditionedR15Policy`，17,756,162 参数

## 结论

本项目已经验证了一条比“为每套牌单独模仿一个玩家”更有扩展性的路线：使用大规模胜者行为克隆学习共享的卡牌、场面、资源与行动表示，再用构筑条件和 deck-specific RL 把同一份基础知识分化成不同牌手。

0019 Epoch 13 在 509 个 source、367 套 exact deck、7,347,132 个 train + validation decisions 上取得 `81.2871%` conditioned greedy exact-action；同一权重以中性 `source_id=0`、不做多龙 BC 或 RL，在官方引擎 30 个 Arena 对手、300 局真实对战中取得 `42.33%` 胜率。它强力支持“共享 Foundation 能迁移到新构筑”，但不证明 Encoder 单独贡献了全部能力，也不证明任意新构筑都能达到全局最优。

完整演进路线是：

```text
Official Engine
    -> Dynamics / Card Physics Pretraining
    -> Universal Winner BC (optional but highly valuable)
    -> Deck Adapter + Value Calibration
    -> Deck-specific RL in a frozen League
    -> Cross-deck Distillation
    -> Next-generation Foundation
```

## 证据边界

### 通用官方规则

- 回合先抽牌，再以任意顺序执行主行动，攻击后回合结束。
- 进化受进场回合和同回合再次进化限制；Rare Candy 不能绕过时机。
- Supporter、手动附能、撤退和 Stadium 各有每回合预算。
- 胜利来自拿完 Prize、对手无可战斗 Pokémon，或对手回合开始无法抽牌。

这些规则定义硬约束，不定义 PPO、BC、网络结构或最优策略。

### 当前 official engine runtime 事实

- 引擎返回当前 selection 的合法 option、`minCount` 和 `maxCount`。
- 引擎执行具体卡牌效果、随机结果、伤害、击倒、Prize 和终局判定。
- 项目策略必须只在合法 option 内输出 ordered full-action；非法动作应被结构性 mask，而不是通过负奖励学习。
- 所有策略强度结论必须来自未修改官方 engine runtime 的真实完整对局。

### 当前项目证据

| 证据 | 数值 | 能证明什么 | 不能证明什么 |
|---|---:|---|---|
| 0019 conditioned greedy exact | `81.2871%` | 多来源共享模型能保持很强的已见分布模仿能力 | 真实胜率或 unseen deck 泛化 |
| 0019 neutral greedy exact | `74.8431%` | 中性 persona 仍保留大量共享知识 | source 条件完全可删除 |
| 多龙 zero-shot Arena | `127/300 = 42.33%` | Foundation 在新构筑上具有真实迁移能力 | Encoder 单独贡献、全局最优或官方榜单分数 |
| 0018 Decoder-only PPO | 100 updates | 小范围末端更新能改变并改善部分策略表现 | 只训练末端永远足够 |

### 未来研究假设

- 官方引擎 transition 能作为无人工标签的卡牌动力学监督。
- 跨层 residual adapter 能在不覆盖 Foundation 的前提下提供足够可塑性。
- 多卡组 League、历史快照与跨分支蒸馏能形成逐代增强的 Foundation。
- 构筑搜索可以成为外层优化，policy adaptation 成为内层优化。

这些假设需要后续消融、同合同 Arena 和官方环境结果验证。

## 当前 0019 每一步实际做什么

### 输入合同

```text
Board entities           [B,E,7] / [B,E,5], E <= 192
Legal options            [B,O,12], O <= 128
Registered deck          [B,D] IDs + multiplicity
Causal resource ledger   [B,D,4] / [B,D,15]
Event memory             [B,T,8] / [B,T,4]
Known opponent hand      [B,H] + unknown count
Source persona           [B], 0 = neutral
```

### 第一层：共享状态与卡牌表示

Card、owner、zone、slot、status 和数值特征形成 entity tokens；四层 Board Transformer 产生：

```text
state_repr       [B,320]
entity_memory    [B,E,320]
```

registered deck、因果资源账本、事件、已知/未知对手手牌和结构化卡牌语义形成额外 memory。Goal-QKV 将 exact deck 组织为 setup、attack/prize、recovery、tempo/survival 四类目标。

这一层不是“知道答案”，而是建立当前玩家视角下可复用的世界表示。

### 第二层：Option 作为 Query 理解局面

每个合法 option 先编码动作类型、区域、卡牌、目标引用、数量和位置，再主动查询状态：

```math
\tilde{o}_i = \operatorname{Attn}(Q=o_i,\ K=M_{state},\ V=M_{state})
```

随后 option 继续读取 goal 和 scenario memory。其问题是：“如果考虑动作 `i`，场面中的哪些证据与我相关？”输出为：

```text
contextualized_options [B,O,320]
```

Boss option 可以关注对手 Bench、HP 和 Prize；Attack option 可以关注能量、伤害、接力和攻击终止回合的机会成本；Evolution option 可以关注进化时钟与构筑资源。

项目代码目前把这部分归入 `encode()`。语义上它已经是 option-conditioned policy reasoning，不能简单理解为纯状态压缩。

### 第三层：Hidden State 作为 Query 选择动作

Decoder hidden 汇总当前 `state_repr` 与已经选择的 action prefix：

```math
q_t = W_q h_t, \qquad k_i = W_k \tilde{o}_i
```

```math
\operatorname{logit}_{t,i} = \frac{q_t^\top k_i}{\sqrt{d}} + b(\tilde{o}_i)
```

因为 option keys 有 `O` 行，`q_t K^T` 自然产生 `O` 个 logits。选中 option 后：

```math
h_{t+1}=\operatorname{GRU}(\tilde{o}_{a_t},h_t)
```

模型重新对剩余 option 排序；STOP 只有在 `minCount` 后可选，生成在 STOP 或 `maxCount` 结束。

因此当前模型同时使用两种方向：

```text
Option as Q -> read state/goal/scenario memory
Hidden as Q -> rank contextualized option keys
```

### Value Head

Value Head 读取 `state_repr`，预测当前 actor-visible 状态下的终局期望：

```math
V_\psi(s_t) \in [-1,1]
```

0018 的初始 value head 新增 103,681 参数。它不会直接选择动作；如果 Encoder 冻结，value loss 也不会改变 Foundation 表示。

## 从 BC 到 RL：冻结什么，训练什么

### 阶段 0：结构性合法性

| 模块 | 状态 |
|---|---|
| Official engine | 生成合法 option 与完整规则结算 |
| Action mask / count contract | 强制启用 |
| Policy weights | 无需为“合法性”训练 |

随机策略也应只从合法 option 中采样。合法性是程序合同，不是 reward shaping。

### 阶段 1：Universal Winner BC

```math
L_{BC}=-\sum_t \log \pi_\theta(a_t^{expert}\mid s_t,a_{<t},deck,source)
```

0019 在这一阶段联合训练状态表示、option-conditioned reasoning 和 pointer Decoder。BC 同时学到：

- 可复用的卡牌与场面表示；
- 人类胜者的动作顺序与策略先验；
- exact deck 与不同 source 对决策分布的影响。

BC exact-action 衡量模仿，不等同于胜率。

### 阶段 2：Value Calibration

冻结 actor，训练 value head 拟合完整官方引擎 Episode 的终局回报：

```math
L_V=\mathbb{E}_t[(V_\psi(s_t)-G_t)^2]
```

若只使用终局胜负且 `gamma=1`，`G_t` 是该局最终的 `+1/-1/0`。这一步初始化 critic，不声称增强 policy。

### 阶段 3：Deck-specific RL

0018 的保守边界冻结所有 state/option representation，只训练约 1,130,883 个 Decoder + Value 参数。核心 PPO 目标为：

```math
L_{policy}=-\mathbb{E}[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]
```

并结合 value loss、entropy、behavior KL guard 和对 BC reference 的 KL。核心关心：

- terminal win/loss 是否真正提高；
- 新策略是否只在合法选项之间改变排序；
- 更新是否跨越有价值的旧策略局部最优；
- 是否在固定 Arena、先后手与 opponent snapshot 下稳定提升。

### 阶段 4：渐进式可塑性

只训练最终 pointer head 不一定足以表达新卡或新构筑。更强的边界应是：

```text
Frozen Card/Board/Event Foundation
    + trainable card/deck residual
    + trainable option-to-memory adapter
    + trainable pointer/value heads
    + optional last scenario layer
```

跨层 adapter 可以改变早、中、后层的信息流，却不覆盖 Foundation 原参数。只有出现可测平台期，才按较低学习率开放最后的 scenario/semantic 层。

遗忘保护使用：

```math
L = L_{PPO}+\lambda_VL_V+\lambda_{old}L_{anchor}
  +\lambda_{KL}D_{KL}(\pi_{foundation}\Vert\pi_{current})
  +\lambda_{dyn}L_{dynamics}
```

Reference KL 应主要约束旧 anchor states；新 deck states 必须允许更大偏离，否则策略无法离开旧 BC regime。

## 在 Engine 的“物理学”上预训练

### 数据来源

不需要真人动作标签。实验策略只需在官方引擎合法 option 中进行覆盖率导向探索，记录：

```text
(observation_before, ordered_action, resolved_events, observation_after)
```

竞技构筑不是必要条件，但合法且机制可达的实验构筑仍然必要。目标卡必须能被抽到、打出、进化、附能或触发；完全随机 60 张牌会产生大量不可达机制。

### 结构

```text
Card Encoder E_card
State Encoder E_state
Action Encoder E_action
Dynamics Head F
```

### 自监督目标

```math
L_{dynamics}=\lambda_\Delta L_\Delta+\lambda_eL_{event}
 +\lambda_zL_{latent}+\lambda_iL_{inverse}+\lambda_mL_{masked}
```

- `L_delta`：预测伤害、区域移动、资源预算、手牌/牌库、进化和状态的可见变化。
- `L_event`：预测引擎产生的 effect/event 类别与参与卡牌。
- `L_latent`：由 `E_state(s_t)` 和 `E_action(a_t)` 预测 target encoder 的 `E_state(s_{t+1})`。
- `L_inverse`：根据前后状态推断动作，防止表示退化为静态复制。
- `L_masked`：恢复遮挡的卡牌、区域或资源属性。

随机抽牌、coin flip 和隐藏手牌必须建模为分布或从 loss 中 mask；不能要求模型预测不可观测的唯一真值。

Dynamics pretraining 学到“动作会造成什么”，不学“什么时候值得这样做”。策略价值仍由 BC、搜索或 RL 获得。

## 从规则知识演变为顶尖牌手

### 没有真人数据时

```text
Coverage exploration
    -> dynamics-aware representation
    -> legal random policy + value head
    -> curriculum self-play
    -> historical-snapshot League
    -> deck-specialized policies
```

理论上，官方引擎给出合法动作、精确 transition 和终局胜负，足以定义一个可学习的部分可观测随机博弈。RL 可以在这个博弈中优化期望回报。

但 dynamics model 并不自动产生最优策略；PTCG 的隐藏信息、随机性、长程信用分配、动作序列与 matchup 循环使全局最优没有现实保证。工程目标应是对明确 opponent/deck 分布的强策略或近似均衡。

### 有真人数据时

Universal winner BC 是非常强的样本效率加速器：它将人类积累的长程策略、动作语法和资源偏好直接注入 policy，避免从随机对局重新发现全部组合。最完整路线不是删除 BC，而是：

```text
Engine dynamics -> Universal BC -> Deck RL -> League consolidation
```

## Arena：共享 Foundation、独立进化

### 运行结构

```text
One resident Foundation on GPU
    ├── Dragapult adapter + pointer/value heads
    ├── Alakazam adapter + pointer/value heads
    ├── Marnie adapter + pointer/value heads
    └── Candidate-deck adapter + pointer/value heads
```

每个引擎 observation 仍需动态编码；可共享的是权重、批量 forward 和部分静态 deck/card token。异构 adapter 需要按 adapter ID 分组或使用堆叠参数完成批量推理，不能靠逐请求重新加载模型。

### League 而不是两个在线模型无限互打

所有策略同时变化会产生非平稳性和循环克制。Arena 应保留：

- 当前策略与冻结历史 checkpoint；
- 多种构筑、强弱层级和先后手；
- 完整 matchup 向量，而不只是一维 Elo；
- 可重建 opponent snapshot 与采样权重；
- 固定评测合同下的晋级门槛。

### 代际 consolidation

```text
Foundation G0 (immutable)
    -> many deck-specific RL branches
    -> select robust policies and trajectories
    -> replay original BC anchors
    -> multi-policy distillation
    -> Foundation G1 (new immutable asset)
```

蒸馏目标可写为：

```math
L_{G1}=L_{old\ BC}+\sum_d w_dD_{KL}(\pi_d\Vert\pi_{G1})
 +\lambda_{dyn}L_{dynamics}
```

G0 永远保留；任何单 deck 失败都不会污染它。只有跨 matchup 稳健、通过官方引擎验证的新能力才进入 G1。

## 构筑搜索的外层优化

构筑搜索是双层问题：

```math
\max_{deck\in\mathcal{D}}\ \operatorname{ArenaScore}
  (deck,\ \pi^*_{deck})
```

```math
\pi^*_{deck}=\operatorname{Adapt}(\pi_{foundation},deck,\text{fixed compute budget})
```

可采用 zero-shot 筛选、短程 adapter RL、successive halving 和完整 finalist RL。必须给每个候选相同或可审计的适配预算，否则比较的是训练投入而不是构筑质量。

## 为什么从理论上可行

1. **充分监督来源：**引擎给出合法 action set、transition kernel 与 terminal reward，定义了策略优化问题。
2. **表示复用：**大量卡组共享区域、回合预算、资源、进化、目标选择和攻击提交语义；共享参数降低每个 deck 的样本复杂度。
3. **条件化分解：**exact deck、state、option 和历史形成条件，允许同一 Foundation 对不同构筑产生不同 representation。
4. **Pointer 归纳偏置：**动态合法 option 是输出词表，`qK^T` 在可变候选集合上保持对应关系和排列等变性。
5. **Actor-Critic 信用分配：**Value/GAE 将终局胜负转化为各决策点的 advantage，使长程准备动作能够收到训练信号。
6. **Residual specialization：**adapter 在不破坏共享参数的前提下提供 deck-specific function class。
7. **League 与蒸馏：**历史策略缓解非平稳性，跨分支蒸馏把局部创新合并为下一代共享先验。

这是一套可行机制，不是收敛证明。部分可观测、随机博弈和有限算力意味着最终只能通过严格的同合同 official-engine 评测与真实比赛结果确认强度。

## 面向官方 Scale-up 的核心叙事

本方案的价值不只是一套提交策略，而是把官方引擎转化为可扩展的学习基础设施：

- 引擎是规则与卡牌效果的真实世界模型；
- 人类 Episode 是高价值策略先验，而不是唯一知识来源；
- Foundation 把跨卡组共通知识压缩为一次 GPU 常驻；
- Adapter 与 League 允许许多构筑以低成本独立进化；
- Distillation 将 Arena 中验证过的新打法合并回下一代 Foundation；
- 最终可以扩展到新卡学习、构筑探索和更大规模的官方智能体生态。

当前竞赛仍以真实指标为第一优先级。研究路线的意义在于：现有 0019 与 0018 已经提供了可以审计的第一块证据，而不是只停留在概念图上。

## 事实来源

- [`experiments/0019_universal_winner_bc/DESIGN.md`](../../experiments/0019_universal_winner_bc/DESIGN.md)
- [`experiments/0018_alakazam_terminal_rl/DESIGN.md`](../../experiments/0018_alakazam_terminal_rl/DESIGN.md)
- [`experiments/0020_pluggable_deck_rl/evaluation/V2_zero_shot_dragapult.html`](../../experiments/0020_pluggable_deck_rl/evaluation/V2_zero_shot_dragapult.html)
- [`archive/pretrained/0019_universal_winner_bc_0730_epoch13/README.md`](../../archive/pretrained/0019_universal_winner_bc_0730_epoch13/README.md)
- [`docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`](../rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md)

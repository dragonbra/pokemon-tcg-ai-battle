# 0013 Semantic Goal Policy 设计规格

**日期：** 2026-07-26
**项目 ID：** `0013_semantic_goal_policy`
**状态：** 待用户书面审阅
**研究阶段：** 行为克隆表示研究；预留 value calibration 与 PPO 合同

## 1. 目的与边界

本项目从 `train/alakazam_sota_feature_engineering`（0012）出发，研究一套可被不同 deck-specific policy 复用的通用 agent 表示架构。首轮仍只训练胡地卡组和单一专家 Yushin Ito，不混合其他 deck、team 或 expert 标签。

项目要为 agent 建立：

- **眼睛：** 严格玩家视角的当前状态、结构化卡牌语义、因果资源账本、认识状态和事件记忆；
- **脑子：** 根据当前目标查询固定构筑能力和本局实际剩余资源的 Goal-QKV 表示，再对当前世界进行关系推理；
- **手：** 在官方引擎提供的合法 options 内执行完整有序动作，并让 BC、rollout 和 PPO 使用同一个动作概率合同。

首轮不做以下事项：

- 不训练跨卡组共享权重；
- 不引入卡文语言模型；
- 不让模型生成脱离官方 options 的动作 DSL；
- 不修改 `engine/source/`；
- 不自动把 candidate 收编进正式 opponent pool；
- 不进行 Kaggle submission；
- 不把 transition auxiliary head 称为 world model；
- 不以离线 exact-action 指标单独证明策略强度。

## 2. 0012 基线结论

0012 是一个已经通过官方引擎闭环验证的 full-action candidate pointer policy：

1. 官方引擎提供当前合法 `select.option`；模型不生成合法候选。
2. 当前状态被编码为 global、entity 和 option 张量。
3. 4 层、宽度 320 的 Transformer 编码 CLS 与 entities。
4. option 通过 cross-attention 读取状态。
5. GRU pointer decoder 自回归选择互不重复的 option，并在满足 `minCount` 后预测 STOP。
6. BC 学习 expert 的完整有序选择序列和终止时机。

0012 的主要表示瓶颈是：

- 主要依赖 card ID，缺少可组合的卡牌功能语义；
- 缺少持久的 player-local 资源知识；检索时看到的完整 deck 只作为瞬时 entities；
- unknown、missing、padding 和 clipped terminal category 存在语义混叠；
- 构筑不是可查询的能力集合；
- 实体、账本、事件和 option 的关系表达较弱；
- 历史方案尚未形成可训练、可在线复用的因果状态机；
- 没有统一的 value、sampling log-prob、entropy 和 PPO reevaluation 合同；
- winner-only BC、offline exact 与真实胜率之间存在明确差异，不能把模仿一致率当作牌力。

## 3. 项目身份与目录

项目采用仓库新格式三层边界：

```text
train/0013_semantic_goal_policy/          # 可执行的项目实现
experiments/0013_semantic_goal_policy/    # tracked 设计、审计、决策、版本和评测事实
rl_runs/0013_semantic_goal_policy/        # dataset、checkpoint、events、W&B staging
```

`0013_semantic_goal_policy` 明确取代未提交架构草案中作为首项目示例的 `0013_alakazam_rollout_value_calibration`。后者的 rollout/value 目标并入本项目后续阶段或另行分配下一项目编号，不再占用 0013。

实现责任边界为：

```text
train/0013_semantic_goal_policy/
├── __init__.py
├── __main__.py
├── configs/
├── features/       # typed schema、card semantics、observation/action codec
├── knowledge/      # causal state、self/opponent ledger、event updates
├── data/           # source reader、record、split、shards、audit
├── model/          # shared encoders、Goal-QKV、variants、policy/value heads
├── objective/      # BC、representation/counterfactual metrics、future value API
├── training/       # epoch、validation、checkpoint、status、experiment scheduler
├── runtime/        # stateful official-engine inference、sampling/evaluation API
├── export/         # self-contained candidate package
└── tests/
```

模块可以在实施时按合理粒度合并，但以下合同必须能独立测试：schema、causal knowledge state、dataset record、model registry、full-action probability、training metrics、runtime parity 和 candidate export。

项目级现状文档放在：

```text
experiments/0013_semantic_goal_policy/DESIGN.md
experiments/0013_semantic_goal_policy/DESIGN.html
```

两者必须随 feature schema、模型结构、action contract、训练目标或项目阶段同步更新。`DESIGN.html` 应提供交互式字段到张量、token、attention、goal routing 和 heads 的视觉说明。

## 4. 数据合同

### 4.1 来源

数据来自 `data/raw/` 中 2026-07-18 至 2026-07-25 的每日官方 episode archive，并包括 source manifest 明确列出的 2026-07-24 patched episode。

只保留：

- `info.TeamNames` 经 casefold 与空白归一化后精确匹配 Yushin Ito；
- Yushin Ito 对应 player 为 episode winner；
- 该 player 的完整 episode-player trajectory；
- 能通过 action、observation 和 causal visibility 合同的决策。

当前只读审计得到的预期原始规模如下，正式 builder 仍须重新计算并写入 audit：

| 日期 | 官方 episodes | team-name matches | winner groups | 兼容 decisions |
|---|---:|---:|---:|---:|
| 2026-07-20 | 4,545 | 559 | 301 | 25,013 |
| 2026-07-21 | 4,612 | 516 | 277 | 22,548 |
| 2026-07-22 | 4,639 | 434 | 229 | 17,067 |
| 2026-07-23 | 4,559 | 326 | 174 | 13,571 |
| 2026-07-24 | 4,445 | 339 | 188 | 16,382 |
| **总计** | **22,800** | **2,174** | **1,169** | **94,581** |

所有胜利数据仍需记录 winner-only selection/survivorship bias。匹配 replay 中没有可用的 submission ID，无法据此区分 Yushin Ito 在八天内是否更新 policy。因此本项目把 `Yushin Ito + 2026-07-18..25 + winner-only` 定义为用户明确批准的**命名专家来源边界**，但不得声称它是已验证的同一二进制策略版本。manifest 必须记录 `submission_id_unavailable=true`，dataset audit 必须按日期和 deck manifest hash 分组报告标签、action type、decision type 与关键指标分布；registered deck conditioning 必须使用每局真实己方构筑。若同一可比状态/合法候选合同下的跨日期或跨 deck-hash 冲突率超过训练前冻结的阈值，dataset gate 必须失败并停止正式训练，不得将冲突无条件混合。

### 4.2 动作顺序

新 builder 必须直接保留 replay `selected` 的原始有序列表。不得复用会对 action 执行 `sorted(action)` 的旧 builder 路径。完整动作标签包括：

- 原始有序、互不重复的 option indices；
- 官方 `minCount`、`maxCount`；
- 隐式或显式追加的 STOP target；
- action length；
- legal option semantic identity 与 actor-visible collision-safe occurrence identity；后者对语义相同的重复 options 按原 observation 中的可见 occurrence 做确定性编号，只用于建立 permutation 前后的双射，不作为模型 feature；任何 label 无法一一 remap 时 fail closed。

长动作不能被静默丢弃。若模型有训练长度上限，builder 必须 fail closed 或将该版本明确列为不支持，不能像 0012 V9 一样只在 runtime 外推而没有训练合同。

### 4.3 切分

八天数据先合并，再按完整 `episode + player` group 做固定随机切分：

- 只有 `train` 和 `validation`；
- 不设置离线 test split；
- validation 用于 early stopping、checkpoint 选择、M0–M5 比较和研究决策；
- 正式 official-engine evaluation 是本项目的最终测试；
- 同一 episode-player group 的任何 decision 不得跨 split；
- split seed、算法版本、group identity hash、每日/各 split counts 和 deck variant 分布必须进入 manifest；
- 默认使用稳定 hash assignment，而不是依赖输入遍历顺序的 PRNG shuffle；
- split 比例在实现计划中选择常规默认值并在首次 dataset build 前冻结。

数据中存在每天两个近似 Yushin Ito 构筑版本。它们仍属于同一专家姓名来源，但 dataset audit 必须按 deck manifest hash 分组记录，模型输入必须使用每局己方真实 deck manifest。训练和 validation 应保持各构筑变体都有可审计分布。若标签冲突分析发现两个构筑实际对应不可兼容策略，则必须停止并请求新的 source/deck 边界决策。

### 4.4 BC/RL 兼容 record

每个 decision record 至少包含：

- episode、player、step、date、split 和 source identity；
- 自己的 60 卡 deck manifest 及 hash；
- acting-player 当前 observation；
- causal knowledge state 的可重建输入或其经 schema-versioned builder 生成的 typed features；
- 当前官方 legal options、min/max 和原始 ordered selected action；
- actor-local 历史事件定位；
- terminal outcome，作为未来 return/value target 来源；
- schema、codec、ledger 和 action-contract version/hash。

`terminal_outcome`、未来 reward、episode result 和 future frame 绝不能进入 BC observation feature。当前数据没有逐 decision reward、behavior propensity、Q target 或 counterfactual outcome；这些字段不得伪造。actor 的下一个决策是 semi-MDP transition，不能把全局下一 frame 无条件当作 actor-perspective next state。

value-ready contract 采用版本化的 player-relative 原始结果：win=`+1`、loss=`-1`、draw=`0`；engine error/aborted/truncated episode 不产生 value target。`V(s_t)` 定义在 action-before、当前 actor perspective 的状态。首轮 BC records 只保留 raw terminal outcome 与到 terminal 的 official full-turn/decision duration，不预先构造 discounted returns；正式 value calibration 前必须在独立版本中冻结 discount、semi-MDP time basis、bootstrap/truncation 和 seat/perspective conversion。

## 5. 严格玩家视角与数据安全

每个时刻 `t` 的 feature 只允许由以下信息产生：

- 当前行动方的 `frame.obs`；
- 该玩家在时间 `≤t` 合法收到的 redacted logs；
- 该玩家自己的历史 actions；
- 该玩家自己的提交 deck manifest；
- 同一因果 builder 在此前时刻维护的 player-local knowledge state。

明确禁止：

- visualizer omniscient `current`、`select` 或 logs；
- replay `action` 中的对手完整 deck；
- future frames、后来发生的 deck view 或后来揭示的手牌；
- episode result、terminal outcome 或 reward 注入 observation；
- 通过全知 serial、hidden prize 或隐藏 deck ordering 反推玩家不可见信息。

离线 builder 和在线 official-engine runtime 必须复用同一 causal state implementation。任何仅能在 batch 构建时获得、在线 runtime 不可获得的 feature 都不得进入模型。

## 6. “眼睛”：Typed Observation + Knowledge Schema

模型输入由五类动态 token 和一类固定构筑能力 token 组成。

### 6.1 State tokens

表示当前决策时机和全局规则状态：

- actor、先后手、turn、turn action count；
- supporter、stadium、manual energy、retreat、turn-end 等使用状态；
- select type/context/effect context；
- `minCount`、`maxCount`、remaining damage/energy requirement；
- 双方 deck/hand/prize 数量；
- bench capacity、当前阶段和公开 stadium。

categorical、numeric、missing、unknown 和 not-applicable 通道必须分离。数值 normalization/cap 必须有审计范围和 overflow policy；不得把 overflow clip 到正常终端值而不标记。

### 6.2 Entity tokens

表示当前世界中的具体实例：

- self/opponent player；
- active、bench、hand、discard、stadium、looking 等 zones；
- Pokémon、Energy、Tool、evolution stack、temporary visible cards；
- HP/damage、statuses、attachments、entry/evolution timing；
- owner、viewer、zone、slot 和 physical serial（仅作为可见实例追踪键，不作为隐藏信息通道）。

显式关系包括：

- `owned_by`；
- `located_in`；
- `attached_to`；
- `evolves_from`；
- `targets`；
- `moves_from` / `moves_to`；
- `observed_in`；
- `summarizes`；
- `previous_event`。

关系是结构偏置，不编码策略答案。

### 6.3 Shared structured card semantics

card ID embedding 保留为小型 identity residual，并使用独立 `UNK_CARD`。card ID 不再承担全部卡牌含义。

共享 card semantic encoder 编码：

- Pokémon / Trainer / Energy 类型；
- Stage、HP、type、weakness/resistance、retreat；
- attacks、attack cost、damage 和结构化条件；
- Trainer subtype 和每回合使用约束；
- 抽牌、检索、回收、移动、换位、附着、伤害、状态、Prize 等结构化效果原语；
- target、source/destination zone、filter、count、condition 和 once-per-turn semantics。

首版只使用官方结构化卡牌数据和可审计的 effect primitives，不引入自由文本模型。相同 semantic encoder 必须供 registered deck、current entities、ledger identity、events 和 options 共同引用，避免多套不一致的 card embeddings。

### 6.4 Registered deck capability tokens

自己的 60 卡构筑是牌手天然知道的信息。每种 card identity 以：

- card semantic representation；
- card ID residual；
- multiplicity；
- capability/functional multi-labels；

构成 registered deck token。它表示该构筑的理论能力和资源冗余，不表示本局当前仍可用的资源。

对手完整 deck manifest 永不进入模型。

### 6.5 Causal ledger / belief tokens

资源账本同时保留具体 card ID 和可迁移功能组两级粒度。

己方 identity ledger 至少记录：

- initial copies；
- 当前 visible zone counts；
- known deck count；
- inferred Prize count；
- looking/temporary zone；
- source event、last-confirmed age 和 validity；
- knowledge stage：unobserved、visible-now、remembered、inferred-exact、bounded、unknown。

functional ledger 由结构化卡牌语义确定性聚合，例如 setup/evolution、attacker、search、draw、recovery、switching、energy access 和 disruption。它只提供事实汇总，不提供手写动作优先级。

对手 ledger 是非对称的：只记录该玩家合法公开看见或通过效果私下看见的信息，以及未知数量边界，不注入对手构筑。

### 6.6 Deck view 与 Prize 推断

官方引擎在涉及 deck selection 的时刻通过 `select.deck` 向行动方提供当时完整 deck；eligible cards 单独由 `select.option` 表示。因此：

1. 首次 full deck view 前，具体 deck/Prize 分布标记为 unknown/masked；
2. full deck view 的 decision 可以看到当前完整 deck，但 selected action 的后果不能提前进入该状态；
3. 从该时刻后的决策开始，builder 结合己方初始构筑、完整 deck 和已知 zones 做资源守恒；
4. 若消元能够确定 Prize identities，标记为 `inferred_exact`；
5. 未能确定时使用 unknown 或显式上下界，不能用 0 代替；
6. shuffle 清除 deck order/top-position 知识，但保留可确认的 membership/multiplicity；
7. Prize take、Prize placement/swap、Looking 和回收操作按 actor-visible logs 因果更新。

不得用未来第一次 deck view 反向填充此前 decision。

### 6.7 对手手牌知识

对手手牌表示为 known card-instance tokens 与 anonymous unknown slots 的组合：

- 未查看时，只知道 `handCount`，全部为 unknown slots；
- 合法查看整手牌后，当前可见 cards 转为带来源的 known instances；
- 对手打出、弃置或公开移动已知牌时，匹配实例离开 hand 并进入新 zone；
- 普通隐藏抽牌增加 unknown slots；
- 公开检索到手的 card 可以作为 known instance；
- hand reset、hand-to-deck shuffle 或无法保持身份对应的混合操作会使相应 known instances 降级为 unknown distribution；
- 不知道绝不编码成该卡数量为 0。

实际实现前必须只读核验引擎对查看手牌、公开检索、随机弃置和 hand reset 的 redacted log/serial 语义，并把发现写入 visibility audit。

### 6.8 Event memory

事件流记录因果变化，而非只记录己方 selected option 摘要：

- draw、search、reveal、move、discard、recover、attach、evolve、retreat；
- attack、damage、KO、Prize；
- shuffle、deck view、hand view；
- self/opponent action summaries；
- actor、source/target entities/zones、card semantics、relative decision/turn age 和 visibility source。

事件窗口可有固定上限，但截断方式、窗口覆盖率和 ledger 已吸收的信息必须审计。ledger 是确定性长期事实；event memory 提供近期因果上下文，二者不能互相替代。

## 7. “脑子”：Semantic Goal Policy Reference Architecture

### 7.1 模型规模

首个完整 reference 目标为约 18M–28M 参数，以 official-engine CPU evaluation 和后续大量 GPU rollout 的吞吐为约束。初始 reference 超参数：

- shared `d_model = 384`；
- 6 层 pre-norm relational state Transformer；
- 8 attention heads；
- FFN 约 1536；
- 2 层 semantic option cross-attention；
- 小型 causal Transformer 或 recurrent full-action decoder。

具体参数可通过 smoke 和显存/吞吐审计微调，但任何正式变化必须新建版本并更新 DESIGN。

### 7.2 Typed token adapters

State、entity、ledger、event 和 option 使用职责独立的 typed adapters，再映射到共享 `d_model`。card semantic encoder 是共享底座。

需要区分：

- padding mask；
- unknown mask；
- missing/malformed mask；
- not-applicable mask；
- visibility/epistemic state；
- relation type；
- self/opponent perspective。

### 7.3 Goal-QKV 核心路由

采用四个首版 goal slots：

1. setup / board development；
2. attack / Prize progress；
3. resource access / recovery；
4. tempo / survival。

这些是可学习的战略查询角色，不是人工动作标签或规则策略。

每个 goal query：

- `Q` 来自 public state、goal-role embedding 和 role-specific numeric slice；
- `K` 来自 registered deck card capability 和 live ledger capability；
- `V` 是独立投影的战略资源表示；
- 分别读取 registered deck 的理论能力与 live ledger 的实际可用性/认识状态；
- 返回 goal-conditioned context token。

物理语义是：构筑不是一个静态标签，而是一组能被当前问题查询的能力；live ledger 则告诉模型这些能力在本局是否仍可实现。

### 7.4 Relational state Transformer

输入为：

- `[STATE]` token；
- current public/private-visible entities；
- ledger tokens；
- recent event tokens；
- four goal context tokens。

relation type 通过 attention bias 或等价可审计机制注入。输出包括全局 contextual state 和各实体/账本/事件的 contextual embeddings。

世界状态先独立编码，不能由 option ordinal 或候选数量定义。相同状态只编码一次，供 policy 与 value 共用。

### 7.5 Semantic option encoder

每个官方合法 option 编码：

- action/select/effect primitive；
- source/target entity references；
- source/target zones；
- card/attack/energy/tool/count 参数；
- move/resource dependency semantics；
- context card 和 min/max contract。

option query 通过 cross-attention 读取 contextual state、相关 entities、goal contexts、ledger 和 events。模型只给官方合法 options 排序，不自行构造合法动作。

### 7.6 Full-action decoder

完整动作是一个有序、互不重复的 option sequence 加 STOP：

- 每步只在尚未选择的官方 options 与条件允许的 STOP 上归一化；
- STOP 在 `minCount` 前 masked；
- 当已选数量小于 `maxCount` 时，STOP 是显式随机变量；当达到 `maxCount` 时，终止由合同强制，追加的 terminal marker 概率定义为 1、log-prob 为 0、entropy 为 0，不再经过模型 softmax；
- teacher forcing、greedy evaluation、stochastic rollout 和 PPO reevaluation 共用相同实现；
- option permutation 后必须同步 remap 所有 option-aligned fields 和 labels；
- decoder 不以原始 option ordinal 作为策略捷径。

统一概率定义：

```text
log π(a | s)
  = Σ_t log π(option_t | s, option_<t)
    + log π(STOP | s, option_≤K)
```

entropy 是所有**实际发生、非强制** decoder decisions 的 entropy 求和；同时记录按 decision step 平均的 normalized entropy 作为跨 action-length 诊断。PPO objective 使用 action-level summed log-prob 计算 ratio，并把 summed entropy 与 mean-step entropy 分开记录；entropy bonus 的选择与系数必须在首次 PPO 版本前冻结。

### 7.7 Heads

共享骨干支持：

- **Policy head：** semantic full-action pointer logits；
- **State value head `V(s)`：** 严格 player-perspective、action-before state value；
- **训练期 auxiliary heads：** resource availability、knowledge state、goal relevance、公开 transition consistency 或 masked semantics。

辅助目标必须只预测当时合法可知或明确作为监督标签的事实，不能向 policy 输入 future outcome。辅助 heads 可拆卸，其 loss 权重必须成为显式实验变量。

首轮以 BC policy 为主；value head/API 在 M5 达到 value-ready，但正式 value calibration 和 PPO 是后续阶段，不能用未校准 value 输出冒充策略价值。

## 8. M0–M5 可证伪实验阶梯

每个模型是 model registry 中的明确 variant，不使用难以审计的巨型布尔开关模型。

| Variant | 相对前一版本的新增 | 研究问题 |
|---|---|---|
| M0 | 0012-compatible baseline，使用新数据与统一训练/评测合同 | 在新 dataset 上复现已知下界 |
| M1 | 独立 UNK/masks + shared structured card semantics | 看懂可组合卡牌能力是否有效 |
| M2 | registered deck mean-pool residual → CLS | 知道固定构筑 prior 是否有效 |
| M3 | four Goal-QKV queries → registered deck capabilities | 按目标查询构筑是否优于平均摘要 |
| M4 | causal live ledger、Prize/known-hand epistemic state，并进入 Goal-QKV | 实际剩余资源知识是否改善计划 |
| M5 | relation bias、event memory、state value-ready contract | 完整 BC/RL 通用基座是否产生净收益 |

M0 必须尽量保持 0012 action/architecture baseline，但使用新数据合同；它不能通过读取新 ledger 或 goal features 获得隐性优势。

每项 variant 必须在 config、DESIGN、checkpoint metadata 和 W&B `model_variant` 中显式标识。失败或中止版本不删除、不覆盖。

## 9. 表示与决策举证

证据分四层。

### 9.1 合同正确性

- schema/shape/range/mask；
- episode-player split isolation；
- causal visibility；
- card/zone resource conservation；
- option/entity/deck permutation consistency；
- train/runtime feature parity；
- teacher/sampling/PPO action probability parity；
- candidate official-engine legality。

### 9.2 Representation probes

用训练期 probe 或冻结表示线性 probe 检查：

- resource availability；
- exact/bounded/unknown Prize state；
- attack readiness；
- evolution/setup route；
- goal-relevant registered deck capability；
- opponent known-hand versus unknown slots。

probe 高分不单独构成策略强度证据，也不得通过泄漏标签实现。

### 9.3 Causal counterfactual pairs

至少包含：

- 相同公开局面，将 Alakazam ledger 从 `deck=1` 改为 `deck=0, Prize=1`；依赖该资源的路线 logit 应下降；
- 相同真实隐藏分布，比较 deck-view 前 unknown 与合法 view 后 inferred-exact；模型只能在后者利用身份；
- 替换 registered deck 的备用攻击/回收能力；相关 goal 改变，无关 goal 保持稳定；
- 对手 hand view 后 known、打出后移除、隐藏抽牌后新增 unknown；
- unknown 与 confirmed-zero 的动作响应必须可区分。

记录 expected-direction rate、relevant margin 和 irrelevant-goal stability。

### 9.4 Official-engine strength

唯一真实策略强度结论来自正式 official-engine runtime：

- 固定 candidate package hash；
- 固定 enabled opponent catalog snapshot；
- 每个 enabled opponent 20 局；
- 本机默认 `--workers 8 --worker-cpu-threads 1`，除非 benchmark 表明需调整；
- 固定 seat/seed contract、engine revision 和 metric profile；
- 记录 wins/losses/errors、win rate、first/second-player、completion、full turns、setup/relay/attack metrics 和耗时。

离线 exact 下降而 official-engine 胜率上升是允许的重要结果，但必须排除 error、完成率、matchup 和随机预算差异。

## 10. 训练、W&B 与十小时调度

### 10.1 正式版本

每个通过 smoke 门槛的实际训练都分配新的不可覆盖 `V<n>_<tag>`。建议语义顺序为 M0→M5，但实际 `Vn` 由严格 next-version allocator 分配；任何失败也消耗版本。

运行资产：

```text
rl_runs/0013_semantic_goal_policy/versions/V<n>_<tag>/
├── artifact/
│   ├── training_config.json
│   ├── training_metrics.jsonl
│   ├── training_summary.json
│   └── status.json
├── checkpoint/
├── tensorboard/
└── wandb/
```

每个 epoch 对完整 train 和 validation 做同口径评测。canonical 写入顺序是 JSONL flush → TensorBoard → W&B。

### 10.2 W&B 合同

正式训练使用：

```text
entity: dragon_bra
project: pokemon-tcg-policy-learning
mode: online
group: 0013_semantic_goal_policy
name: V<n>_<tag>
run_id: stable(project_id, version_name)
job_type: bc_train（首轮）
```

每个 repository version 一一映射一个稳定 W&B run。仅真实断点恢复可 resume；模型语义、schema、数据或超参数变化必须新建 version/run。

W&B config 至少记录：

- project/version/model variant；
- schema、codec、ledger、action-contract versions/hashes；
- dataset manifest/hash、split seed/counts；
- deck manifest distribution；
- expert/source identity；
- engine revision、git commit/status；
- model dimensions/parameter count；
- optimizer、schedule、seed 和 early-stopping rule。

metrics namespaces：

- `bc/*`：train/validation loss、token accuracy、exact、free-running validity、length buckets；
- `representation/*`：resource、Prize、goal-role、knowledge-state probes；
- `counterfactual/*`：direction rate、margin、stability、unknown-vs-zero；
- `invariance/*`：option/entity/deck permutation agreement 和 aligned KL；
- `system/*`：epoch seconds、examples/s、decisions/s、GPU peak memory、parameter count；
- `eval/*`：只从正式 official-engine report 镜像。

不上传 checkpoint、optimizer、dataset、raw observation、replay、trace 或 source patch。

每个正式 W&B run 还必须在关键 checkpoint 选择完成和 run 正常结束时，将以下**小型、非秘密、可追溯审计文件**保存到该 run 的 Files 区，并使用 SDK 的立即上传语义 `wandb.save(path, policy="now", base_path=<version artifact root>)`：

- `training_summary.json`；
- `status.json`；
- `checkpoint_selection.json`；
- `model_contract.json`；
- `dataset_reference.json` 或等价的小型 dataset contract/hash 引用；
- `metrics_snapshot.json`：仅包含关键 epoch/selected checkpoint 的精简指标和曲线摘要，不复制完整 observation 或大体量历史。

上传清单、每个文件的 SHA-256、上传时机和结果必须记录到 `wandb_snapshot_manifest.json`；该 manifest 本身也用 `policy="now"` 上传。若 `wandb.save` 不可用、返回未上传或发生异常，本地文件和 checkpoint 不回滚，但 `status.json` 中的 W&B snapshot 状态必须为 failed 并记录原因。不得使用该机制上传 checkpoint、optimizer state、完整 `training_metrics.jsonl`、dataset、replay 或 trace。由于 `policy="now"` 属于向外部 private W&B 项目上传文件的操作，本规格中的用户批准只覆盖上述固定白名单和本项目正式 run，不扩展到其他文件或其他服务。

当前 W&B 基础设施在 0013 正式训练前必须修复：

- 接受新 canonical nested metrics path；
- 使用版本内 sibling `wandb/` staging；
- 明确 BC/value/PPO/rollout axes，而不是只靠 metric name 猜测；
- 把 project/entity/run ID/URL/sync state/failure reason 写入 status；
- exception 退出时向 W&B 传递非零 exit code；
- W&B failure 不回滚本地指标或 checkpoint。

### 10.3 至少十小时持续 GPU 探索

用户授权：

- GPU 持续使用目标至少 10 小时；其 primary accounting 是从长训练阶段开始后，至少一个健康训练进程实际执行 GPU work 的**非重叠 elapsed intervals** 总和，不按并行 job 重复累计；短暂 checkpoint/validation/job 切换间隔、device utilization、idle、failed 和 overlapping intervals 分开记录；允许间隔上限和 telemetry 采样频率在 protocol 中冻结；
- 在冻结的数据、schema、M0–M5 和预定义超参数边界内自适应分配；
- 自适应 GPU 预算开始前，M0–M5 每个通过基础 smoke 的 variant 至少完成 protocol 冻结的 minimum allocation（默认一个正式 seed 且至少达到 minimum epochs/examples）；只有硬失败或达到 minimum allocation 后才可淘汰；
- 弱变体可按预定义 validation/stability 门槛早停；
- 剩余预算可转给更有希望的变体、额外 seed 或延长训练；
- 不得在无人确认时改变数据来源、引入新 feature、修改 reward 或越出 M0–M5；
- 每次实际正式训练必须使用独立 version 和 W&B run；不能在一个 run 中切换模型语义；
- 调度决定、停止原因和累计 GPU wall-clock 必须写入 experiment decisions/status；
- 若实现、数据审计、W&B 登录、GPU 或训练稳定性阻止安全启动，不得用空转或失败循环凑满 10 小时；应保留证据并报告 blocker。

自适应优先级依据 validation BC、representation、counterfactual、invariance、runtime validity、吞吐和稳定性，不以单一 exact 排名。

### 10.4 Checkpoint 保留与自动选择决策

每个正式版本至少保留以下彼此独立的 checkpoint，不得用后续保存覆盖：

- `latest`：最后一个完整、可恢复的 epoch；
- `best_validation_loss`：最低完整 validation BC loss；
- `best_validation_exact`：最高完整 validation exact-action；
- `best_representation`：按冻结的 representation composite 取得最高分；
- `best_decision_quality`：按冻结的 counterfactual/invariance 门槛与 composite 取得最高分；
- `selected_for_evaluation`：实际导出并评测的 checkpoint 的物理副本或内容寻址的不可变引用。

如果多个 criterion 在同一 epoch，manifest 可以引用同一 checkpoint hash，但不能因节省空间而失去各 criterion 的选择记录。每个 checkpoint metadata 至少记录 epoch、global step、model variant、config/dataset/schema hashes、完整 train/validation metrics、保存 criterion、SHA-256、参数量和创建时间。

自动正式评测只从通过以下硬门槛的 checkpoint 中选择：

1. dataset、causality、ledger、action 和 runtime audits 全部通过；
2. validation free-running legality/completion 无回归，official candidate validate 通过；
3. 无 NaN/Inf、AMP overflow、checkpoint corruption 或 W&B/local metric divergence；
4. option/entity/deck permutation consistency 达到规格阈值；
5. counterfactual direction 和 unknown-vs-zero 测试达到规格阈值；
6. 推理吞吐满足预先冻结的评测/rollout 可运行下限。

在合格集合中，选择逻辑按以下词典序执行，而不是在训练完成后临时改变权重：

1. 最大化冻结的 `decision_quality_composite`，由 counterfactual expected-direction、knowledge-mask discrimination、irrelevant-goal stability 和 invariance 组成；
2. 若在预定义容差内并列，最大化 validation exact-action；
3. 再并列时最小化 validation BC loss；
4. 再并列时选择更早 epoch，以降低过拟合和推理漂移风险；
5. 不把参数量或速度直接混入质量分；它们只作为硬吞吐门槛和单独报告项。

composite 的具体分量、归一化、阈值和并列容差必须在第一次正式训练前写入 config/decision record，之后不得根据结果修改。若 representation/counterfactual 指标尚未达到足够可信度，自动评测选择退化为通过所有硬门槛后 `best_validation_exact`，并在决定中明确标记 `fallback_reason`；不得暗中主观挑选曲线最好看的 checkpoint。

选择结果必须写入可追溯的 `checkpoint_selection.json` 和版本 decision/status，至少包括：

- 所有候选 checkpoint 及其 hashes；
- 每项门槛的 pass/fail 和证据路径；
- composite 分项、总分、排序和 tie-break；
- 被淘汰 checkpoint 的明确原因；
- 最终选择的版本、epoch、checkpoint path/hash 和 candidate package hash；
- 选择时读取的最后一条 canonical JSONL record；
- 对应 W&B run ID/URL，但本地 JSONL 仍为事实源；
- official-engine report path/run ID/hash。

自动选择 checkpoint 后，所有原始 checkpoint 和 optimizer/resume state 保持完整，不因导出 candidate 或正式 evaluation 删除、覆盖或重命名。明日 W&B 对照验收时，应能从 run summary 看到选中 epoch、selection criterion/composite、candidate hash 和 formal evaluation backlink，并能用本地记录重放同一选择。

## 11. 运行时、错误处理和 fail-closed 门槛

### 11.1 因果更新顺序

每个决策：

1. 接收 actor observation；
2. 消费当前可见 incoming logs；
3. 更新并校验 causal knowledge state；
4. 编码 `s_t`；
5. 选择 full action `a_t`；
6. 把动作记录为 pending event；
7. 等待后续官方 observation/log 确认结果；
8. reconcile transition，进入下一时刻。

selected action 的后果不能提前进入 `s_t`。

### 11.2 统一 Policy API

概念接口：

```python
encode(input: TypedPolicyInput) -> PolicyState
sample_action(state, legal_options, deterministic=False) -> ActionSample
evaluate_action(state, selected_sequence) -> ActionEvaluation
```

`ActionSample/Evaluation` 至少包含 selected sequence、aggregated log-prob、entropy 和 `V(s)`。deterministic candidate 仍只返回官方合法 action contract 所需结构。

### 11.3 Fail closed

- schema/version/shape/mask/invalid ID 错误：拒绝 sample 或 run；
- ledger 总量不守恒、serial 双重占区、future knowledge：audit failure；
- expert action 不在 legal options、重复或 min/max 不符：拒绝并记录；
- runtime 永不返回非法 index；
- W&B 失败保留本地证据并把 sync 标为 failed；
- dataset、version、正式 evaluation 已存在时拒绝覆盖；
- candidate package 验证失败时不得进入正式 evaluation。

## 12. 分阶段实施、选择与验收顺序

完整工作必须拆成具有 stop-gate 的连续阶段；下游阶段不得在上游验收失败时为了凑满 GPU 时间而启动：

1. **基础设施阶段：** 整合新实验项目生命周期，修正 0013 身份、W&B nested-path/status/snapshot 合同，并迁移 evaluation 正式路径到 `experiments/<project>/evaluation/`；保留 legacy 报告只读兼容。相关测试全部通过才进入下一阶段。
2. **数据合同阶段：** 实现 source reader、patched-over-original precedence、collision-safe ordered action identity、context-specific visibility audit、structured semantics ontology v1、causal ledger 和 deterministic stratified split；完成 identity/split/visibility/conservation/action audit 后才可训练。
3. **基础模型阶段：** 实现统一 typed contracts、M0–M3 和 full-action probability parity；每个 variant 至少完成预定义 minimum-allocation 正式 run 后才允许自适应淘汰。
4. **完整表示阶段：** 实现 M4–M5、opponent known-hand ledger、event runtime parity 和 value-ready API；通过 counterfactual/invariance/runtime gates。
5. **长训练阶段：** 在冻结 protocol 内自适应利用至少 10 小时 active GPU elapsed time，执行 version-local checkpoint selection 和 project-level winner selection。
6. **导出评测阶段：** 导出 project winner candidate、validate；若 package validation 失败，按冻结排序尝试下一 eligible version winner，否则以 no-eligible 结束。只对最终一个合格 candidate 做每 enabled opponent 20 局的正式 official-engine evaluation。

### 12.1 两级 checkpoint 选择与后置 provenance

- **Version-local preliminary selection：** 只依据训练结束时已存在的 dataset/model/probability/invariance/counterfactual/runtime evidence，为每个版本选择一个 winner checkpoint。
- **Project-level preliminary selection：** 用冻结的跨版本排序规则比较所有 eligible version winners，生成 `project_checkpoint_selection.json`，选出一个待导出的 checkpoint。
- **Export/validation：** 导出后才产生 candidate hash 并运行 package validation。失败时按冻结排序选择下一名；不修改分数或门槛。
- **Final provenance：** formal evaluation 后另写不可变 `selection_provenance_final.json`，加入 candidate hash、validation evidence、report path/run ID/hash。不得把尚未存在的字段塞入 preliminary selection。

每个正式版本可以保存 `selected_version_checkpoint`，但全项目只有一个 `selected_for_formal_evaluation`。若所有 checkpoint 或 candidate 都未通过，`no_eligible_checkpoint` 是合法且可审计的终态：保留所有 runs/checkpoints，上传白名单审计快照，生成失败排序与原因，不执行正式强度评测，也不降低门槛。

### 12.2 Pre-run protocol 冻结文件

任何正式 dataset build/train/eval 前，必须写入并 hash 一个 `pre_run_protocol.json`，明确而非留待结果后选择：

- train/validation 比例、稳定 hash 算法、seed、按 date + deck-manifest-hash 分层规则、稀有 strata 处理与 minimum groups；
- patched episode 以 canonical episode ID 替换 archive 原件的优先级和去重规则；
- event window/truncation；
- card effect ontology version/hash、mapping/人工复核流程、coverage threshold、multi-effect 表示和 `UNKNOWN_EFFECT` fallback；
- deck/hand view context 的 `none|eligible_subset|full_membership|ordered_view` 分类与 exact inference 许可；
- model seeds、minimum allocation、epoch/early-stop bounds、超参数搜索边界；
- 所有 audit/eligibility thresholds、composite 公式/normalization、tie tolerance 和 trusted-metric 条件；
- active GPU time 定义、允许的短切换间隔和 telemetry；
- official evaluation metric profile ID/revision、exact per-game seed/seat schedule、catalog/package/engine/evaluation-code hash manifest。

默认值由实施计划依据仓库现有常规和数据审计制定；一旦正式运行开始不得根据结果修改，修改必须新建 protocol/version。

### 12.3 W&B 与正式评测生命周期

训练 run 在训练完成时可正常 finish。导出/评测后，允许且只允许使用同一稳定 run ID 以 `resume="allow"` 重新打开对应 private W&B run，幂等写入白名单 audit snapshot、有限 `eval/*` scalars 和 formal backlink，然后再次 finish。若没有合格 candidate 或 evaluation 失败，写入明确终态，不伪造 eval 指标。

W&B snapshot 使用 immutable generation：先生成并 hash 固定白名单 payload，上传 payload，再写不包含自身 hash 的 generation manifest 并上传；最终本地 status 单独更新。manifest 的本地 hash记录在 status/summary，不制造自引用。所有允许文件必须位于 resolved artifact root、是非 symlink regular file、通过 secret/raw-data scan，单文件与总字节上限写入 protocol；不合规则拒绝上传。

正式 evaluation 的新路径迁移、immutable refusal、index refresh、`artifact/evaluation.json` backlink 和 legacy read-only compatibility 必须有测试并在评测前通过。

## 13. 完成标准

0013 首轮完成要求：

- 项目三层目录和版本生命周期符合新架构；
- 数据来源、winner 筛选、deck variants、split 和所有 hashes 可审计；
- dataset 无 player-perspective/future/omniscient leakage；
- causal ledger 在离线与 runtime 中行为一致；
- structured semantics、Goal-QKV、ledger、event/relation 和 action contract 有独立测试；
- M0–M5 至少经过 smoke，正式训练版本按门槛和十小时调度执行并保留证据；
- 每个正式 epoch 的 train/validation metrics 同步到 JSONL、TensorBoard 和 W&B，sync 状态可追溯；
- representation/counterfactual/invariance 指标可比较；
- 最佳合格 candidate 完成自包含验证和每 opponent 20 局的官方引擎正式评测；
- DESIGN.md/HTML 与最终 schema、shapes、参数、数据 audit、checkpoint metadata 和当前阶段一致；
- 最终结论允许 M0–M5 任一模型获胜，不预设完整 reference 一定优于简单基线。

## 14. 已批准决策摘要

- 项目：`0013_semantic_goal_policy`；
- 通用架构、deck-specific policy weights；
- 结构化卡牌语义优先，card ID 为 residual；
- current entities + causal event memory + asymmetric resource ledger；
- 自己完整构筑已知，对手构筑未知；
- card ID 与 functional group 两级账本；
- full deck view 后才解锁 Prize 推断；
- 对手手牌支持 unknown→known→zone move→新增 unknown 的因果更新；
- Goal-QKV 同时查询 registered deck capability 和 live ledger；
- BC/RL 共用骨干与 semantic full-action pointer；
- 约 18M–28M 的中型 reference；
- M0–M5 可证伪实验阶梯；
- Yushin Ito 2026-07-18 至 2026-07-25 全部胜利对局；
- 八天混合、episode-player group 固定随机 train/validation split；
- 无离线 test，official-engine evaluation 为最终测试；
- 门槛后所有正式训练使用 online W&B；
- 至少 10 小时持续 GPU、自适应预算分配；
- 最佳合格 candidate 自动正式评测，每个 enabled opponent 20 局；
- 不自动收编 opponent，不进行 Kaggle submission。

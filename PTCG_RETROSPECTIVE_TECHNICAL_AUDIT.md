# Pokemon TCG Kaggle 赛后复盘技术取证审计

审计日期：2026-08-19

审计范围：当前 worktree、Git history、`train/`、`experiments/`、`rl_runs/`、候选 archive 与 `.tmp/evaluation/` 中仍存在的证据。
证据边界：这是 read-only forensic audit；除本报告外未修改实现、未启动训练。当前 worktree 中 0045 V14-V20 的一部分记录尚未提交，因此下文会明确标为“当前 worktree 证据”，不能把它们描述成某个已发布 commit 的内容。

| Claim | Verdict | Evidence | Confidence | Safe to publish? |
|---|---|---|---|---|
| 早期模型只有 card-ID embedding | **Wrong** | `train/0010_alakazam_sota_model/model.py:136-318,344-425` | High | Rewrite |
| 早期模型是 raw-state / ID-heavy，缺少卡牌静态 mechanics | **Confirmed** | `train/0010_alakazam_sota_model/model.py:1-5,136-318` | High | Yes |
| 后来直接从 official engine 抽取 prototype/attack/effect ontology | **Confirmed** | `train/0031_rule_faithful_semantic_foundation_pretraining/tools/full_engine_prototype_export.cpp:52-264` | High | Yes |
| 后期完全移除了 card-ID embedding | **Wrong** | `train/0031_rule_faithful_semantic_foundation_pretraining/model/prototype_encoder.py:160-164,385-414` | High | Rewrite |
| semantic features 已证明消除了 Ogerpon 错误附能 | **Unknown** | `train/0025_semantic_foundation_pretraining/tests/test_semantic_contract.py:96-101` 只证明可表示性 | Medium | No |
| 最早 PPO 没有 critic/value network | **Wrong** | `train/0017_dragapult_terminal_rl/policy/actor_critic.py:57-103` | High | Rewrite |
| “第一轮 rollout 的 value calibration 过拟合” | **Partial** | `train/0017_dragapult_terminal_rl/training/value.py:47-166`; V4 calibration metrics | Medium | Rewrite |
| 后来训练了 BC-pretrained value head | **Partial** | `train/0036_dedicated_action_value_network/model/value_network.py:16-150`; `data/catalog.py:135-174` | High | Rewrite |
| RL 始终只用 final win/loss、从未做 prize shaping | **Wrong** | `train/0038_action_boundary_rl/integrated/prize.py:42-111`; `training/ppo_full_semantic.py:346-418` | High | No |
| 0037 的 shared Option LoRA 同时收到 policy/value gradient | **Confirmed** | `train/0037_dragapult_value_initialized_rl/training/ppo_full_semantic.py:83-106,196-246` | High | Yes |
| 0042 为隔离梯度删除 Option LoRA，并以显式 detach 隔离 critic context | **Confirmed** | `experiments/0042_full_model_design/DECISIONS.md:5-11`; `policy/strategy_adapters.py:13-48` | High | Yes |
| 后期开 encoder adaptation 后强度被严格消融证明提升 | **Unknown / suggestive only** | 0045 V14-V19 logs；V14 checkpoint omission 破坏严格比较 | Medium | Rewrite |
| Frozen-0806 曾实际成为 0809 trunk + 0806 head hybrid | **Confirmed** | `experiments/0040_dragapult_0809_action_boundary_rl/POLICY_IDENTITY_AUDIT_2026-08-11.md:3-28` | High | Yes |
| “约 67% training curve vs 约 64% independent eval”是同合同对比 | **Wrong** | V2 `champion.json:8`; V6 `report.json:48937-48938` | High | No |
| 最终 0045 checkpoints 是在 deck 067 上训练、却误包 deck 007 | **Wrong on available evidence** | V14/V17/V19 `training_config.json`; V20 config; archive manifests | High | No |
| 曾构造包含 Moltres 的错误 deck-067 package | **Confirmed** | `.tmp/evaluation/invalid_0045_007_v3_067_package_20260817/package_dir/manifest.json:15-20,31`; `deck.csv` | High | Yes, with caveat |
| 错误 package 实际提交且耗尽最后 submission opportunities | **Unknown** | 现存 manifest 明写 `kaggle_submitted: false`；无 submission receipt/quota log | Low | No |

## 1. Engine semantics

### Historical reconstruction

**0010：raw observation + identity-heavy representation。** `0010_alakazam_sota_model` 的文件头明确说它只使用公开 card ID 和结构位置，不读取 card metadata 或 hand-authored semantics。它并非“只有 card ID”：global token 包含 select type/context、先手、每回合限制、turn/action count、deck/hand/prize 数、option 数、剩余 damage counter/energy cost 和 bench 数；entity token 包含 owner/zone/slot/kind/status/parent、当前伤害、附能/工具/退化卡数量和本回合出现标记；option token 包含 action type、source/target area/card ID/entity ref、number、slot 和 ordinal。真正缺少的是卡牌原型的静态规则结构，例如 HP 上限的语义、进化关系、招式能量颜色与 effect graph。

**0013：先出现一层部分语义，而不是从 ID-only 一步跳到完整 engine ontology。** `card_semantics.py` 从官方 CSV 与文本规则构建 card kind、stage、category/type、HP、retreat、weakness/resistance、evolution、move damage/energy、effect kind/target/condition 和若干 capability flag；其中部分 effect/capability 是 regex/heuristic text parsing。compiler 把它压成 64 维语义向量，模型把 `card_identity` embedding 与 semantic projection 相加/拼接进实体、deck、ledger、event、option 路径。这是有意义的中间阶段，但不是 engine internal effect graph 的忠实导出。

**0025：把 official engine 当作 prototype specification。** read-only C++ exporter 调用 engine 的 `InitializeAll()`，遍历 `CardTable`、`SkillTable`、`AttackTable`，导出 card、attack、skill、effect、target/condition/value/flag 等结构。0025 compiler 还显式计算 attack energy requirement 与场上已附 energy type 的匹配，并生成 typed/total deficit、target HP after damage、KO 等派生 option features。

**0031：保留规则因子，但收回“答案型”派生特征。** 0031 保留 card/attack/skill/effect prototypes、resolved attached-energy units 和 legal option structure，却将 `typed_energy_deficit`、`total_energy_deficit`、`target_hp_after_damage`、`is_ko` 等列为 actor-forbidden derived answers。也就是说，后来的 production representation 仍可看到“招式需要什么颜色”和“当前附了什么颜色”，但要由网络组合这些事实，而不是直接拿到 deficit/KO 答案。

### Code evidence

- Project/commit：0010 的当前历史实现；0013 semantic foundation 的关键提交 `6f37eedc`（2026-07-26）；0025/0031 语义基线的关键历史提交 `17bfa7b6`、`9c75b7aa`、`39bbb649`、`7b8bc859`、`ad2573c4`（2026-08-02 至 08-05）。
- Early global/entity/option features：`train/0010_alakazam_sota_model/model.py:136-318`，embedding composition 在 `:344-425`。
- Partial semantics：`train/0013_semantic_goal_policy/features/card_semantics.py:35-68,185-218`；`features/compiler.py:83-103,262-331`；`model/variants.py:96-143,292-340`。
- Full engine exporter：`train/0031_rule_faithful_semantic_foundation_pretraining/tools/full_engine_prototype_export.cpp:52-113`（effect/target/condition/values）、`:125-161`（skills/abilities）、`:164-234`（cards）、`:245-264`（attacks）。
- Static prototype encoder：`train/0031_rule_faithful_semantic_foundation_pretraining/model/prototype_encoder.py:116-117,160-164,226-336,385-452`。
- Dynamic state fields：`train/0031_rule_faithful_semantic_foundation_pretraining/contracts/fields.py:32-167`。
- Legal option fields：同文件 `:169-199`。
- Prototype load/flatten：`train/0031_rule_faithful_semantic_foundation_pretraining/domain/prototypes.py:57-135`。
- 0031 字段边界：`train/0031_rule_faithful_semantic_foundation_pretraining/contracts/prototype_field_inventory.json:11-48`；防止派生答案泄漏的测试在 `tests/test_features.py:147-160`。
- Ogerpon representation test：`train/0025_semantic_foundation_pretraining/tests/test_semantic_contract.py:96-101`。对 attack 120，Lightning attachment 的 typed deficit 为 3，Grass attachment 为 2（归一化后分别 0.3/0.2）。

最适合公开解释的 representative static semantic features 是：

1. card category、Pokemon type、Energy type、Trainer subtype；
2. HP、stage、`evolves_from` 与 rule flags（ex/Tera 等）；
3. attack base damage、attack flags；
4. attack 的 ordered Energy requirement 及颜色；
5. retreat cost、weakness 与 resistance；
6. Ability/skill 的 once-per-turn、self-KO 等 flags；
7. effect 的 target player/area、condition/comparator、numeric values；
8. linked skill/attack/effect references，使 prototype 不只是独立标量表。

三类输入必须分开表述：

| Layer | Examples | Evidence |
|---|---|---|
| Card/prototype static | card type、HP、stage/evolution、attack damage/cost/colors、retreat、weakness/resistance、skill/effect graph | `prototype_encoder.py:226-336` |
| Dynamic state | owner/zone/slot、current HP/damage、attached Energy units、tools/pre-evolution、turn flags、resource beliefs、events | `contracts/fields.py:32-167` |
| Legal option/action | action type、source/target、attack ID、context/effect card、slot/serial/energy/tool/number/count | `contracts/fields.py:169-199` |

Card ID embedding 在后期**仍然保留**。`PrototypeEncoder` 将 identity embedding、typed categorical/numeric semantic projections，以及 linked skill/attack/effect representations 合成 card prototype；它不是用 semantics 替换 ID，而是让 identity 与 mechanics 并存。

### Technical interpretation

Ogerpon + wrong Energy 的核心 representation 问题在 0010 中确实存在：模型能识别“这是 Ogerpon/Lightning Energy/某个 attach option”，却没有静态输入告诉它该 attack 需要哪些 Energy colors。0025 的实现把 attack cost 与 attached energy types 同时编译进 option representation，甚至短暂提供了 typed deficit；0031 则保留原始规则因子而禁止直接派生答案。因此新增特征让“Lightning 不满足 Grass requirement”成为可由网络计算的关系，而不必仅靠 card-ID 共现重新发现。

代码**不能证明**这些特征已经消除具体 replay 错误。现存单元测试证明字段和数值区分存在，不是行为层面的 before/after official-engine evaluation。审计也没有定位到可验证该 Ogerpon anecdote 的原始 replay ID，因此公开时应把它称为观察到的 failure example，而不是由当前仓库独立复现的结果。

### What our memory got wrong

- “早期只有数值和 card-ID embedding”过度简化：早期已有丰富的动态结构、实体关系和 legal-option features。
- “后来才有任何 semantics”不准确：0013 已有 CSV/text-derived partial semantics；0025/0031 才是 engine-structured prototype 的实质升级。
- “后期直接给模型 energy deficit/KO 答案”只适用于 0025 的阶段性实现；0031 明确禁掉了这些派生答案。
- “Card ID 被 semantic representation 替代”错误；二者被组合使用。
- “新 representation 修好了 Ogerpon 行为”证据不足；它只证明修复了 representational blindness。

### Safe public wording

早期模型并非只看 card ID：它已经编码了区域、伤害、附能数量、回合状态和 legal option 等动态结构，但卡牌的静态 mechanics 主要仍要从身份共现中学习。我们随后先加入了基于卡表和文本的部分语义，再构建只读 exporter，把 official engine 中的 card、attack、skill 和 effect 结构编译成 prototype features。后期表示同时保留 card-ID embedding 与 HP、进化、招式伤害、能量颜色需求、撤退费及 effect target/condition 等语义。这样，像“某个招式需要 Grass，而当前附着的是 Lightning”不再是只能从 ID 频率猜测的关系；但仓库证据只能证明该关系变得可表示，不能单凭此断言具体错误已被完全消除。

### Confidence

**High** for feature chronology and composition：有实现、field inventory 和 tests 交叉验证。**Medium** for Ogerpon anecdote：可验证 representation contract，但找不到原始 replay 与行为级消融。

## 2. Value / critic

### Historical reconstruction

**Before：0017 最早可工作的 PPO 从一开始就有 critic。** `ActorCritic` 包含 `LayerNorm -> Linear -> GELU -> Linear -> Tanh` 的 scalar value head，末层 zero-init。actor backbone 被冻结，初始 trainable policy capacity 是 action decoder；value calibration 独立训练 value head。rollout reward 是 win `+1`、draw `0`、loss `-1`。

初始 calibration 在一份固定 rollout batch 上做 4 epochs、episode-weighted MSE，target 是每个 transition 所属 episode 的 terminal return。随后 PPO 同时优化 decoder 与 value head。GAE 采用 `gamma=1.0`、`lambda=0.95`：非终局即时 reward 为 0，终局 bootstrap 为 0；中间状态通过 `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)` 和递推 GAE 得到 advantage/return。因此它既有 bootstrapping，也有 GAE；value 不是冻结的，而是在 PPO 中在线更新。

**“第一轮 rollout value calibration 过拟合”的实际对应。** V4 calibration batch 有 39,820 decisions，34,748 来自 loss、5,072 来自 win；训练后 value loss `0.3813341`、explained variance `0.1264989`，prediction mean `-0.736768`，loss-state mean `-0.766079`，win-state mean `-0.535961`。该 batch 同时用于拟合与报告，没有 held-out calibration split。它说明 critic 在单一且严重 loss-skewed 的早期 batch 上只学到有限区分，而且之后立刻被带入 PPO；但这不足以统计证明“过拟合”。更准确的说法是 **single-batch, weakly validated, poorly discriminative initialization**。

**Change：0036 dedicated value pretraining。** 0036 冻结完整 semantic encoder，在 official replay states 上训练独立 latent-query value decoder。8 个 learned queries 经过两层 cross-attention/self-attention/FFN；query 0 输出 win probability/value，query 1 输出 opponent archetype，query 2 输出 final prize differential class。主 value 是 `2*sigmoid(logit)-1`。targets 是最终胜负、对手 archetype 和离散 final prize difference `[-6,6]`；loss 是 BCE 加两个 weighted cross-entropy。它不是在 BC imitation loss 中附带训练的 head，更准确名称是 **offline replay-pretrained multi-task critic/value network**。

**After：0037 PPO 使用 pretrained value 并继续更新。** 0037 strict-load 0036 V2 epoch-5 checkpoint。PPO 中 encoder base 仍冻结，但 value queries/blocks/final norm/scalar value head 可训练；archetype 与 final-prize-diff auxiliary heads 冻结。policy 和 value 共用 semantic state/options，且当时 shared Option LoRA/LN 也在 optimizer 内，所以 total loss 的 value gradient 会实际流入这些 shared adapters；并非“只有 value head 更新”。

**Reward chronology。** 0017 与 0037 的环境 policy reward 是 terminal win/draw/loss。0038 起新增 directional prize delta：每步按 focal/opponent prizes taken 的方向变化，以 `1/24` 缩放，单独算 prize GAE、标准化后以固定 `0.1` 权重加入 actor advantage，同时训练独立 `V_prize` MSE。故“我们始终只优化最终胜负、从未 reward shaping”与代码冲突。必须区分 0036 的 final-prize-difference **prediction target** 与 0038 的 prize-delta **policy shaping signal**：前者本身不是 RL reward，后者确实改变 policy gradient。

### Code evidence

- Initial PPO commit：`1321d73c`（2026-07-28，0017 terminal-reward RL framework）。
- Critic architecture/trainables：`train/0017_dragapult_terminal_rl/policy/actor_critic.py:57-103`。
- Fixed-batch value calibration：`train/0017_dragapult_terminal_rl/training/value.py:47-166`。
- Calibration log：`rl_runs/0017_dragapult_terminal_rl/versions/V4_value_calibration_512/artifact/training_summary.json:2-4,88-124`。
- GAE/returns：`train/0017_dragapult_terminal_rl/training/batch.py:65-106`。
- PPO optimizer/loss：`train/0017_dragapult_terminal_rl/training/ppo.py:52-62,116-147`。
- Terminal reward：`train/0017_dragapult_terminal_rl/rollout/worker.py:21-24`。
- 0017 chronology/results：`experiments/0017_dragapult_terminal_rl/DESIGN.md:162-193,291-382`。V9 从约 9.4% 升至峰值 18.9%、结束 17.85%；V13 峰值 20.6%、结束 17.9%，但记录本身把后者解释为 policy drift，而不是 critic failure。formal BC/RL 记录为 41/260 与 68/260。
- Dedicated value commit：`e81921fb`（2026-08-08）。
- 0036 heads/decoder/frozen encoder：`train/0036_dedicated_action_value_network/model/value_network.py:16-150`。
- Targets：`train/0036_dedicated_action_value_network/data/catalog.py:135-174`。
- Objective：`train/0036_dedicated_action_value_network/training/objective.py:38-57`。
- Design/results：`experiments/0036_dedicated_action_value_network/DESIGN.md:11-96`；V2 epoch 5 BCE `0.5267365`，但没有与 RL strength 的因果对照。
- PPO integration commit：`68132076`（0037）；checkpoint load/trainables 在 `policy/actor_critic.py:23-26,64-84,131-150`，optimizer 与 total backward 在 `training/ppo_full_semantic.py:83-106,196-246`。
- Prize shaping：`train/0038_action_boundary_rl/integrated/prize.py:42-111`；`training/ppo_full_semantic.py:346-351,388-418`；0040 design 也明确保留该 objective，见 `experiments/0040_dragapult_0809_action_boundary_rl/DESIGN.md:35-40`。

### Technical interpretation

这次演进不是“从没有 critic 到有 critic”，而是从 **zero-init scalar critic + 单一早期 on-policy batch calibration**，升级到 **冻结 semantic encoder 上的多任务 replay pretraining，再在 PPO 中继续更新 critic trunk/head 与部分共享 adaptation**。早期 GAE 实现本身是标准的 terminal-reward credit assignment 形式；问题更多在 initialization/data coverage/validation 与 shared representation，而不是“没有 bootstrapping/GAE”。

现有记录支持“PPO 能快速改善、也可能在之后回落”，但不能证明回落由 critic calibration 导致。V13 在 policy performance 回落时仍有健康的 EV/KL 诊断，历史设计文档也归因于 policy drift。0036 之后没有一个只改变 critic pretraining、其他变量完全相同的 official-engine controlled ablation，因此不能把后续提升因果归功于 pretrained critic。

Dusknoir self-KO 对 prize proxy 的潜在错误激励在游戏逻辑上合理，但本审计没有找到一份历史 experiment decision 把它记录为 0038 前后 reward 选择的直接因果。更重要的是，实际代码后来确实把 prize advantage 加进 actor objective，所以不能用这段动机反向描述全部历史实现。

### What our memory got wrong

- 最早 PPO 并非没有 value network；critic 从 0017 第一版就存在、先校准、后在线更新。
- “对第一轮 rollout 过拟合”最多是部分准确的口语。仓库能证明单 batch、无 holdout、严重类别偏斜与低 EV，不能证明 generalization gap 意义上的 overfitting。
- “BC-pretrained value head”名称不严谨。0036 用 official replay state/final outcome 做 multi-task offline value training，不是 action-imitation BC loss。
- pretrained value 在 PPO 中不是冻结常量；value decoder/head 继续更新，0037 的 shared Option adapters 也接收 value gradient。
- “全程只用 final win/loss reward”错误。0038/0040/0045 均保留了 prize advantage 对 actor 的 shaping contribution。
- prize-difference auxiliary prediction 与 RL reward 不是同一件事；但 0038 又确实存在独立的 prize shaping，因此两边都不能混写。
- 现有日志不能证明 critic 改造是训练恶化被解决的唯一或主要原因。

### Safe public wording

我们最早的 PPO 已经包含 scalar critic、terminal win/draw/loss return、bootstrapping 和 GAE；问题不是“没有 value network”。它先在首批固定 rollout 上做几轮 value calibration，再与 decoder 一起在线更新，但这份初始化数据高度偏向失败局、没有 held-out calibration，explained variance 也很低，因此更准确地说是一个覆盖有限、验证不足的 critic 初始化。随后我们在 official replay states 上预训练了一个多任务 critic：主 head 预测最终胜负，辅助 heads 预测对手 archetype 与最终 prize difference，再将其带入 PPO 继续训练。需要同时承认，早期阶段主要使用 terminal outcome，而 0038 以后 actor objective 还加入过低权重的 directional prize advantage；final-prize prediction target 本身不等于 reward，但后来的 prize advantage 确实属于 reward shaping。现有日志显示 policy 会先改善后回落，却不能以受控实验把这种回落归因于 critic 一项。

### Confidence

**High** for architecture、targets、GAE、optimizer 和 reward path；代码与 configs 一致。**Medium** for historical causal interpretation；缺少 critic-only controlled ablation 和 held-out calibration evidence。

## 3. LoRA / gradients

### Historical reconstruction

#### A. LoRA chronology

| Stage | Placement/config | What changed | Evidence |
|---|---|---|---|
| 0037 initial broad adaptation (`096d388e`, 2026-08-08) | State board layers 0/1/3；merged self-attn Q/K/V + O；FFN linear1/2；r8, alpha16；可选全 State/Option LayerNorm | 早期 encoder-side broad adaptation | historical `train/0037.../policy/adaptation.py`; design V4 |
| 0037 narrowed (`574c8719`, 2026-08-08) | final Option block 1，self/cross attention **Q/V only**，K 为 0；r4, alpha8；可选 final Option output norm | 从 broad State adaptation 收缩到 10,240-param Option LoRA + 640 norm | 0037 adaptation/model audit |
| 0042 (`099d8356`, 2026-08-11) | 删除 0040 inherited final Option Q/V LoRA 与 local Option LN | encoder 完全冻结；新增 value/policy strategy adapters；显式 detach critic context | `experiments/0042.../DECISIONS.md:5-11` |
| 0044 (`1219a04a`, 2026-08-14) | policy-only final Option self/cross Q/V，r4, alpha8 | immutable value branch 与 policy-LoRA branch 在 final block 前分叉 | `policy/option_policy_lora.py:12-96`; tests |
| 0045 V13 (`5eb70dda`, 2026-08-16) | State final board Q/V/O、event Q/V/O、family-fusion Linear 0/2；final Option self/cross Q/V/O；均 r16, alpha16 | 重新开放 shared State 与更宽 Option attention adaptation | `DESIGN.md:71-94`; `shared_encoder_lora.py:57-110` |
| 0045 V19（当前未提交 worktree） | 加 final Option FFN LoRA r16、final State board FFN LoRA r16、final Option LayerNorm | 在 V17 U125 上零 delta 扩展；分别 40,960、40,960、640 params | `DESIGN.md:222-226`; `policy/ffn_lora_expansion.py` |

0037 broad version 对 merged `in_proj_weight` 施加 LoRA，因此 Q/K/V 都可变；narrowed 版本明确只构造 Q/V delta。0045 V13 的 State self-attention 同样是 merged Q/V + separate O，不更新 K；Option adapter 的 expanded profile 为 Q/V/O。

#### B. Actual gradient paths

**0037：** policy loss 经 semantic state/options、shared final Option LoRA/LN、decoder；value loss 经同一 state/options、同一 shared Option LoRA/LN、value decoder/head。optimizer 同时持有 decoder、value parameters 与 LoRA/LN，`total_loss.backward()` 一次执行，没有在 shared options 前 detach。因此 value loss **实际可以**更新 shared Option LoRA/LN。

**0042：** encoders 与 inherited LoRA 被移除/冻结。value path 先生成 q1/meta/value；`StrategyAdapters` 对 q1、meta logits/value 做 detach，再形成 policy strategy context。policy loss 可更新 policy adapter/decoder，但不能沿 strategy context 回到 value side；value loss也没有 encoder adapter 可更新。这里的隔离来自删除 shared adapter **加上**显式 detach/独立 optimizer ownership，而不只是“拿掉 LoRA”。

**0044：** shared immutable Option prefix 后，value 使用不含 LoRA 的 final block，policy 使用 functional policy-only LoRA final block。测试直接验证 policy loss 使 LoRA grads 非零，而 value loss 不产生 LoRA grad；mutating LoRA 改变 policy logits、不改变 value。

**0045 V13/V14+：** StateEncoder LoRA 位于 policy/value 共用 state path，设计明确写明它收到两类 gradient；policy-only Option LoRA/FFN/LN 只在 policy branch，value branch不经过。Critic output 本身没有连到 Actor logits。V14 在线训练确实更新这些 State LoRA，但 checkpoint writer 漏存 16 个 A/B tensors；V16 起完整保存并记录首步 16/16 均改变。

#### C. “为了隔离 gradient 把 LoRA 拿掉”

这个总结**方向部分准确，但遗漏关键事实**。0042 删除的是 0040 inherited final Option Q/V LoRA 与 local Option LN；剩余 policy trainables 是 action decoder、allocation head 和新 policy strategy adapter等，critic 有自己的 value trunk/adapter/prize head。它同时明确在 q1/meta/value context 处 detach，因此实现并非单纯靠牺牲 encoder adaptation 来隔离。

更准确的 retrospective 是：团队当时选择了一个容易审计的完全冻结 encoder 边界，并在 critic-to-policy strategy context 上加显式 detach；代价是失去 encoder-side RL adaptation。0044 随后证明可用 branch-specific functional LoRA 在保留 isolation 的同时恢复 Option adaptation，0045 再有意识地让 State adaptation 成为 shared policy/value path。不能说“更好的显式 gradient control 当时完全没做”，因为 0042 已经做了；可以说当时没有把这种控制扩展为 encoder 内的 branch-specific adaptation。

#### D. Placement and training speed

early-layer LoRA 参数少不代表 backward 便宜。只要早期 layer 的 A/B trainable，autograd 就必须保存该点及其后续计算所需 activation，并把梯度穿过后续 encoder graph 回传到 adapter；base weights 即使 `requires_grad=False`，中间算子也仍参与反向传播。decoder-only 或 last-block functional LoRA 的 backward frontier 更短，显存和计算都可能显著较小。

0037 broad V4 是直接证据：它触及 board layers 0/1/3、FFN 与 norm，记录达到约 15.9/16.3 GiB，并在约 1,279 秒后因 wall budget 中断，未完成可比较的正常 update series。这个记录支持“明显更慢、更耗内存”，但没有同机同 batch 的完整 benchmark，所以**不能公开写具体慢了多少倍**。

#### E. Did encoder adaptation help?

- **Strong evidence：gradient ownership/adapter activity。** 0044 tests 证明 branch isolation；0045 V16 U105->U106 记录 16/16 State LoRA tensors 改变，combined L2 update norm `0.16736956`。这证明它们被训练，不证明 strength gain。
- **Suggestive evidence：** 0037 narrowed LoRA run 的 fixed evaluation 从 U0 `287-225`（56.05%）到 U30 `298-214`（58.20%）、U32 `305-207`（59.57%），但同时 decoder/value 也在更新。0045 V17 U125 达 `56.0547%`，V19 若干非周期点 U151/U163 分别 `55.2734%`/`55.0781%`，而 U175 为 `53.5156%`；没有单调提升。
- **Invalid as full-State-LoRA evidence：** V14 U105 `54.1016%` 和 V15 router `57.6172%` 均来自漏存 16 State LoRA tensors 的 reconstructable subset，不能代表在线 full adapter。
- **Retrospective hypothesis only：** “decoder-only 遇到 representation ceiling”“最后几天 encoder adaptation 是明显改善的根因”。它们与设计动机一致，但无 clean LoRA-off/on、same-seed、same-checkpoint、only-one-variable official-engine ablation。

### Code evidence

- 0037 commits：`096d388e` broad adaptation；`574c8719` narrowed final Option Q/V。
- 0037 shared gradient graph：`train/0037_dragapult_value_initialized_rl/policy/actor_critic.py:131-150`；`training/ppo_full_semantic.py:83-106,196-246`。
- 0042 removal rationale：`experiments/0042_full_model_design/DECISIONS.md:5-11`；trainable boundary `train/0042_full_model_design/policy/actor_critic.py:143-145,229-270`；detach `policy/strategy_adapters.py:13-48`。
- 0044 branch-specific LoRA：`train/0044_g2_dragapult_policy_option_lora/policy/option_policy_lora.py:12-96`；fork `policy/actor_critic.py:182-199`；gradient/mutation tests `tests/test_policy_only_option_lora.py:33-69`。
- 0045 V13 inventory：`experiments/0045_single_deck_expert_minimal_lora/DESIGN.md:71-94`；`train/.../policy/shared_encoder_lora.py:57-110`。
- 0045 checkpoint omission/recovery：`experiments/0045_single_deck_expert_minimal_lora/decisions.md:41-47`；`DESIGN.md:196-216`。这些是当前未提交 worktree 证据。
- 0045 V19 expansion：`DESIGN.md:222-226`；`decisions.md:39-40`，同属当前 worktree。

### Technical interpretation

这里真正演进的是 **adaptation placement 与 gradient ownership contract**。0037 把 adapter 放在共享 representation 上，policy/value 自然耦合；0042 用 frozen encoder + detached strategy context 得到简单可审计的隔离，但缩小了 policy representation capacity；0044 用分支化 final-block LoRA 同时得到隔离与局部 adaptation；0045 又显式选择 shared State adaptation，让 policy/value 共同塑造状态表示，并把 Option adaptation 保持在 policy branch。

训练速度应从 backward graph 深度和 activation retention 解释，而不能只看 trainable parameter count。强度方面，日志最多支持“值得继续探索”，不支持单因素因果结论。

### What our memory got wrong

- 早期并非始终 decoder-only；0037 第一版已经尝试过 broad State LoRA/FFN/LN，只是成本过高后迅速收缩。
- “为了隔离只把 LoRA 删除”不完整；0042 同时有显式 detach 与分开的 strategy adapters。
- “value 从未影响 policy-side/shared LoRA”错误：0037 shared Option LoRA 与 0045 shared State LoRA 都可接收 value gradient。
- “LoRA 参数少，所以训练应接近 decoder-only 速度”错误；backward frontier 才是主要计算边界。
- “重新开放 encoder adaptation 后显著改善已被证明”证据不足，尤其 V14 checkpoint omission 使一批看似相关的 evaluation 无法重建在线完整模型。

### Safe public wording

我们最初倾向把 pretrained semantic encoder 当作通用表示，只更新 decoder，但实际实验很快触及了 adaptation placement 问题。0037 曾把 LoRA 放进较早的 StateEncoder attention/FFN，随后因显存和 wall-time 成本收缩到 final Option block；这种成本来自必须保留并反传后续 encoder graph，而不是 LoRA 参数量本身。之后我们为了建立清晰的 policy/value ownership，冻结 encoder、移除 shared Option adapters，并在 critic-derived context 上显式 detach；0044 又用 policy-only final-block LoRA 恢复了隔离的 Option adaptation。0045 最后重新开放 shared State attention/fusion LoRA，并进一步试验 FFN 与 LayerNorm，但现有 evaluation 同时受多变量变化和一次 checkpoint omission 影响，只能说结果支持继续适配 encoder，不能声称它被严格消融证明为最终提升的唯一原因。

### Claims we should NOT make

- 不要说“最早所有 RL 都只训练 decoder”。
- 不要说“value loss 从未进入任何 policy/shared LoRA”。
- 不要说“0042 完全没有显式 gradient control，只是粗暴删 LoRA”。
- 不要给 early-layer LoRA slowdown 编造倍数。
- 不要把 V14/V15 evaluation 当作完整 State-LoRA policy 的结果。
- 不要把后期曲线改善写成 encoder adaptation 的 clean causal proof。

### Confidence

**High** for module placement、rank/alpha、detach 和 autograd ownership；实现与 tests 足以判断。**Medium** for performance attribution；日志不是单变量消融，且 0045 后期证据尚未提交并包含已知 checkpoint defect。

## 4. Frozen opponent identity

### Historical reconstruction

**发生过真实 stitched policy。** 0040 V2 CUDA resident path 先用 focal 0809 actor 构建 `Semantic0031DeviceAdapter`，计算 0809 prototype encoder、state encoder、option input encoder 与 option transformer layer 0，再接入加载的 0806 option layer 1、final norm 和 action decoder。它因此是 `Hybrid(0809 trunk, 0806 head)`，不是完整 Frozen-0806。

**受影响路径。** CPU collector 直接调用 complete 0806 model；旧 CUDA resident rollout 与 CUDA Frozen-2048 evaluation 调用 hybrid。故同一 requested opponent identity 在 CPU/CUDA materialization 上不一致。V2 update 250、自动标为 leader 的 update 270、update 280，以及相应 `champion.json`、`status.json`、`training_summary.json` 和 `frozen_results/` 均被正式标为 invalid as Frozen-0806 evidence。

**修复。** `7a525953` / `1ee74ca3` 引入 full-policy registry/resolver：先解析 requested `Policy-0806`，校验 checkpoint 与每个 effective component，再给完整 0806 建自己的 CUDA adapter；缺失或 mismatch 在 inference 前 hard fail。协议将 policy identity 定义为 prototype/state/option encoders、全部 transformer layers、norm、LoRA/delta、decoder 及 schema/model metadata 的完整 effective identity，不能只看 decoder/head。

**验证。** repaired lane-1 greedy CPU/CUDA Full-0806 lockstep 用四个历史 divergence seeds 通过，轨迹分别一致 181、210、158、213 decisions，effective identity SHA-256 为 `94a9aff6...d5b`。这证明 CPU/CUDA action trajectory parity，不是 2048-game strength evaluation。canonical protocol 进一步要求 candidate FP32 source -> FP16 portable storage -> strict FP32 runtime 的 deployment-effective hash 与 focal/opponent storage isolation。

### Code evidence

- Launch/fix chronology：`106885a6`（2026-08-10 launch）、`7a525953`（2026-08-11 immutable identity）、`1ee74ca3`（canonical protocol）、`89b0cff4`（U270 Policy-0809 cross-deck results）。
- Root-cause audit：`experiments/0040_dragapult_0809_action_boundary_rl/POLICY_IDENTITY_AUDIT_2026-08-11.md:3-28`。
- Affected artifacts：同文件 `:11-23`。
- CPU/CUDA lockstep：同文件 `:30-41`。
- Design acknowledgement：`experiments/0040_dragapult_0809_action_boundary_rl/DESIGN.md:44-48`。
- Canonical definition：`docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md:55-115`；historical hybrid classification 在 `:760-805`。
- Invalid “67%”：`rl_runs/0040_dragapult_0809_action_boundary_rl/versions/V2_snapshot_loader_fix_long_run/artifact/champion.json:8` 为 `0.67822265625`，即 1389/2048 的 invalid hybrid Frozen result。
- Later “64%”：`rl_runs/0040_dragapult_0809_action_boundary_rl/versions/V6_u270_benchmark_v2_policy0809_engine2_cuda2048/artifact/report.json:48937-48938` 为 1318/2048 = `0.6435546875`，对手是完整 Policy-0809，合同也不同。
- U270 sampled training rollout 实际是 source policy U269 的 305/512 = `59.5703125%`，不是 67%。

### Technical interpretation

这是 policy identity bug，不只是数值精度或 batching bug。相同 architecture/shape、frozen status 或 resident cache 都不能证明两个模块可共享；只有 effective-weight content identity 才能。旧实现把“decoder/head identity”误当作“完整 opponent identity”，从而让 supposedly frozen opponent 随 focal trunk 变化。

“训练曲线约 67%，独立 evaluate 约 64%”的记忆不正确。67.82% 是后来被判无效的 hybrid CUDA Frozen evaluation，不是 stochastic training rollout curve；64.36% 是后来对完整 Policy-0809、不同 benchmark contract 的结果。两者可以作为 bug discovery 前后历史记录，不能作为同合同 generalization gap。

### What our memory got wrong

- stitched composition 的记忆基本正确，但应精确写为 0809 prototype/state/option-input/layer-0 trunk + 0806 layer-1/norm/decoder。
- CPU path 没有同样 stitched；它使用 complete 0806，问题是 CPU/CUDA resolution 不一致。
- 受影响的不只是 evaluation，也包括 V2 CUDA rollout。
- 67% 不是 training curve，且不能与 64% 做 apples-to-apples 对比。
- 四 seed lockstep 是 parity gate，不是 strength validation；不要把它写成 Frozen-0806 CUDA-2048 复测。

### Safe public wording

我们在 8 月 10 日前后的 0040 CUDA resident runtime 中发现了一个完整 policy identity bug：requested opponent 名为 Frozen-0806，但 CUDA 路径实际复用了 focal 0809 的 prototype/state/Option 前半 trunk，只替换为 0806 的 Option 后半、norm 和 decoder。CPU collector 使用完整 0806，因此两条执行路径甚至对同一 policy ID 有不同 materialization。我们随后把 frozen identity 提升为全 effective model 的内容身份，在 lane routing 前解析并校验所有 encoder、transformer、norm、adapter、decoder 与 schema metadata，任何 mismatch 都 hard fail。修复后的完整 0806 在四个历史 divergence seeds 上通过 CPU/CUDA lockstep；旧 V2 的约 67.8% CUDA Frozen 结果被保留作 provenance，但正式标为无效，不能与后来约 64.4% 的完整 Policy-0809 benchmark 直接比较。

### Confidence

**High**。有专门 root-cause audit、历史 artifacts、canonical protocol、effective hash 与 lockstep output 互相印证。

## 5. Final deck mismatch

### Historical reconstruction

现有 repo 证据与当前记忆的方向相反：

- 0045 V14、V16/V17、V19 的 focal training deck 明确是 **deck 007**，不是 067。V19 checkpoint lineage 继承 V17 U125；各版本 `training_config.json` 都记录 `focal_deck_id: "007"`。
- deck 007 是 2026-08-12 从 0042 league `007_dragapult_ex` 导入的 Top-100/best-rank-1 exact list，canonical hash `07bedfff...e725`，file hash `730515d5...4485`。
- deck 067 是 2026-08-13 append 的 user-provided Dragapult/Munkidori control exact list，canonical hash `7bdb3bb1...ca3`，file hash `de79ebd0...bdcc`。仓库没有把它标成“曾用于模仿人类高手的旧实验”。
- V20 **evaluation/selection** 将 V14/V19 checkpoints 以 focal deck 067 做 Policy-0809 Benchmark-V2 CUDA-2048；最终默认选 V19/U170，1186/2048 = `57.91015625%`。这是 cross-deck evaluation，不是相应 checkpoint 在 067 上的最终训练。
- 2026-08-17 确实存在一个 deck-067 package，目录和 archive 名被标为 `INVALID_067_DO_NOT_SUBMIT`。其 manifest candidate 名仍写 `0045-007-PINHAOLONG_V3...`，但 `deck_id`/hash/deck.csv 均为 067，并明确 `kaggle_submitted: false`。
- 现存根目录 V3/V4 archives 的 manifest 与实际 `deck.csv` 都是 deck 007，且也写 `kaggle_submitted: false`。

所以 repo 能证明的是：**067 参与了最终 checkpoint selection，并曾被装入一个随后明确标为 invalid/do-not-submit 的候选包；可见的最终 V3/V4 包则装的是训练 deck 007。** 它不能证明 invalid 067 包被外部提交，更不能证明最后提交机会因此耗尽。

### Code evidence

- deck 007 identity/provenance：`train/0045_single_deck_expert_minimal_lora/assets/decks/registry.json:716-735`；human-readable list 在 `assets/decks/text/007.txt:1-88`。
- deck 067 identity/provenance：同 registry `:7420-7445`；append source `train/0045_single_deck_expert_minimal_lora/append_user_dragapult_decks.py:20-45`；list 在 `assets/decks/text/067.txt:1-90`。
- Training identity：
  - `rl_runs/.../V14_policy0814_identity_gate_fix/artifact/training_config.json:20-21,86`
  - `rl_runs/.../V17_policy0814_v16_u112_complete_delta_restart/artifact/training_config.json:20-21,87,137-138`
  - `rl_runs/.../V19_policy0814_u125_ffn_lora_norm_expansion/artifact/training_config.json:20-21,90,143-144`
- V20 cross-deck selection：`rl_runs/.../V20_pinhaolong_v3_067_candidate_benchmark_v2/artifact/evaluation_config.json:2-13,21-69`；`status.json:1-12`。
- Invalid constructed package：`.tmp/evaluation/invalid_0045_007_v3_067_package_20260817/package_dir/manifest.json:3,15-20,31,82-89`；其 `deck.csv` SHA-256 与 registry 067 file hash 完全相同。
- Visible V3/V4 archives：两者内嵌 manifest 都写 deck 007、canonical hash `07bedfff...e725`、`kaggle_submitted: false`；archive 中 `deck.csv` SHA-256 都是 `730515d5...4485`。
- Exporter 当前 hard gate 与 write path：`train/0045_single_deck_expert_minimal_lora/evaluation/export_kaggle.py:110-119,187-217`；当前文件属于未提交 worktree。

两套 exact list 的全部差异是：

| Card | deck 007 | deck 067 | Delta 067-007 |
|---|---:|---:|---:|
| Basic Psychic Energy `[5]` | 4 | 3 | -1 |
| Dawn `[1231]` | 1 | 0 | -1 |
| Jamming Tower `[1246]` | 2 | 0 | -2 |
| Moltres `[791]` | 0 | 1 | +1 |
| Team Rocket's Watchtower `[1256]` | 0 | 2 | +2 |
| Risky Ruins `[1260]` | 0 | 1 | +1 |

Moltres 确实只出现在 deck 067；`assets/decks/text/067.txt:42,72` 可直接确认。

### Technical interpretation

这里存在一个真实的 **training/evaluation/package identity confusion risk**，但不能按当前记忆写成“trained/intended 067，accidentally packaged old 007”。训练 checkpoint 的 exact-deck conditioning 是 007；V20 却在 067 上进行最终 shortlist selection；随后先产生了一个 manifest/deck 都为 067、名称仍带 007 的 invalid package，最终可见 archives 又回到 007。

因此最稳妥的公开事实是“最终选择合同与训练/交付 exact deck identity 一度不一致，并被本地 artifact audit 捕获”。至于哪一个是团队主观上 intended deck、哪个 archive 实际提交到 Kaggle、发现时是否已无机会，这需要 Kaggle submission receipt、CLI log 或人工记录；repo 当前没有这些证据。

### What our memory got wrong

- “最后 RL curve 对应 067”错误；V14/V17/V19 training configs 都是 007。
- “最终 packaging 打入旧 007”与现存 artifacts 不符：可见正式 V3/V4 包确实是 007，而被标 invalid 的错配包是 067。
- “067 来自模仿人类高手实验”无证据；registry 只称其为 2026-08-13 user-provided exact list。
- Moltres 存在这一点正确，但它在 067，不在 007。
- checkpoint 没有在 067 上对应最终训练这一点正确；它只被 V20 cross-deck evaluated/selected。
- “实际提交错误包、最后 opportunities 耗尽”无法从 repo 证明；所有找到的相关 manifest 都是 `kaggle_submitted: false`。

### Safe public wording

最终阶段暴露的是 exact-deck identity 管理问题，而现存代码不支持我们最初记忆的方向。V14/V17/V19 checkpoints 明确在 deck 007 上训练；之后 V20 却用另一套 deck 067 对同一批 checkpoints 做最终 CUDA-2048 shortlist selection，067 包含 Moltres、两张 Team Rocket's Watchtower 和一张 Risky Ruins。我们确实构造过一个名称仍带 007、实际 `deck.csv` 为 067 的 package，但它在本地被标为 `INVALID_067_DO_NOT_SUBMIT`；现存 V3/V4 archives 则是 deck 007。没有 Kaggle receipt 或 quota log 时，不应公开断言哪个 archive 实际提交，或错误被发现时提交机会已经耗尽。

### Confidence

**High** for deck contents、hashes、training/evaluation identities 与 local packages。**Low** for external submission history and subjective “intended deck”，因为缺少 Kaggle-side receipt/quota evidence。

# Top 10 technically important corrections for the article

1. 早期模型不是 ID-only；它已有大量动态 state/entity/legal-option features，只是缺少卡牌静态 mechanics。
2. semantic 演进有两个阶段：0013 的 CSV/text-derived partial semantics，之后才是 0025/0031 的 engine-structured ontology。
3. 后期没有移除 card-ID embedding，而是把 identity embedding 与 prototype semantics 组合。
4. Ogerpon 测试证明 attack cost 与 attached Energy color 的差异可表示，不证明该行为错误已被消除。
5. 最早 PPO 从一开始就有 critic、bootstrapping 与 GAE；不是后来才“加 value network”。
6. “第一轮 rollout value 过拟合”应改成“单一、loss-skewed、无 held-out 的弱校准初始化”；现有数据不能证明 overfitting。
7. 0036 是 replay-pretrained multi-task critic，不宜简称 BC value head；final-prize prediction target 也不等于 reward。
8. 0038 以后确实把 prize advantage 加入 actor objective，因此不能声称全程只优化 final win/loss。
9. 0037 shared Option LoRA 和 0045 shared State LoRA 都可能接收 value gradient；0042 既删除 adapter 也做了显式 detach，0044 才建立 policy-only Option branch。
10. Frozen 67.82% 是无效 hybrid result，不是 training curve；最终 deck 证据则显示 checkpoints 训练于 007、V20 选择评测用了 067，仓库无法证明错误包实际提交或机会耗尽。

## Audit limitations

- Git worktree 在审计开始前已存在大量用户修改和未跟踪 artifacts；本报告未改动它们。0045 V14-V20 的若干最关键记录只存在于当前 worktree，未来引用时应保留文件/hash 快照。
- W&B 在线服务没有作为事实源重新查询；本报告使用 repo 内 canonical `training_metrics.jsonl`、status/summary 与已有 W&B export metadata。
- 未运行训练或 full evaluation。autograd 结论来自实际 forward graph、detach、`requires_grad`、optimizer ownership 与已有 gradient tests；没有必要创建新 diagnostic。
- 没有发现可独立验证 Teal Mask Ogerpon anecdote 的原始 replay，也没有发现 0045 最终 Kaggle submission receipt/quota record。两项均应保持 `Unknown / insufficient evidence`。

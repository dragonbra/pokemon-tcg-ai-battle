# Alakazam RL 初版设计

状态：draft implementation

## 1. 参考材料

本框架的搜索/自博弈参考是 Kiyota 的 Kaggle notebook：

- https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code
- `references/kiyotah_notebook_notes.md`

训练监控参考是 Kaggle 的静态 report：

- https://storage.googleapis.com/kaggle-forum-message-attachments/3494938/46495/report.html
- `references/kaggle_training_report.html`
- `references/kaggle_training_report_notes.md`

notebook 使用官方 `battle_start`/`battle_select`/`battle_finish` 和
`search_begin`/`search_step`/`search_end`。它的模型输出 value 和候选 action 分数，
MCTS 使用这些输出搜索未来，再用自博弈胜负和搜索结果训练模型。它不是标准 PPO。

## 2. 初版模型合同

`CandidatePolicyValueNet` 的输入输出固定为：

```text
state_numeric       [B, state_numeric_dim]
state_card_ids      [B, state_token_count]
action_type_ids     [B, max_candidates]
action_card_ids     [B, max_candidates]
action_target_ids   [B, max_candidates]
action_numeric      [B, max_candidates, candidate_numeric_dim]
action_mask         [B, max_candidates]

outputs:
value                [B]                         # 当前玩家视角，[-1, 1]
policy_logits        [B, max_candidates]         # 非法/padding 已 mask
```

模型内部由 card/action embedding、state Transformer encoder、candidate scorer 和
value head 组成。动作候选是当前 simulator 给出的 option，不使用全局固定动作编号。
这使得 reward profile、MCTS 次数和训练算法可以迭代，而不必改变 policy/value 接口。

## 3. 状态和动作

初版 state 特征只使用 observation 可见内容：回合、先后手、双方 Prize/牌库/手牌/
弃牌数量、Active/Bench、HP、能量、Supporter/Energy/Retreat flags、select context
和候选动作上下文。未知对手手牌、牌库顺序和未知 Prize 不能泄漏给模型。

初版 `ptcg_features_v1` 使用 24 个 state token；当前 `ptcg_features_v3` 使用 36 个
numeric features、40 个 state token，并加入最近 32 个 teacher/main action 的摘要，
使用位置 embedding 区分 Active、Bench、手牌和弃牌区域。后续增加特征时应增加 schema
version；不要静默改变已有字段的语义。checkpoint metadata 会自动恢复对应旧 schema。

当前 v5 在 v4 的 effect 上下文之外显式编码 `turnActionCount`、双方 Active/Bench 的
`appearThisTurn` 和新入场计数。这些字段用于学习 V6 已确认的进化时机、攻击终止和
Post-KO 接力条件，不改变合法 option/action contract。第一阶段只学习 main action。
卡牌效果中的多选、目标和数量选择继续由规则 handler
完成；确认主动作模型稳定后，再把 effect selection 加入 candidate dataset。

当前 v4 实验已把 170 局 teacher trace 中的 5,338 条 main 和 3,764 条单选 effect
记录编码到同一 versioned state contract；184 条多选记录继续排除。混合训练会让 main
accuracy 降低，因此 effect-only checkpoint 与 v3 main checkpoint 分离。effect-only
validation accuracy 为 `76.86%`、legal action rate 为 `1.0`，但在运行时以 confidence
0.95 接管 effect 后 34 局为 `22/34 = 64.71%`，confidence 0.99 为 `20/34 = 58.82%`，
均有 error，故 effect 接管保持关闭。

v5 temporal BC 在同一 170 局 teacher trace 上把 validation action accuracy 提高到
`54.22%`（v4-main 为 `51.41%`），但 34 局真实探索只有 `22/34 = 64.71%`、0 error。
这说明 `turnActionCount` 与 `appearThisTurn` 对离线分类有帮助，却没有证明策略强度提升，
因此没有启动 17×10 验收。

## 4. Reward 版本

训练目标必须和 reward profile 一起记录：

```text
reward_v1 = terminal win/loss/draw
reward_v2 = v1 + Prize potential
reward_v3 = v2 + attack readiness + Post-KO relay
reward_v4 = v3 + library risk potential + MCTS targets
```

终局胜负是主目标。中间 shaping 使用状态势能差：

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

`setup_relay`、`post_ko_relay`、`attack_quality` 和 `library_pressure` 首先作为审计
指标，只有经过 reward ablation 后才加入 reward。

## 5. 训练阶段

### Phase A：行为克隆

用现有规则 Alakazam agent、合法动作轨迹和可确认的官方 replay 生成：

```text
(observation, legal candidates, selected candidate)
```

用 masked cross entropy 训练 policy。目标是 100% 合法动作、稳定完成对局和接近规则
baseline，而不是声明找到了最优动作。

当前可运行的数据链是：

```text
官方 cg battle trace
  -> rl.ptcg.build_bc_dataset
  -> ptcg_bc_v1 JSONL
  -> rl.ptcg.train_behavior_cloning
  -> best_validation.pt
  -> PTCGCandidatePolicy.from_checkpoint
```

数据集默认只保留 `select.type=0, context=0` 的主动作，且 teacher 必须返回单个
option index。卡牌效果的多选、目标选择和数量选择仍由规则 handler 处理；这是为了先
验证主动作模型与合法候选 contract，不代表效果层已经被神经网络接管。

`ptcg_bc_v1` 保存编码后的 observation 和当时的候选集合，而不是全局动作编号。因此
同一个 checkpoint 仍然只会在当前 simulator 给出的 legal candidates 上打分。特征
schema、模型 schema 和训练参数会写入 checkpoint metadata；变更字段语义时必须升级
schema version。

### Phase B：Value Model

用完整对局胜负训练 `value(s)`。检查 value MAE、胜率校准和它对攻击路线/断档局面的
排序能力。

`ptcg/calibrate_value.py` 提供冻结 policy 的 value-head calibration：它在 teacher 的
完整轨迹上只更新 `value_head.*`，将终局结果变成可审计的 MAE、相关性和 win/loss
分组均值。对 `alakazam_bc_v5_history` 的校准使 validation value MAE 达到 `0.0663`、
相关性达到 `0.9693`，且 39 组模型参数中只有 6 组 value-head 参数变化。这个 checkpoint
适合作为 PUCT 的叶评估器，但不等于 policy 已经变强。

`ptcg/annotate_transition_returns.py` 还提供 transition-level value target：每局最后一个
己方决策得到 terminal reward，中间决策使用同一 player 视角的可见势能差，并按
`G_t = r_t + gamma * G_(t+1)` 反向累计。数据构建器不能直接拿下一条 raw trace entry
计算势能，因为那一条可能属于 opponent；它会寻找下一条同一 player 的有效决策。由于
model value 合同是 `[-1, 1]`，默认将 shaping 缩放为 `0.1` 后再裁剪 return，并把原始
delta、gamma 和 scale 写入记录供审计。

冻结 `alakazam_bc_v7_main_v4` policy 的 transition-return value 校准在 validation 上
达到 correlation `0.9660`、MAE `0.1209`，且仅 6 个 `value_head.*` 参数变化。把该 value
接入当前有限搜索后 34 局为 `20/34` 且有 1 个 error；单独的 transition-return policy
重加权在探索中为 `26/34`、0 error，但完整 17×10 验收只有 `118/170 = 69.41%`、3 个
engine error，低于 teacher 的 `124/170 = 72.94%`，因此两者都不晋级。

当前已提供一个离线过渡实验：dataset 记录终局胜负，`--outcome-weight` 对 policy loss
做胜负重加权，`--value-loss-weight` 训练 value head。它只用于验证 reward 信号和日志
是否有效，不等同于在线 PPO；整局胜负复制给每个动作的粗粒度方案若不能通过冻结评测，
应退回并改用 transition-level shaping 或 MCTS target。

BC trainer 还支持 `--potential-weight` 的可见势能重加权。一次 `potential_weight=0.5`
消融使 library pressure 从 teacher 的 `1.824` 降到 `1.794`，但 34 局 outcome 只有
`22/34 = 64.71%`，因此不晋级。这个结果说明辅助指标变好并不代表 reward shaping
正确；较小的 `potential_weight=0.1` 也只有 `23/34 = 67.65%`，并伴随 Powerful Hand
下降和 library pressure 上升。当前默认保持 0，后续应使用 transition-level return
和 failure-specific target 做更细的消融。

### Phase A.5：DAgger 分布修正

纯 BC 只覆盖 teacher 访问过的状态；candidate 一旦选错 main action，后续状态可能
离开训练分布。`build_dagger_dataset.py` 在 candidate rollout 的实际状态上调用规则
teacher，重新生成合法 main-action label，再用 `merge_datasets.py` 和原 teacher 数据
合并。effect selection 仍只用于保持 teacher 的 turn memory，不进入当前模型训练。

DAgger 不是 reward 优化，也不能凭训练 loss 判断有效。它必须和原 teacher 数据保持可
审计的混合比例，并经过小规模探索和完整 17×10 evaluation；如果关键 opponent 回归，
立即丢弃该 checkpoint。

### Phase C：MCTS-guided self-play

官方 Search API 负责状态转移，policy 提供动作先验，value 评价叶子节点。先使用很小
的搜索预算，收集搜索后的 action/value target，再训练模型。隐藏信息需要多次
determinization，不能直接照搬 notebook 的固定对手占位卡。

当前 `build_mcts_dataset.py` 使用 `ptcg/mcts.py` 的有限预算 PUCT：官方
`search_begin/search_step` 负责状态转移，policy 输出 prior，value head 评价叶节点，
并把 root 的 visit count 归一化为 `mcts_policy` target。每次 simulation 只展开一个新
leaf，预算不会意外变成完整对局 rollout；同时保存 action value、visit count、先验和
搜索参数，方便审计。多选 effect 的组合数超过预算时只保留模型先验最高的一小组组合，
所以它仍是受控研究 collector，不是完整的效果层搜索。

当前 smoke 已确认官方 SearchState 可以正常释放、root visit 总数等于预算，且多动作
节点会产生非塌缩的访问分布。隐藏牌目前每个样本只做一次 determinization，仍需多次
determinization、self-play 数据和固定评测后，才能把它视为有效策略改进。任何 target
变体仍须先通过小规模探索，再走 17×10 正式评测。

collector 现在支持 `--determinizations N`，会对同一 observation 的多次 root visit
count 做聚合。d4 小实验（4 次 determinization、每次 8 次 simulation、115 条记录）
的 34 局探索为 `23/34 = 67.65%`，低于单次 determinization 分支的 `24/34 =
70.59%`，因此没有进入第二次 17×10 验收。这个参数保留给更大 trace、改进 value
model 或 self-play 数据使用；增加搜索次数本身不能弥补错误的叶评估。

另外保留了可选的 `--teacher-policy-weight`，用于把 trace 中的 teacher action 作为
soft target 锚点。d4 + teacher 0.5 的 34 局探索只有 `19/34 = 55.88%` 且有 1 个
error，明显劣于不加锚点的 d4 分支，因此当前默认保持 0，不把 teacher label 与搜索
value 强行混合。

将 5,338 条 teacher BC 记录与 4 倍 d4 搜索记录混合，并以 0.5 soft-policy weight
训练的 34 局探索为 `23/34 = 67.65%`、1 个 error，同样没有进入正式验收。训练器现已
支持 mixed batch：没有 MCTS target 的记录自动使用 hard label 的 one-hot soft target，
因此该能力可以安全用于后续更大规模的搜索数据，但本轮不晋级。

在 calibrated-PUCT candidate 的 34 局真实 rollout 上，DAgger 采集到 1,033 条新状态，
但 candidate 与规则 teacher 的单选主动作标签一致率为 100%，说明这一批数据主要是
状态覆盖而不是纠错。teacher-v5 + DAgger state coverage 的 BC 分支在探索中为
`26/34 = 76.47%`、0 error，随后 17×10 为 `123/170 = 72.35%`、3 个 engine error；
它接近但没有超过 teacher `124/170 = 72.94%`，故不晋级。将 confidence gate 从 0.85
提高到 0.90 的 34 局探索为 `23/34 = 67.65%`、1 error，提高到 0.95 同样为
`23/34 = 67.65%`、0 error；只说明减少模型接管不能单独解决问题。

第一次真实 PUCT target 分支已完成固定 17×10 验收：`alakazam_mcts_puct_soft_v1`
得到 `115/170 = 67.65%`，teacher `alakazam_v9` 在同一正式协议下为 `124/170 =
72.94%`，并出现 4 个 engine error，因此不晋级。该分支的 Powerful Hand 和
Post-KO relay 诊断指标有所上升，但 attack-quality 惩罚项和正确性恶化；这说明当前
单次 hidden-card determinization、模型 value 叶评估和有限样本 target 仍不足以指导
策略改进。失败分支的报告仍保存在 `rl/runs/evaluation/`，不得用过程指标替代结果护栏。

使用校准 value 重新训练的 `alakazam_mcts_calibrated_soft_v1` 在 17×10 上为
`117/170 = 68.82%`，有 3 个 engine error，仍低于 teacher，继续拒绝晋级。它验证了
value calibration 能让搜索 action value 产生有效排序，但当前搜索样本量和 policy
target 仍不足以安全改写主动作策略。

runtime research candidate 现在还支持 `PTCG_RL_SEARCH_OPPONENT_DECK`：它接受固定的
17-opponent deck card pool，所有对局使用同一个混合先验，不读取 opponent ID。这样比
`[1072] * deck_count` 的占位 determinization 更接近真实隐藏牌分布；但 transition-value
checkpoint 加该 pool 的 34 局探索为 `23/34 = 67.65%`、1 个 error，仍未进入正式验收。
该环境变量默认为空，search 默认关闭。

### Phase D：Masked PPO（可选）

`rl/ptcg/train_ppo.py` 已提供第一版 masked PPO-style terminal reward 微调器。它从
checkpoint 的旧 policy 计算 old log-prob，以终局胜负构造 advantage，并始终在 simulator
候选 mask 内更新。第一版只用于验证 policy/value/reward 接口，rollout 必须来自当前
checkpoint；真正的 transition-level shaping、GAE 和 MCTS target 仍待后续实现。

`rl/ptcg/rewards.py` 的第一版 transition shaping 只使用可见的 Prize race、攻击准备度
和牌库安全势能，按 `gamma * Phi(next) - Phi(current)` 记录分量。每个分量都可通过
`shaping_weight` 做消融，不能因为 shaping 曲线上升就替代 17×10 的 outcome 评测。

当前 milestone 已完成 Phase A 的通用实现、toy 验证和受控 PUCT target collector，但
MCTS 分支尚未通过正式护栏，还没有把任何 RL checkpoint 晋级为正式策略。当前仍没有把 checkpoint 接入正式
`work/<name>/main.py`，也没有把 terminal reward 训练误称为已经完成。正式 submission
需要一个不依赖 PyTorch 的推理封装或可接受的运行时方案，并经过 evaluation 的固定
opponent 矩阵验收后才能建立新的 `work/<name>/`。

## 6. 对手池和冻结评测

训练可以从 opponent pool 采样，但不能把 opponent ID 当作策略捷径。每个 checkpoint
都必须使用固定评测矩阵：

```text
17 个 opponent package
先手/后手
固定 seed
固定 deck
固定 MCTS budget
```

评测分为两个层级：少量对局可以用于快速探索、定位错误和调参，但不能作为晋级证据；
任何阶段性 promotion、仓库正式 evaluation 或提交前结论，最低必须是 catalog 中全部
17 个 opponent、每个至少 10 局，即至少 170 局，并包含固定的先手/后手轮换。若结果接近
目标或方差仍大，再提高到每个 opponent 20 或 30 局。

训练日志与冻结评测分开。晋级条件至少包含：胜率改善、完成率不退化、非法动作率为
零、错误/step-limit 受控、关键 opponent 不出现明显回归。

## 7. 监控和 checkpoint

训练记录使用 JSONL 作为事实源，TensorBoard 作为实时视图，evaluation HTML 作为
冻结评测报告。每个 run 记录 model/reward/feature/action schema、seed、opponent pool、
代码 commit 和 search config。

保存三类 checkpoint：

- `latest.pt`：用于恢复训练。
- `periodic_<step>.pt`：用于回溯阶段变化。
- `best_eval.pt`：只有冻结评测通过护栏才晋级。

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

第一阶段只学习 main action。卡牌效果中的多选、目标和数量选择继续由规则 handler
完成；确认主动作模型稳定后，再把 effect selection 加入 candidate dataset。

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

当前已提供一个离线过渡实验：dataset 记录终局胜负，`--outcome-weight` 对 policy loss
做胜负重加权，`--value-loss-weight` 训练 value head。它只用于验证 reward 信号和日志
是否有效，不等同于在线 PPO；整局胜负复制给每个动作的粗粒度方案若不能通过冻结评测，
应退回并改用 transition-level shaping 或 MCTS target。

### Phase C：MCTS-guided self-play

官方 Search API 负责状态转移，policy 提供动作先验，value 评价叶子节点。先使用很小
的搜索预算，收集搜索后的 action/value target，再训练模型。隐藏信息需要多次
determinization，不能直接照搬 notebook 的固定对手占位卡。

### Phase D：Masked PPO（可选）

`rl/ptcg/train_ppo.py` 已提供第一版 masked PPO-style terminal reward 微调器。它从
checkpoint 的旧 policy 计算 old log-prob，以终局胜负构造 advantage，并始终在 simulator
候选 mask 内更新。第一版只用于验证 policy/value/reward 接口，rollout 必须来自当前
checkpoint；真正的 transition-level shaping、GAE 和 MCTS target 仍待后续实现。

`rl/ptcg/rewards.py` 的第一版 transition shaping 只使用可见的 Prize race、攻击准备度
和牌库安全势能，按 `gamma * Phi(next) - Phi(current)` 记录分量。每个分量都可通过
`shaping_weight` 做消融，不能因为 shaping 曲线上升就替代 17×10 的 outcome 评测。

当前 milestone 只完成 Phase A 的通用实现和 toy 验证，还没有把 checkpoint 接入正式
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

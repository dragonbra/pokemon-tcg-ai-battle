# BC V4 接力在线 PPO：今晚训练实现设计

- 日期：2026-07-24
- 状态：今晚执行设计
- 目标：从已选定 BC V4 初始化，跑通一条真实官方引擎在线 rollout → PPO update 闭环
- 优先级：正确性和可审计性优先；高吞吐优化见独立路线图

配套文档：固定评测改造见
[`../../../evaluation/README.md`](../../../evaluation/README.md)，
未来训练吞吐方案见
[`rl-throughput-optimization-roadmap-20260724.md`](rl-throughput-optimization-roadmap-20260724.md)。

## 1. 今晚的完成定义

今晚的“正常 RL 已跑起来”必须同时完成：

```text
冻结 BC V4 初始化
  -> 官方 engine 真实对局 rollout
  -> acting-time action / old_log_prob / old_value
  -> terminal reward / return / GAE
  -> 同步 PPO update
  -> versioned checkpoint / metrics / TensorBoard
  -> 可打包 candidate
  -> 固定 Evaluation
```

只在静态 BC dataset 上重算 logits、对旧标签做 clipped loss，不能称为在线 PPO。当前
`train/alakazam_bc_rl/training/train_ppo.py` 属于旧的离线过渡实验：它使用 legacy
`CandidatePolicyValueNet`、静态记录和单目标动作，不直接兼容本次 BC V4 的
`FullActionPolicyValueNet`、count head 与完整多选动作合同。历史文件保留用于审计，
今晚新增独立在线入口，不静默改变旧实验语义。

## 2. 冻结初始化来源

当前实际存在并可加载的来源为：

```text
work/yushin_ito_bc_capacity_v4/strategy/model.bin
```

冻结身份：

| 字段 | 值 |
|---|---|
| experiment | `0004-bc_capacity_search_210m/V4_s_d192_l2_lr5e4_s7` |
| model | `FullActionPolicyValueNet` |
| feature schema | `ptcg_features_universal` |
| action contract | `full_action_set_v1` |
| d_model / layers / hidden | `192 / 2 / 384` |
| parameter count | `3,539,395` |
| package checkpoint SHA-256 | `cb2aa2b5c03d9fe0ac9718cd29342799dfaaa9b2a4125f764dab3f093ab81788` |

共享配置记录的原始路径
`rl_runs/checkpoint/0004-bc_capacity_search_210m/V4_s_d192_l2_lr5e4_s7/best_validation.pt`
当前不在本地。因此 RL manifest 必须把 package `model.bin` 作为真实 source checkpoint，
保存其 SHA-256 和 metadata 快照，不能记录一个不存在的源文件。

BC checkpoint 中的 encoder、candidate policy head 和 selection count head 可以直接接力。
value head 虽然存在，但没有进入 BC loss，属于未训练参数，不能把其初始输出当作可靠胜率。

## 3. 今晚非目标

- 不修改 `engine/source/`；
- 不实现 native vector engine；
- 不实现常驻 opponent actor pool；
- 不实现外部动态 GPU inference server；
- 不加入 reward shaping、MCTS 或自博弈 league；
- 不混合多个 expert 的 BC 标签；
- 不根据一次训练 loss、rollout 胜率或离线指标自动晋级模型；
- 不以牺牲 episode/state 隔离换取吞吐。

## 4. 建议代码边界

新增正式在线训练模块，不把训练逻辑放进 `evaluation/`：

```text
rl/
├── rollout/
│   ├── __init__.py
│   ├── action_distribution.py   # 完整动作采样、log_prob、entropy
│   ├── buffer.py                # rollout schema、GAE、minibatch
│   ├── collector.py             # 冻结 policy version，收集官方引擎对局
│   ├── environment.py           # 训练专用官方 runtime episode adapter
│   └── opponent.py              # catalog、固定采样和 episode reset
└── train/
    └── train_online_ppo.py      # 配置、collect/update 循环和产物写入
```

可复用 `train/alakazam_bc_rl/features.py`、`train/alakazam_bc_rl/full_action_model.py`、checkpoint、run、logging 和
storage 工具。Evaluation 只负责训练后固定评测，不作为在线训练器内部组件。

## 5. 环境与 episode 合同

第一版采用正确性优先的独立 episode worker：

- 每个 episode 使用官方 `cg/libcg.so` runtime 运行真实对局；
- 每局开始显式 reset learner history、opponent history、effect serial 和 RNG/session 状态；
- 每局结束必须调用官方 finish 接口；
- 对手来自 `evaluation/configs/opponents.json` 的本地标准 package；
- opponent 按冻结分布采样，同一 episode 不更换 opponent；
- candidate/opponent 先后手按 episode 交替；
- rollout worker 使用冻结的同一个 `policy_version` checkpoint；
- 第一版允许 worker 在 CPU 加载冻结 policy，trainer 在 GPU 更新；避免先实现 GPU server；
- worker 设置 `OMP/MKL/OpenBLAS/NUMEXPR=1`，避免进程乘线程。

训练 rollout 可以在 `/tmp` 保存临时完整 trace；长期 run 只保存紧凑 transition、汇总和
少量失败证据。任何 engine error、非法动作、超时或 reset 异常都要标记该 episode 无效，
不得混入 PPO update。

## 6. 完整动作概率合同

BC 当前解码是 `argmax count + top-k candidate logits`，适合确定性 Evaluation，但不能直接
提供 PPO 探索所需的随机联合动作概率。在线 PPO 使用以下显式分布。

### 6.1 Selection count

1. 从 observation 读取 `minCount`、`maxCount` 和合法 candidate 数；
2. 对 count logits mask，只允许
   `[minCount, min(maxCount, legal_candidate_count)]`；
3. 从 masked categorical 采样 `k`；
4. 保存 count action 和 `log_prob_count`。

### 6.2 Candidate selection

- `k=0`：不选择 candidate，联合概率只有 count 概率；
- `k=1`：在 legal candidate logits 上做 categorical；
- `k>1`：按 logits 做有序、无放回的逐步 categorical 采样；每一步 mask 已选 candidate；
- acting record 保存原始 `sample_order`；
- 默认把 `sample_order` 作为 engine action；只有对应 select context 已验证为无序集合时，
  才允许生成 canonical sorted action，并同时保存两者；
- PPO 重算 log-prob 时必须使用保存的 `sample_order` 和每一步相同的 mask；
- 联合 log-prob 为 count log-prob 与各次无放回选择 log-prob 之和。

这里把采样顺序视为策略内部的 latent action。即使 engine 只消费排序后的集合，PPO ratio
仍针对当时实际采到的 latent action 重算。不能用多标签 BCE 数值冒充联合 log-prob。

### 6.3 推理模式

- rollout：随机采样，可配置 entropy/temperature，但配置必须写入 manifest；
- Evaluation/package：保持确定性的 masked `argmax count + top-k`；
- 两种模式共用 feature schema、模型参数和合法 action mask，不共用 decoder 语义。

## 7. Rollout record

每个 learner decision 至少保存：

```text
experiment_id
version_id
episode_id
opponent_name
candidate_first
policy_version
decision_index
turn / yourIndex
encoded observation tensors
legal action mask
minCount / maxCount
sampled count
sample_order
engine_selected_indices
old_log_prob
old_value
reward
done
terminal_result
valid_episode
```

可以保存 encoded tensor 或可无损重建它们的 observation + frozen codec metadata。第一版
优先保存 encoded 输入，减少 update 时 schema 漂移风险。每批 rollout manifest 记录
feature schema、action contract、source checkpoint hash、policy version、opponent 分布、
随机种子和有效/无效 episode 数。

## 8. Reward、return 与 GAE

第一版只使用终局结果，并始终从 learner 视角定义：

```text
win  = +1
loss = -1
draw =  0
non-terminal reward = 0
```

不加入 Prize、Powerful Hand、Post-KO、牌库压力或攻击质量 shaping。这些指标继续用于
Evaluation 诊断，只有独立 reward ablation 才能进入训练目标。

第一版建议：

- `gamma = 1.0`；
- `gae_lambda = 0.95`；
- terminal transition 不 bootstrap；
- 因超时、worker error 或非法动作中止的 episode 整局丢弃；
- advantage 只在有效 rollout batch 内标准化；
- value target 保存原始 return，value 输出保持 `[-1, 1]`。

## 9. Value head 启动

因为 BC 没有训练 value head，今晚采用可审计的两段启动：

1. 加载 encoder、policy head 和 count head；
2. 将 value head 最终输出初始化为接近 0，manifest 记录 reset 方法；
3. 收集第一批真实 rollout；
4. policy 全冻结，只对 value head 做短暂 warm-up；
5. 检查 value loss、MAE、预测均值和 win/loss 分组；
6. 再打开完整 PPO update。

warm-up 只用于避免随机 value baseline 污染第一批 advantage，不构成策略提升证据。若 value
输出、loss 或梯度出现 NaN/Inf，立即停止，不进入 policy update。

## 10. PPO update

第一版使用同步 PPO：完整 rollout batch 由一个冻结 `policy_version` 产生，collect 期间
禁止 optimizer step；collect 完成后再更新模型。

核心目标：

```text
ratio = exp(new_joint_log_prob - old_joint_log_prob)
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-eps, 1+eps) * advantage))
total_loss = policy_loss
           + value_coef * value_loss
           - entropy_coef * joint_entropy
```

第一轮保守配置：

| 参数 | smoke | 首个正常 iteration |
|---|---:|---:|
| learner decisions | 256–512 | 4096 |
| minibatch | 128 | 256 |
| PPO epochs | 1–2 | 2–4 |
| clip ratio | 0.2 | 0.2 |
| learning rate | `1e-5` | 从 smoke 证据决定 |
| value loss coefficient | 0.5 | 0.5 |
| entropy coefficient | 0.01 | 从实际 entropy 决定 |
| max grad norm | 1.0 | 1.0 |

`learning rate`、epochs 和 entropy 不能根据正式 Evaluation 结果在同一个 version 中反复
试调；每次语义或超参数变化分配新的 `V<n>_<tag>`。

## 11. 指标与 TensorBoard

至少记录：

- rollout decisions/s、episodes/s、有效/无效 episode；
- win/loss/draw、平均 episode decisions、按 opponent 和先后手分解；
- action legal rate、count 范围错误、engine/worker error；
- old log-prob 重算最大误差；
- policy loss、value loss、entropy、approx KL、clip fraction；
- explained variance、value mean/std、return mean/std；
- gradient norm、learning rate；
- policy version、rollout batch id、checkpoint hash；
- collect、encode、update、evaluation wall time。

rollout 胜率用于训练健康观察，不能替代固定 Evaluation，也不能触发自动晋级。

## 12. Run 与产物

先由工具分配新的项目级实验，不能写回 BC004：

```bash
python3 -m rl_environment.runs create yushin_ito_bc_v4_online_ppo \
  --objective "Online PPO initialized from the selected BC V4 checkpoint"
```

假设工具返回 `<experiment>`，首版使用类似 `V1_online_ppo_smoke` 的严格版本名：

```text
rl_runs/<experiment>/V1_online_ppo_smoke/
rl_runs/tensorboard/<experiment>/V1_online_ppo_smoke/
rl_runs/checkpoint/<experiment>/V1_online_ppo_smoke/
rl_runs/<experiment>/evaluation/V1_online_ppo_smoke/
```

启动前检查四个目标目录没有旧文件。失败版本保留 status、错误证据和恢复说明，不覆盖、
不删除、不复用版本号。

## 13. 今晚执行顺序与停止条件

### Gate A：纯概率合同

- count mask 覆盖 0、1、多选和 min=max；
- 无放回采样不重复、永远合法；
- acting log-prob 与 batch 重算误差小于 `1e-5`；
- 极端 logits 下没有 NaN/Inf。

失败则停止，不启动 engine rollout。

### Gate B：官方 engine rollout smoke

- 至少 8 个完整 episode；
- learner/opponent state 每局 reset；
- 100% 动作在 simulator legal options 内；
- 0 engine error、0 worker error；
- terminal result 与 learner reward perspective 一致。

失败则只修复环境/记录合同，不执行 optimizer step。

### Gate C：rollout-only batch

- 收集 256–512 learner decisions；
- collect 期间 policy hash 不变；
- transition、done、return、GAE 数量完全对齐；
- 无效 episode 没有进入 update batch。

### Gate D：一次 PPO update

- 完成 value warm-up；
- loss、KL、entropy、clip fraction 和梯度均为有限值；
- update 后 checkpoint 可重新加载；
- 固定 observation/action 上可以重算 new log-prob；
- package inference 仍只返回合法动作。

### Gate E：固定 Evaluation

```bash
python3 -m evaluation run \
  --candidate work/<rl-candidate> \
  --opponents all \
  --games 10 \
  --workers 8 \
  --worker-cpu-threads 1 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output rl_runs/evaluation/<000N-experiment>/V1_online_ppo_smoke.html
```

必须 180/180 completed、0 error。结果交给用户审核；不因训练指标、一次 rollout 胜率或
单个 opponent 的随机结果自动 promote。

## 14. 必须同步的设计文档

实现过程中同步更新 `train/alakazam_bc_rl/DESIGN.html`，至少包括：

- stochastic full-action distribution 和联合 log-prob；
- rollout record 字段和 tensor shape；
- value reset/warm-up；
- terminal reward、return、GAE；
- PPO loss、policy version 和 collect/update 边界；
- 当前项目阶段从“RL 尚未实现”推进到真实完成的 gate；
- 正式工作包与研究 checkpoint 的边界。

页面只能展示已经通过相应 gate 的能力，不能根据本设计提前把 rollout/PPO 标为已实现。

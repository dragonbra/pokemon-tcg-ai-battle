CUDA 接入完成后，请基于 `0038_action_boundary_rl` 启动今晚的正式 Full-stack PPO 实验。

这次不是 shadow/probe 实验。我们的策略是业务结果优先：把目前已经实现、理论上有正面作用的模块同时启用，先观察全量方案的胜率、Value 健康程度和训练吞吐；如果结果不好，再从共同 update-0 做消融。

## 一、实验起点

实验命名：

```text
V5_canonical_frozen_cuda_fresh_rl
```

必须从公共起点开始：

```text
V3_update0_chance_boundary_fallback
PPO updates = 0
Zero-Shot Policy-0806
pre-RL Value initialization
allocation head 已完成 BC warm start
```

明确要求：

- 不加载 V5 或任何 RL-updated 权重。
- 不从 V5 checkpoint 续训。
- 保留 0037 已验证的 LoRA、Action Decoder 和 Value Decoder 训练结构。
- 保留官方 `engine/source/`、ABI、observation schema 和 `select` 返回协议。
- CUDA 只作为 rollout/evaluation backend 的无损替换，不改变 Agent 对官方引擎的返回形态。

## 二、CUDA 接入

CUDA 的算法正确性已经在 L137 V5 路径验证过，本轮不再安排大规模 CUDA parity 研究。

完成接入后只做必要的集成 smoke：

- backend 能正常创建和关闭；
- 256～512 局零 error；
- legal action、终局、reward、macro unfolding 正常；
- 无内存快速累积；
- checkpoint/evaluation 路径可用。

Smoke 通过后直接启动正式 PPO，不需要再为 CUDA 单独建立实验分支。

请在 run manifest 中记录：

```text
cuda engine version / commit
backend config
worker count
inference batch
rollout batch
games/s
strategic policy forwards/s
```

## 三、训练参数范围

沿用当前简化方案：

```text
Base Encoder                         frozen
Option Encoder 最后一层 LoRA          trainable
Action Decoder                       trainable
Value Decoder                        trainable
Allocation Head                      trainable
LayerNorm                            frozen
```

要求：

- 不扩大 LoRA 覆盖范围。
- 不启用 LayerNorm tuning。
- 不重新调整已经约定的 PPO、Action、Value 和 LoRA 学习率。
- Allocation Head 进入 Actor optimizer，并使用 macro root 与 allocation 的 joint logprob。
- 保存并输出完整 trainable parameter table 和 optimizer parameter groups。

## 四、启用完整 Action Boundary

以下模块全部开启：

```text
DecisionGate                         ON
forced shortcut                      ON
observe-only                         ON
macro trajectory                     ON
Phantom Dive canonical allocation    ON
chance/info boundary                 ON
```

约束：

- forced/internal callback 不得进入 PPO trajectory。
- forced/internal callback 不得进入 policy loss、value loss、GAE 或 entropy。
- Phantom Dive 必须保持一次 macro strategic transition。
- 对官方引擎仍逐步返回原始 `select: list[int]`。
- Allocation joint logprob、old logprob 和 replay contract 必须严格一致。
- fallback 保留 fail-closed，并单独统计。

## 五、启用 Full Reward Profile

正式开启目前已批准并实现的 reward 模块，而不是只做 shadow logging。

主 Value 语义必须保持：

```text
V_win / Query 0 target = terminal win/loss return，范围仍为 ±1
```

Prize 信号使用已经约定的弱 directional objective：

```text
net_prize_delta
= own_prizes_taken - opponent_prizes_taken

effective scale
= 每张净奖赏卡 1/24
```

如果当前实现采用独立 Prize Value/Advantage，则使用：

```text
A_actor = A_win + A_prize_scaled
```

不要把 prize reward 重复缩放两次。

要求分别记录：

```text
terminal return
own prize component
opponent prize component
net prize component
各 component 的 episode sum
A_prize / A_win 的 std ratio
prize/win gradient norm ratio
```

已经实现并通过机会条件检查的其他 reward component 可以按 Full profile 启用；但不要临时加入“第二回合必须攻击”“必须连续进攻”等无条件硬编码奖励。

## 六、启用 Opponent Meta Conditioning

开启当前已经实现的 opponent-aware 模块：

```text
Opponent Archetype Head              ON
Opponent Capability Head             ON（若已实现）
Actor Conditioning                   ON
Value Conditioning                   ON
```

保留以下梯度安全边界：

```text
opponent embedding → Actor/Value      detach = true
meta loss → shared LoRA               false
```

也就是说：

- Actor 和 Value 正常使用 opponent embedding。
- PPO/Value loss 不反向破坏 Meta Head。
- Meta auxiliary loss 暂不进入共享 LoRA。
- Meta Head 只能读取当前时点已经公开的信息。
- 真实 opponent deck/archetype 只能作为监督标签，不能作为 Policy 输入。
- 禁止输入隐藏牌表、seed、环境 deck ID 或未来信息。

Opponent fusion 必须使用零初始化 residual、零门控或其他保持 update-0 行为的等价机制。如果当前实现不能保证这一点，请先报告 Full-stack update-0 相对 V3 的行为差异，再启动 PPO。

## 七、Rollout 与优化预算

正式配置：

```text
rollout_games_per_update = 2048
optimizer_budget_mode = fixed_optimizer_budget
max_updates = 50
checkpoint_every = 1
```

要求：

- 每个 update 使用新采集的完整 on-policy trajectory。
- old logprob、old value、GAE 和 return 在 rollout 后冻结。
- rollout 增大后，从完整样本池均匀采样 optimizer minibatch。
- 不因为 rollout 变大而隐式增加 optimizer step 数量。
- 不硬编码旧的 512 games/update 假设。

每个 update 记录：

```text
games collected
primitive selects
strategic transitions
forced shortcuts
macro transitions
optimizer samples consumed
sample coverage ratio
sample reuse ratio
rollout seconds
PPO seconds
evaluation seconds
end-to-end games/hour
GPU allocated/reserved
CPU RSS
```

以后性能对比同时按 games、strategic transitions 和 wall-clock 报告，不能只比较 minutes/update。

## 八、固定评测面板

正式实验只使用全局 canonical Frozen-0806 面板：

```text
contract = frozen_0806_seeded_2048_v2
evaluation_seed = 341512806
8 replicas × 256 environment-frequency slots = 2048 games
007 schedule SHA-256 = 98b58bced460c1a2e622ae4b39bf506294fcb0230bb42e4117aaa6efc73c9ce9
```

要求：

- 所有 seed 固定并版本化。
- 先后手各半。
- matchup 构成固定。
- 不重复一组 256 seed。
- 256-slot 中每个 opponent 的次数必须与当前环境频率一致，不得改成 55 deck 均匀分布。
- evaluation seed 不得进入训练 rollout。
- 所有 checkpoint 使用同一面板。
- 先对完整 Full-stack update-0 跑 Core 2048，之后才能比较训练增量。
- 评测方式必须与 `policy_0806/.../reports/007_dragapult_ex.html` 的 seed、slot、replica、seat、opponent 权重、greedy 和 CUDA backend 对齐；仅 action contract/model checkpoint 随实验版本变化。

评测频率：

```text
Core 2048：
update 0，之后每 5 updates
```

不再运行 8192 局 Extended 分支，避免不同 checkpoint 使用不同评估规模而削弱可比性。

训练 rollout 可使用不同于 evaluation 的随机 seed，但 `rollout_games_per_update` 必须是 256 的整数倍；每个连续 256-game unit 都必须精确复现 Frozen catalog 的 opponent 频率。unit 内顺序和 seat 可以按训练 seed 随机化，整体保持先后手平衡。

输出：

```text
W-L-D
overall win rate
Wilson 95% CI
first-player / second-player
per-matchup result
paired delta vs full-stack update-0
loss→win / win→loss flips
paired bootstrap CI 或 McNemar
```

不要使用单个最高 checkpoint 作为结论。至少连续两个评测点维持提升，才能称为稳定收益。

## 九、Value 与 PPO 诊断

除总 Value loss 外，必须记录：

```text
value MSE
normalized value MSE = MSE / Var(return)
explained variance
value prediction mean/std
return target mean/std
GAE mean/std
value gradient norm
```

按阶段拆分：

```text
opening
midgame
near-terminal
terminal-adjacent
```

按动作边界拆分：

```text
ordinary strategic root
macro root
forced/internal callback
```

其中 forced/internal callback 的 PPO/Value transition 数量必须为零。

继续保留：

```text
behavior replay MAE
behavior KL
clip fraction
entropy
approx KL
PPO epoch completion
NaN/Inf checks
```

需要特别注意：Raw Value loss 下降不等于 Value 变好。请同时判断 early/midgame Explained Variance，避免再次出现 V5 中“Value loss 下降但 EV 下滑”的情况。

## 十、模块诊断

Action Boundary：

```text
primitive-select / strategic-transition ratio
Phantom Dive focal forward count
macro fallback rate
allocation entropy
allocation KL
```

Reward：

```text
各 reward component episode sum
reward component 与终局胜负的相关性
A_prize / A_win 规模
对应 gradient norm
```

Meta：

```text
archetype accuracy/top-k
capability accuracy
NLL/calibration
按回合阶段的准确率
unknown/uncertainty rate
opponent fusion residual norm
```

这些以聚合标量为主，不要默认保存全量 per-step tensor 或 replay trace，避免重新引入内存和吞吐问题。

## 十一、停止条件

不要因为 update 5 或 update 10 的一次 Frozen 波动自动停止。业务表现原则上至少观察到 update 20。

只在以下技术问题出现时立即暂停：

- NaN/Inf；
- on-policy/replay 合同失败；
- CUDA 或 official action legality 异常；
- error/fallback 显著上升；
- GPU/CPU 内存持续单调累积；
- warm-up 后吞吐连续多个 update 明显下降；
- Value 输出坍缩到零或持续饱和到 ±1；
- early/midgame Value EV 与 policy 指标同时持续恶化。

正常情况下不设置 update 上限，持续运行直到用户明确要求停止。训练进程只在完整 update
边界检查 `artifact/STOP_REQUESTED`，确保不会留下半个 PPO update；上述技术异常仍须立即 fail closed。

## 十二、后续消融原则

这次 Full-stack run 的目标是先看综合业务结果，不要求今晚完成模块归因。

所有模块必须保留独立配置开关，后续消融必须重新从相同的 V3 common update-0 开始，不能简单在 Full-stack 已训练 checkpoint 上关闭模块。

如果 Full-stack 明显变差，优先顺序是：

```text
1. Reward OFF，保留 Atomic + Meta
2. Meta Conditioning OFF，保留 Atomic + Reward
3. 只保留 Atomic + terminal reward
4. 最后才检查 Allocation/Action Boundary 子模块
```

CUDA 不进入消融；它以后视为固定训练基础设施。

如果 Full-stack 明显变好，则保留这条主线继续迭代，后续只做 Reward OFF 和 Meta OFF 两个风险消融，不因学术归因阻碍最佳方案继续训练。

## 十三、启动前回报

CUDA 接入及 smoke 通过后，请先输出一份简短 launch manifest：

- 实际起始 checkpoint；
- PPO updates 是否为 0；
- 所有功能开关最终值；
- trainable parameter table；
- optimizer parameter groups；
- rollout/eval backend；
- rollout games/update；
- optimizer budget；
- reward 数学定义及实际系数；
- Meta 梯度边界；
- 固定评测 seed manifest；
- 正式启动命令；
- 预估 games/hour；正式 run 的完成时间由用户手动停止决定。

确认这些内容与上述合同一致且 GPU 空闲后，启动 `V5_canonical_frozen_cuda_fresh_rl`。失败的 V4 保留为错误均匀面板与启动诊断的审计记录，不得续写或冒充正式可比基线。

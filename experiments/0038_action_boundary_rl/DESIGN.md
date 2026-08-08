# 0038 Action Boundary RL 设计

## 目标和证据边界

0038 的目标是让强化学习时间只推进于真实游戏决策，而不是官方 UI callback。通用官方规则仍以攻击为回合提交；Phantom Dive 的 200 主伤害和后场六个伤害指示物共同属于同一次攻击。当前 engine runtime 事实是六个指示物通过 `context=14`、`remainDamageCounter=6..1` 的六次 primitive `select` 完成。项目策略假设是：此链没有随机、隐藏信息揭示、对手响应或优先权转移时，可作为一个 compound action。

官方 `engine/source/`、ABI、observation、option index 和返回 `list[int]` 均不改。0038 只改 Agent、rollout wrapper、trajectory、附加 head 和训练适配层。

## 零样本基线

- State Encoder、Option Encoder、Root Action Decoder：加载 Policy-0806 epoch 11。
- Value：加载 0036 V2 epoch 5 latent-query value head。
- Conditional allocation head：0038 新增；用同一 Zero-Shot policy 在 legacy sequential contract 下采集的 official-engine 标签做短 BC warm start，之后才允许 PPO。
- optimizer、scheduler、on-policy buffer：不继承。
- GRU 保持 request-local，每个真正的 root request 重新初始化；forced/macro callback 不维护跨请求 hidden。其状态变化通过完整消费 observation/history 进入下一个真实 request。

初始化的权威来源为 `friend_0806_epoch11/model.pt`（Policy-0806）和 0036 V2 epoch-5 pre-RL latent-query Value。0037 V5 或任何 `checkpoint/update-*` 都被拒绝。Last Option self/cross attention merged Q/V LoRA 固定 r/alpha=4/8，A 为 Kaiming、B 为零，因而 update-0 是精确 zero-delta；Action Decoder 全量可训练，原始 Encoder 冻结。Prize/Meta 等辅助模块使用隔离 RNG，关闭模块或 zero conditioner 时不会改变 core/LoRA 初值、root logits、legal mask 或 `V_win`。

## DecisionGate

Gate 位于官方 observation 读取后、tensor compile 前。主分类为 `TERMINAL`、`LEGAL_EMPTY_PASS`、`MASK_ERROR`、`FORCED`、`STRATEGIC`，并可附带 `CHANCE_BOUNDARY`、`INFORMATION_BOUNDARY`、`PRIORITY_TRANSFER` 边界标记。

- 0 options：terminal、合法 `min=max=0` empty pass 或 fail-closed mask error。
- 仅当 canonical completion 为 1 且没有 STOP/cancel/decline/pass 时才 forced。
- forced 路径执行 `observe_only`，更新 knowledge/history/event；不 compile tensor、不调用 Transformer/GRU/Value/服务、不写 policy transition。
- 所有其他情况保持 strategic。未知语义不做 shortcut。

## Phantom Dive compound action

Canonical action：

```text
DragapultDamageAllocation {
  target_ids: stable (player_index, serial, card_id, initial_bench_slot),
  counters: nonnegative integer vector,
  total: 6
}
```

`n` 个目标的分配数是 `C(n+5,6)`，`n=1..5` 分别为 `1,7,28,84,210`。目标以 stable identity canonicalize；allocation 不包含选择顺序，因此没有排列 alias。

第一版的明确支持域是官方 callback 中 `1≤n≤5`。若卡面效果扩展 Bench 使 `n>5`，wrapper 不伪造 210-option mask：在线走 legacy sequential fallback，并把该局 macro trajectory 标为 invalid；数据采集统计并跳过该链。

若攻击者处于混乱状态，官方在 Phantom Dive root commit 后、allocation callback 前揭示硬币结果。v2 protocol adapter 将其视为真实 `CHANCE_BOUNDARY`：不预提交 allocation，在线安全回退 legacy 参数链，并把整局 macro trajectory 标为 invalid。任何未预见的 pending-transaction 漂移同样先清理 transaction、保存 trace，再回退，不终止本可继续的 official game。

概率严格分层：

```text
log P(complete) = log P(root) + log P(allocation | state, Phantom Dive)
```

Root decoder 的 logits、mask 和归一化集合不变。只有 root 已选中 Phantom Dive 后，allocation head 才在最多 210 个 canonical allocations 内归一化。`n=1` 参数 forced，parameter logprob/entropy 均为 0。

allocation head 复用同一次 `EncodedState.summary/cards` 与 root option embedding。每个目标的可见特征宽度为 12：分配数、放置后剩余 HP、立即 KO、可拿 Prize、距 KO 损伤、Bench 位置、当前伤害、附着能量、状态、可见进化风险、HP headroom、分配比例。目标行经共享 MLP 后 sum pooling，并加入集中/分散统计，所以同时置换目标及其分配不会改变 logit。

## 官方协议适配

`MacroPlanner` 在第一次 root 决策生成完整 allocation；`PendingMacroTransaction` 以 battle ID 隔离；`OfficialProtocolExecutor` 在之后六个 callback 中：

1. 正常消费 observation/history；
2. 校验 battle、actor、context、remain count、effect 和 legal mask；
3. 按 serial/card identity 在当前 Bench/legal option 中重新定位；
4. 返回原格式单元素 `list[int]`；
5. 第六次后清理 transaction。

漂移、目标消失、空 mask、超时、terminal 或优先权异常均 fail closed 并使 rollout macro 无效。官方 primitive 调用次数不减少；focal 模型 forward 从 root+六 callback 的 7 次降为 1 次。

## PolicyTrajectory

`EngineEventTrace` 保存所有 observations、primitive selects、forced events、reward/terminal/winner 与边界。`PolicyTransition` 每个真实决策仅一条：pre-action features、canonical macro、root/parameter/joint old logprob、pre-action value、累计 reward/discount、post-commit observation、next value、done、event span 和有效性。

Forced/macro 内部 callback 不产生 Value 样本，不进入 policy loss、entropy、KL、clip fraction 或 PPO ratio，也不额外应用 gamma/lambda。Semi-MDP GAE 使用：

```text
delta_t = accumulated_reward_t + accumulated_discount_t * next_value_t - value_t
A_t = delta_t + accumulated_discount_t * lambda * A_(t+1)
```

terminal 时 `next_value=0`。entropy/KL/clip 的分母仅为 valid strategic transition；因此不可与 V5 历史曲线直接横向比较。

## 模块化集成目标

- `V_win`：Value Query 0 和终局 ±1 target 完全不变。
- `V_prize`：使用 latent Query 3 的小 sidecar；`r_prize=(己方拿奖-对方拿奖)/24`，以 own-turn clock 独立计算 `A_prize`，默认 `gamma_prize=lambda_prize=0.97`。`A_win` 与 `A_prize` 分别标准化，再用固定 `alpha_prize` 组合。支持 off/directional/terminal_neutral。
- `OpponentMetaHead`：只读 deployable public state/history，Frozen deck ID 只作 label。taxonomy v1 含 unknown/ambiguous 与 Frozen catalog archetype；预测分布经 detach 和 zero-initialized conditioner 进入 Action summary，PPO 不反向到 Meta Head。
- Tempo：第一版只有公开 legal Attack opportunity、第二个己方回合攻击、连续攻击 streak 和中断原因等指标；不启用 tempo reward/loss，curriculum 只允许增加自然 opportunity state 的采样。
- Loss Registry：独立登记 policy/value-win、value-prize、opponent-meta、root entropy、allocation entropy 与预留 tempo loss；每项有 flag、weight、optimizer group、日志和 checkpoint metadata。

预设为 BASE、PRIZE、META、PRIZE_META、INTEGRATED。共同 update-0 是 Zero-Shot core + fresh zero-delta LoRA + BC allocation head；optimizer、scheduler、RNG、rollout buffer 和 old-policy snapshot 每个新实验重新建立。

## Rollout 扩容与统计频率

rollout 配置不再假设固定 512 局或固定 decision 数。`engine_backend`、games/workers/envs/batch/inflight/queue/seed shards、PPO physical minibatch、gradient accumulation、epochs 和 optimizer steps 均显式配置。accelerated backend 必须先通过 observation/legal/reward-terminal-winner/RNG/full-game/macro parity，并与 official backend 使用同一 Agent action contract。

V4 使用 `accelerated:cuda_resident`：兼容规则 blob 与 CUDA state machine 驻留 GPU，Agent 仍执行同一 0038 DecisionGate、hierarchical Phantom allocation 和 primitive-select 合同。CUDA collector 只把真实 decision boundary 写入 `PolicyTrajectory`；公开奖赏数和状态位等 allocation 条件特征随 macro action 保存，用于精确重放 joint old-logprob，不引入隐藏字段。任何 actor、目标、计数、chance 或 transaction 漂移都标记 invalid/fallback 并排除 PPO。

扩容有两种显式模式：`fixed_epochs` 会随样本增加 optimizer steps；默认优先 `fixed_optimizer_budget`，固定每 update optimizer steps，通过 effective minibatch、累积或抽样吸收更多 rollout。切换模式必须由用户确认，不能连带静默改变 LR、epochs、effective batch 或 advantage normalization 范围。

指标分三档：每 update 只聚合已有 forward 的低成本 scalar 与七阶段 wall time；Frozen point 从已有 2,048 局逐局结果离线聚合；梯度范数/cosine、allocation margin 和详细 calibration 只在固定小 minibatch 的 update 0/5/10/之后每 10 次运行。exhaustive parity、hidden leakage、RNG/state A/B 和 fault injection 仅在测试/CI。

## Frozen Evaluation

正式评估唯一合同为全局 `frozen_0806_seeded_2048_v2`：evaluation seed `341512806`，8 replicas × 原始 256-slot 环境频率分布，共 2,048 个唯一 engine seed，先后手各 1,024。007 标准 schedule SHA-256 为 `98b58bced460c1a2e622ae4b39bf506294fcb0230bb42e4117aaa6efc73c9ce9`。update 0 与之后每 5 updates 必须复用相同 seed、slot、replica、opponent 权重、seat、greedy、CUDA backend 和 termination contract；相对 update-0 报告 paired flips、McNemar、Wilson CI、seat、matchup 和 replica 方差。历史 `0038_frozen_2048_v1` 是均匀 exact-deck 面板，仅保留 V2/V3 证据，不再用于正式比较。

训练 rollout 不复用 evaluation seed，但以 256 games 为不可拆分频率单位。每个连续 256-game unit 精确包含 Frozen catalog 的 opponent slot counts，unit 内按 `(training seed, source policy update)` 随机排列并平衡 seat；因此 2,048-game update 含 8 个自然分布单位，同时保持 2,048 个独立 engine seed。

2,048 paired games 适合判断约 2pp 的方向与稳定性，但很小差异不能自动宣称显著。按 discordant rate 0.10/0.20/0.30 的近似规划，2pp 约需 1,962/3,923/5,884 局，1pp 约需其四倍；最终检验使用实际 paired flips。

## Checkpoint 与阶段

checkpoint schema 为 `0038_model_only_checkpoint_v1`，强制记录 action、gate、canonicalizer、trajectory 与 official adapter 五个版本。model-only payload 禁止 optimizer/scheduler/RNG/rollout/replay。

公共基线为 `V3_update0_chance_boundary_fallback`，PPO updates=0，official adapter 为 `0038_official_primitive_adapter_v2_chance_fallback`。Frozen 结果为 1308-739-1、0 error、14 safe fallback；Meta 全面板聚合因 logits 容器类型错误未发布，代码已修正，遵循“不得为统计额外 forward”合同没有单独重跑。

CUDA resident Action Boundary、吞吐 benchmark 与真实 CUDA rollout→PPO replay smoke 已完成。V4 在错误的均匀 exact-deck 面板完成 update-0 后，于首个 PPO update 前因 sparse-diagnostic predicate 未导入而 fail closed；没有产生 update-1 model checkpoint，其 `63.77%` 不得与 canonical 007 `57.32%` 比较。V4 保留为失败审计记录。

下一正式版本为严格递增的 `V5_canonical_frozen_cuda_fresh_rl`：仍从 V3 common update-0 加载 BC-warm-started allocation head，启用 `INTEGRATED` preset，并重新初始化 optimizer/scheduler/RNG/rollout/old-policy。每 update 采集 2,048 局、固定 optimizer budget；update 0 和之后每 5 updates 运行 canonical Frozen-0806 2,048，不启用 8,192 分支。run 不设 update 上限，在完整 update 边界响应人工 stop sentinel。GPU 空闲并完成 CPU-only 合同测试前不得启动。
